"""Checkpointed, bounded research workflow reusing the baseline's tools."""

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from threading import Lock
from typing import Any, TypedDict
from uuid import UUID

import psycopg
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    messages_from_dict,
    messages_to_dict,
)
from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from psycopg.rows import dict_row
from pydantic import ValidationError

from app.config import Settings
from app.mcp_client import mcp_research_tools
from app.rag_embeddings import RAGError
from app.rag_models import Evidence
from app.rag_service import RAGService
from app.research_models import (
    GraphResearchRequest,
    GraphResearchRun,
    ResearchDraft,
    ResearchPlan,
)
from app.research_prompts import PLANNING, RESEARCH_SYSTEM
from app.research_service import (
    ModelInvoker,
    ResearchError,
    ResearchService,
    chat_model,
)
from app.sec_client import SECError
from app.sec_service import SECService


class ResearchState(TypedDict):
    # All values are plain JSON data so checkpoints need no custom deserializer.
    question: str
    model: str
    status: str
    plan: dict | None
    messages: list[dict]
    pending_tools: list[dict]
    evidence: dict[str, dict]
    draft: dict | None
    result: dict | None
    tools_called: list[str]
    tool_errors: list[str]
    model_calls: int
    errors: list[dict]


class GraphResearchService(ResearchService):
    def __init__(
        self,
        sec: SECService,
        rag: RAGService,
        settings: Settings,
        *,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        super().__init__(sec, rag, settings)
        # Only tests inject an in-memory saver; application requests use PostgreSQL.
        self.checkpointer = checkpointer
        self._lock = Lock()

    @contextmanager
    def _checkpoint(self, run_id: UUID) -> Iterator[BaseCheckpointSaver]:
        # One inference workflow per local process; fail fast on concurrent requests.
        if not self._lock.acquire(blocking=False):
            raise ResearchError("Another graph request is active; retry later", 409)
        try:
            if self.checkpointer is not None:
                yield self.checkpointer
                return
            try:
                with psycopg.connect(
                    self.settings.database_url.get_secret_value(),
                    autocommit=True,
                    prepare_threshold=0,
                    row_factory=dict_row,
                    connect_timeout=5,
                ) as conn:
                    # Also prevent two processes from advancing the same thread.
                    locked = conn.execute(
                        "SELECT pg_try_advisory_lock("
                        "hashtextextended(%s, 0)) AS locked",
                        (str(run_id),),
                    ).fetchone()
                    if not locked or not locked["locked"]:
                        raise ResearchError("This research run is already active", 409)
                    saver = PostgresSaver(conn)
                    saver.setup()
                    yield saver
            except psycopg.Error as exc:
                raise ResearchError(
                    "Research checkpoint database unavailable; check PostgreSQL", 503
                ) from exc
        finally:
            self._lock.release()

    def run(self, request: GraphResearchRequest) -> GraphResearchRun:
        with self._checkpoint(request.run_id) as saver:
            graph = self.build_graph(saver)
            config = self._config(request.run_id)
            if graph.get_state(config).values:
                raise ResearchError("Research run ID already exists", 409)
            state: ResearchState = {
                "question": request.question,
                "model": self.settings.chat_model,
                "status": "running",
                "plan": None,
                "messages": [],
                "pending_tools": [],
                "evidence": {},
                "draft": None,
                "result": None,
                "tools_called": [],
                "tool_errors": [],
                "model_calls": 0,
                "errors": [],
            }
            graph.invoke(state, config, durability="sync")
            return self._read(graph, request.run_id)

    def get_run(self, run_id: UUID, *, resume: bool = False) -> GraphResearchRun:
        with self._checkpoint(run_id) as saver:
            graph = self.build_graph(saver)
            config = self._config(run_id)
            snapshot = graph.get_state(config)
            if not snapshot.values:
                raise ResearchError("Research run not found", 404)
            if resume and snapshot.next:
                if snapshot.values["model"] != self.settings.chat_model:
                    raise ResearchError(
                        "Restore the run's chat model before resuming", 409
                    )
                graph.invoke(None, config, durability="sync")
            return self._read(graph, run_id)

    def _config(self, run_id: UUID) -> dict:
        return {
            "configurable": {"thread_id": str(run_id)},
            "recursion_limit": 4 * self.settings.research_max_tool_calls + 10,
        }

    def _read(self, graph: CompiledStateGraph, run_id: UUID) -> GraphResearchRun:
        snapshot = graph.get_state(self._config(run_id))
        state = snapshot.values
        return GraphResearchRun(
            run_id=run_id,
            status=state["status"],
            question=state["question"],
            plan=state["plan"],
            result=state["result"],
            errors=state["errors"],
            tools_called=state["tools_called"],
            model_calls=state["model_calls"],
            next_nodes=list(snapshot.next),
        )

    def build_graph(self, saver: BaseCheckpointSaver) -> CompiledStateGraph:
        builder = StateGraph(ResearchState)
        for name, node in (
            ("plan", self._plan),
            ("select_tools", self._select_tools),
            ("execute_tool", self._execute_tool),
            ("synthesize", self._synthesize),
            ("validate", self._validate),
        ):
            builder.add_node(name, self._guard(name, node))
        builder.add_edge(START, "plan")
        for name in ("plan", "select_tools", "execute_tool", "synthesize"):
            builder.add_conditional_edges(name, self._route)
        builder.add_edge("validate", END)
        return builder.compile(checkpointer=saver)

    @staticmethod
    def _route(state: ResearchState) -> str:
        if state["status"] == "failed":
            return END
        if state["draft"] is not None:
            return "validate"
        if state["plan"]["scope"] == "unsupported":
            return "synthesize"
        if state["pending_tools"]:
            return "execute_tool"
        # After tools, ask the model whether more navigation/evidence is needed.
        messages = messages_from_dict(state["messages"])
        if messages and isinstance(messages[-1], AIMessage):
            return "synthesize"
        return "select_tools"

    def _guard(self, stage: str, node: Callable) -> Callable:
        def guarded(state: ResearchState) -> dict:
            calls = state["model_calls"]

            def invoke(
                runnable: Runnable[Any, Any], messages: list[BaseMessage]
            ) -> Any:
                nonlocal calls
                calls += 1
                return self._invoke(runnable, messages)

            try:
                updates = (
                    node(state, invoke)
                    if stage in {"plan", "select_tools", "synthesize"}
                    else node(state)
                )
                return {**updates, "model_calls": calls}
            except (ResearchError, SECError, RAGError, ValidationError) as exc:
                return {
                    "model_calls": calls,
                    "status": "failed",
                    "result": None,
                    "errors": state["errors"]
                    + [
                        {
                            "stage": stage,
                            "message": (
                                "Invalid structured model output"
                                if isinstance(exc, ValidationError)
                                else str(exc)
                            ),
                            "status_code": getattr(exc, "status_code", 502),
                        }
                    ],
                }

        return guarded

    def _check_context(self, value: object) -> None:
        if len(json.dumps(value, ensure_ascii=False)) > (
            self.settings.research_max_context_chars
        ):
            raise ResearchError("Research context limit exceeded; narrow the question")

    def _plan(self, state: ResearchState, invoke: ModelInvoker) -> dict:
        model = chat_model(self.settings)
        plan = ResearchPlan.model_validate(
            invoke(
                model.with_structured_output(ResearchPlan, method="json_schema"),
                [SystemMessage(PLANNING), HumanMessage(state["question"])],
            )
        )
        messages = [
            SystemMessage(RESEARCH_SYSTEM),
            HumanMessage(
                json.dumps(
                    {
                        "question": state["question"],
                        "plan": plan.model_dump(mode="json"),
                    }
                )
            ),
        ]
        data = messages_to_dict(messages)
        self._check_context(data)
        return {"plan": plan.model_dump(mode="json"), "messages": data}

    def _select_tools(self, state: ResearchState, invoke: ModelInvoker) -> dict:
        tools = mcp_research_tools(self.settings, {})
        self._check_context(state["messages"])
        reply = invoke(
            chat_model(self.settings).bind_tools(tools),
            messages_from_dict(state["messages"]),
        )
        if not isinstance(reply, AIMessage) or reply.invalid_tool_calls:
            raise ResearchError("Ollama returned malformed tool calls")
        if len(state["tools_called"]) + len(reply.tool_calls) > (
            self.settings.research_max_tool_calls
        ):
            raise ResearchError(
                "Research tool-call limit exceeded; narrow the question"
            )
        ids = [call["id"] for call in reply.tool_calls]
        if not all(ids) or len(set(ids)) != len(ids):
            raise ResearchError("Ollama returned missing or duplicate tool-call IDs")
        names = {tool.name for tool in tools}
        if any(call["name"] not in names for call in reply.tool_calls):
            raise ResearchError("Ollama requested an unknown research tool")
        messages = state["messages"] + messages_to_dict([reply])
        self._check_context(messages)
        return {"messages": messages, "pending_tools": reply.tool_calls}

    def _execute_tool(self, state: ResearchState) -> dict:
        self._check_context(state["messages"])
        evidence = {
            key: Evidence.model_validate(item)
            for key, item in state["evidence"].items()
        }
        tools = {
            tool.name: tool for tool in mcp_research_tools(self.settings, evidence)
        }
        call = state["pending_tools"][0]
        called = state["tools_called"] + [call["name"]]
        errors = list(state["tool_errors"])
        status = "success"
        try:
            result = tools[call["name"]].invoke(call["args"])
            content = json.dumps(result, ensure_ascii=False)
        except ValidationError as exc:
            if errors:
                # Persist the failed attempt as well as the terminal error.
                return {
                    "tools_called": called,
                    "status": "failed",
                    "errors": state["errors"]
                    + [
                        {
                            "stage": "execute_tool",
                            "status_code": 502,
                            "message": (
                                "Ollama repeatedly supplied invalid research "
                                "tool inputs"
                            ),
                        }
                    ],
                }
            content = json.dumps(
                {
                    "error": "Invalid inputs; correct fields and retry once",
                    "details": [
                        {"field": list(e["loc"]), "message": e["msg"]}
                        for e in exc.errors()
                    ],
                }
            )
            errors.append(f"{call['name']}: {content}")
            status = "error"
        except (SECError, RAGError, ResearchError) as exc:
            return {
                "tools_called": called,
                "status": "failed",
                "errors": state["errors"]
                + [
                    {
                        "stage": "execute_tool",
                        "status_code": exc.status_code,
                        "message": str(exc),
                    }
                ],
            }
        messages = state["messages"] + messages_to_dict(
            [ToolMessage(content, tool_call_id=call["id"], status=status)]
        )
        # Save the successful tool result before the next node checks its budget.
        return {
            "tools_called": called,
            "tool_errors": errors,
            "messages": messages,
            "pending_tools": state["pending_tools"][1:],
            "evidence": {
                key: item.model_dump(mode="json") for key, item in evidence.items()
            },
        }

    def _synthesize(self, state: ResearchState, invoke: ModelInvoker) -> dict:
        plan = ResearchPlan.model_validate(state["plan"])
        if plan.scope == "unsupported" or not state["evidence"]:
            draft = ResearchDraft(
                status="insufficient_evidence",
                claims=[],
                limitations=[
                    plan.limitation or "No indexed filing evidence was retrieved."
                ],
            )
        else:
            draft = self.synthesize(
                chat_model(self.settings),
                state["question"],
                {
                    key: Evidence.model_validate(item)
                    for key, item in state["evidence"].items()
                },
                invoke=invoke,
            )
        return {"draft": draft.model_dump(mode="json")}

    def _validate(self, state: ResearchState) -> dict:
        result = self.validate_result(
            ResearchDraft.model_validate(state["draft"]),
            {
                key: Evidence.model_validate(item)
                for key, item in state["evidence"].items()
            },
            state["tools_called"],
            state["model_calls"],
            state["tool_errors"],
        )
        return {"status": "completed", "result": result.model_dump(mode="json")}
