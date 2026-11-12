"""Bounded LangChain tool-calling baseline, without graph orchestration."""

import json
from collections.abc import Callable
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from pydantic import ValidationError

from app.config import Settings
from app.rag_embeddings import RAGError
from app.rag_models import Evidence
from app.rag_service import RAGService
from app.research_models import ResearchDraft, ResearchRequest, ResearchResult
from app.research_prompts import RESEARCH_SYSTEM, SYNTHESIS
from app.research_tools import research_tools
from app.sec_client import SECError
from app.sec_service import SECService

ModelInvoker = Callable[[Runnable[Any, Any], list[BaseMessage]], Any]


class ResearchError(Exception):
    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


def chat_model(settings: Settings) -> ChatOllama:
    return ChatOllama(
        model=settings.chat_model,
        base_url=settings.ollama_base_url,
        temperature=0,
        reasoning=False,
        num_ctx=settings.chat_context_size,
        num_predict=settings.chat_max_tokens,
        keep_alive=0,
        client_kwargs={"timeout": settings.chat_timeout_seconds},
    )


class ResearchService:
    def __init__(self, sec: SECService, rag: RAGService, settings: Settings) -> None:
        self.sec = sec
        self.rag = rag
        self.settings = settings

    def research(self, request: ResearchRequest) -> ResearchResult:
        evidence: dict[str, Evidence] = {}
        tools = research_tools(self.sec, self.rag, evidence)
        by_name = {tool.name: tool for tool in tools}
        model = chat_model(self.settings)
        caller = model.bind_tools(tools)
        messages: list[BaseMessage] = [
            SystemMessage(RESEARCH_SYSTEM),
            HumanMessage(request.question),
        ]
        called: list[str] = []
        tool_errors: list[str] = []
        model_calls = 0
        used_chars = len(RESEARCH_SYSTEM) + len(request.question)
        while True:
            reply = self._invoke(caller, messages)
            model_calls += 1
            if not isinstance(reply, AIMessage) or reply.invalid_tool_calls:
                raise ResearchError("Ollama returned malformed tool calls")
            # Discard intermediate free-form prose; only the validated draft is public.
            used_chars += len(reply.model_dump_json())
            messages.append(reply)
            if not reply.tool_calls:
                break
            if (
                len(called) + len(reply.tool_calls)
                > self.settings.research_max_tool_calls
            ):
                raise ResearchError(
                    "Research tool-call limit exceeded; narrow the question"
                )
            ids = [call["id"] for call in reply.tool_calls]
            if not all(ids) or len(set(ids)) != len(ids):
                raise ResearchError(
                    "Ollama returned missing or duplicate tool-call IDs"
                )
            for call in reply.tool_calls:
                tool = by_name.get(call["name"])
                if tool is None:
                    raise ResearchError("Ollama requested an unknown research tool")
                called.append(tool.name)
                try:
                    result = tool.invoke(call["args"])
                except ValidationError as exc:
                    # Give the local model one explicit correction opportunity.
                    # Invalid attempts still consume the total tool-call budget.
                    if tool_errors:
                        raise ResearchError(
                            "Ollama repeatedly supplied invalid research tool inputs"
                        ) from exc
                    details = [
                        {"field": list(error["loc"]), "message": error["msg"]}
                        for error in exc.errors()
                    ]
                    content = json.dumps(
                        {
                            "error": "Invalid inputs; correct fields and retry once",
                            "details": details,
                        }
                    )
                    tool_errors.append(f"{tool.name}: {content}")
                    used_chars += len(content)
                    messages.append(
                        ToolMessage(content, tool_call_id=call["id"], status="error")
                    )
                    continue
                except (SECError, RAGError) as exc:
                    raise ResearchError(str(exc), exc.status_code) from exc
                content = json.dumps(result, ensure_ascii=False)
                used_chars += len(content)
                messages.append(ToolMessage(content, tool_call_id=call["id"]))
            if used_chars > self.settings.research_max_context_chars:
                raise ResearchError(
                    "Research context limit exceeded; narrow the question"
                )
        draft = self.synthesize(model, request.question, evidence)
        return self.validate_result(
            draft, evidence, called, model_calls + 1, tool_errors
        )

    def synthesize(
        self,
        model: ChatOllama,
        question: str,
        evidence: dict[str, Evidence],
        *,
        invoke: ModelInvoker | None = None,
    ) -> ResearchDraft:
        # Synthesis receives the original question and verified tool data directly,
        # without intermediate model prose or tool-call syntax to imitate.
        synthesis_input = json.dumps(
            {
                "question": question,
                "evidence": [
                    {
                        "evidence_id": item.chunk.chunk_id,
                        "text": item.chunk.text,
                        "section": item.chunk.section,
                        "filing": item.chunk.filing.model_dump(mode="json"),
                    }
                    for item in evidence.values()
                ],
            },
            ensure_ascii=False,
        )
        if (
            len(synthesis_input) + len(SYNTHESIS)
            > self.settings.research_max_context_chars
        ):
            raise ResearchError("Research context limit exceeded; narrow the question")
        draft = (invoke or self._invoke)(
            model.with_structured_output(ResearchDraft, method="json_schema"),
            [SystemMessage(SYNTHESIS), HumanMessage(synthesis_input)],
        )
        try:
            return ResearchDraft.model_validate(draft)
        except ValidationError as exc:
            raise ResearchError(
                "Ollama returned an invalid structured research answer"
            ) from exc

    def validate_result(
        self,
        draft: ResearchDraft,
        evidence: dict[str, Evidence],
        called: list[str],
        model_calls: int,
        tool_errors: list[str],
    ) -> ResearchResult:
        cited = list(
            dict.fromkeys(key for claim in draft.claims for key in claim.evidence_ids)
        )
        if any(key not in evidence for key in cited):
            raise ResearchError("Research answer cites evidence that was not retrieved")
        return ResearchResult(
            **draft.model_dump(),
            model=self.settings.chat_model,
            evidence=[evidence[key] for key in cited],
            tools_called=called,
            model_calls=model_calls,
            tool_errors=tool_errors,
        )

    @staticmethod
    def _invoke(runnable: Runnable[Any, Any], messages: list[BaseMessage]) -> Any:
        try:
            return runnable.invoke(messages)
        except Exception as exc:
            raise ResearchError(
                "Local chat request or output parsing failed; check Ollama, the "
                "configured model and its tool/structured-output support",
                503,
            ) from exc
