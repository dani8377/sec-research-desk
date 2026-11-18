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
from app.research_context import (
    fit_messages,
    prepare_arguments,
    selection_reply,
    tool_key,
    tool_message_content,
)
from app.research_models import ResearchDraft, ResearchRequest, ResearchResult
from app.research_prompts import RESEARCH_SYSTEM, SYNTHESIS
from app.research_tools import research_tools
from app.sec_client import SECError
from app.sec_service import SECService
from app.telemetry import traced

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

    def _selection_messages(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        try:
            return fit_messages(messages, self.settings.research_max_context_chars)
        except ValueError as exc:
            raise ResearchError(str(exc)) from exc

    @traced("research.baseline")
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
        warnings: list[str] = []
        executed: set[str] = set()
        model_calls = 0
        stop = False
        while not stop:
            messages = self._selection_messages(messages)
            reply = self._invoke(caller, messages)
            model_calls += 1
            if not isinstance(reply, AIMessage) or reply.invalid_tool_calls:
                raise ResearchError("Ollama returned malformed tool calls")
            messages.append(selection_reply(reply))
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
                    arguments, repairs = prepare_arguments(
                        tool.name, call["args"], request.question
                    )
                    warnings.extend(repairs)
                    key = tool_key(tool.name, arguments)
                    if key in executed:
                        stop = True
                        note = (
                            "Repeated identical tool request skipped; "
                            "synthesis uses retained evidence."
                        )
                        warnings.append(note)
                        content = json.dumps({"duplicate_request": True, "note": note})
                    else:
                        known_ids = set(evidence)
                        result = tool.invoke(arguments)
                        executed.add(key)
                        warnings.extend(result.get("warnings", []))
                        result = {
                            **result,
                            "warnings": result.get("warnings", []) + repairs,
                        }
                        content = tool_message_content(tool.name, result, known_ids)
                except (ValidationError, SECError, RAGError) as exc:
                    if not isinstance(exc, ValidationError) and exc.status_code != 422:
                        raise ResearchError(str(exc), exc.status_code) from exc
                    if tool_errors:
                        raise ResearchError(
                            "Ollama repeatedly supplied invalid research tool inputs"
                        ) from exc
                    details = (
                        [
                            {"field": list(error["loc"]), "message": error["msg"]}
                            for error in exc.errors()
                        ]
                        if isinstance(exc, ValidationError)
                        else str(exc)
                    )
                    content = json.dumps(
                        {
                            "error": "Invalid inputs; correct fields and retry once",
                            "details": details,
                        }
                    )
                    tool_errors.append(f"{tool.name}: {content}")
                    messages.append(
                        ToolMessage(content, tool_call_id=call["id"], status="error")
                    )
                    continue
                messages.append(ToolMessage(content, tool_call_id=call["id"]))
        if evidence:

            def invoke(
                runnable: Runnable[Any, Any], messages: list[BaseMessage]
            ) -> Any:
                nonlocal model_calls
                model_calls += 1
                return self._invoke(runnable, messages)

            draft = self.synthesize(model, request.question, evidence, invoke=invoke)
        else:
            draft = ResearchDraft(
                status="insufficient_evidence",
                claims=[],
                limitations=[
                    "No indexed filing evidence was retrieved; no answer was generated."
                ],
            )
        return self.validate_result(
            draft,
            evidence,
            called,
            model_calls,
            tool_errors,
            tool_warnings=list(dict.fromkeys(warnings)),
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
        packet: list[dict] = []
        omitted = 0
        for item in evidence.values():
            source = {
                "evidence_id": item.chunk.chunk_id,
                "text": item.chunk.text,
                "section": item.chunk.section,
                "filing": item.chunk.filing.model_dump(mode="json"),
            }
            candidate = json.dumps(
                {"question": question, "evidence": packet + [source]},
                ensure_ascii=False,
            )
            # Reserve space for a deterministic disclosure of excluded passages.
            if (
                len(candidate) + len(SYNTHESIS) + 300
                > self.settings.research_max_context_chars
            ):
                omitted += 1
            else:
                packet.append(source)
        if not packet:
            return ResearchDraft(
                status="insufficient_evidence",
                claims=[],
                limitations=[
                    "Retrieved passages could not fit the synthesis context budget."
                ],
            )
        synthesis_input = json.dumps(
            {"question": question, "evidence": packet}, ensure_ascii=False
        )
        draft = (invoke or self._invoke)(
            model.with_structured_output(ResearchDraft, method="json_schema"),
            [SystemMessage(SYNTHESIS), HumanMessage(synthesis_input)],
        )
        try:
            parsed = ResearchDraft.model_validate(draft)
            allowed = {item["evidence_id"] for item in packet}
            if omitted and any(
                key not in allowed
                for claim in parsed.claims
                for key in claim.evidence_ids
            ):
                raise ResearchError(
                    "Research answer cites evidence that was not retrieved "
                    "or supplied to synthesis"
                )
            if omitted:
                parsed.limitations = parsed.limitations[:11] + [
                    f"{omitted} retrieved passages were excluded to fit "
                    "the synthesis context budget."
                ]
            return parsed
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
        *,
        tool_warnings: list[str] | None = None,
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
            tool_warnings=tool_warnings or [],
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
