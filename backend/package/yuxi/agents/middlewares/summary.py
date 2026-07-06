"""Yuxi adapter for DeepAgents conversation summarization middleware."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Awaitable, Callable, Iterable
from contextvars import ContextVar
from typing import Any

from deepagents.middleware.summarization import SummarizationMiddleware
from langchain.agents.middleware.summarization import ContextSize
from langchain.agents.middleware.types import ExtendedModelResponse, ModelRequest, ModelResponse
from langchain.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, ToolMessage, get_buffer_string
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.config import get_stream_writer
from langgraph.constants import TAG_NOSTREAM

from yuxi.agents.backends.composite import create_agent_composite_backend
from yuxi.utils.paths import VIRTUAL_PATH_CONVERSATION_HISTORY, VIRTUAL_PATH_LARGE_TOOL_RESULTS

_APPROX_CHARS_PER_TOKEN = 4
_DEFAULT_SUMMARY_TOOL_RESULT_LIMIT_TOKENS = 500
_TOOL_RESULT_SAVED_MARKER = "yuxi_tool_result_saved"
_SUMMARY_BACKEND: ContextVar[Any | None] = ContextVar("yuxi_summary_backend", default=None)
_SUMMARY_SANITIZED_MESSAGES: ContextVar[dict[tuple[int, ...], list[AnyMessage]] | None] = ContextVar(
    "yuxi_summary_sanitized_messages",
    default=None,
)
_SUMMARY_COMPRESSION_STATE: ContextVar[dict[str, bool] | None] = ContextVar(
    "yuxi_summary_compression_state",
    default=None,
)


def _emit_compression(status: str, **extra: Any) -> None:
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    writer({"type": "yuxi.context_compression", "status": status, **extra})


def _emit_compression_started_once() -> None:
    state = _SUMMARY_COMPRESSION_STATE.get()
    if state is not None and state.get("started"):
        return
    if state is not None:
        state["started"] = True
    _emit_compression("started")


def _count_tokens_for_summary_trigger(messages: Iterable[Any], **kwargs: Any) -> int:
    kwargs.pop("use_usage_metadata_scaling", None)
    return count_tokens_approximately(messages, use_usage_metadata_scaling=False, **kwargs)


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(part for part in parts if part)
    return "" if content is None else str(content)


def _tool_result_path(tool_name: str | None, content: str, large_tool_results_prefix: str) -> str:
    safe_tool_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", (tool_name or "").strip()).strip(".-") or "tool-result"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    return f"{large_tool_results_prefix}/{safe_tool_name}-{digest}.txt"


def _preview_tool_result(content: str, token_limit: int | None) -> tuple[str, int]:
    text = content.strip()
    if token_limit is None:
        return text, 0
    if token_limit <= 0:
        return "", len(text)

    max_chars = token_limit * _APPROX_CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text, 0

    preview = text[:max_chars].rstrip()
    return preview, len(text) - len(preview)


def _write_tool_result(backend, path: str, content: str) -> str | None:
    if backend is None:
        return None

    result = backend.write(path, content)
    error = getattr(result, "error", None)
    if not error:
        return path
    if "already exists" in str(error).lower():
        return path
    raise RuntimeError(f"Failed to write tool result to {path}: {error}")


def _tool_result_replacement_content(
    message: ToolMessage,
    *,
    backend,
    tool_result_offload_token_limit: int | None,
    large_tool_results_prefix: str,
) -> str:
    content = _extract_text_content(message.content)
    approx_tokens = max((len(content) + _APPROX_CHARS_PER_TOKEN - 1) // _APPROX_CHARS_PER_TOKEN, 1)
    tool_name = message.name if isinstance(message.name, str) and message.name else None
    path = _write_tool_result(backend, _tool_result_path(tool_name, content, large_tool_results_prefix), content)
    preview, omitted_chars = _preview_tool_result(content, tool_result_offload_token_limit)

    lines = [
        "[Tool result saved]",
        f"Tool: {tool_name or 'unknown'}",
        f"Approx tokens: {approx_tokens}",
    ]
    if path:
        lines.append(f"Full output path: {path}")
    if preview:
        lines.extend(["", "Output preview:", preview])
    if omitted_chars:
        lines.append(f"\n[Truncated {omitted_chars} chars. Read the full output from the saved file.]")
    return "\n".join(lines)


def _replace_tool_message_content(
    message: ToolMessage,
    *,
    backend,
    tool_result_offload_token_limit: int | None,
    large_tool_results_prefix: str,
) -> ToolMessage:
    additional_kwargs = dict(getattr(message, "additional_kwargs", {}) or {})
    additional_kwargs[_TOOL_RESULT_SAVED_MARKER] = True
    return message.model_copy(
        update={
            "content": _tool_result_replacement_content(
                message,
                backend=backend,
                tool_result_offload_token_limit=tool_result_offload_token_limit,
                large_tool_results_prefix=large_tool_results_prefix,
            ),
            "additional_kwargs": additional_kwargs,
        }
    )


def sanitize_messages_for_summary(
    messages: list[AnyMessage],
    *,
    backend=None,
    tool_result_offload_token_limit: int | None = _DEFAULT_SUMMARY_TOOL_RESULT_LIMIT_TOKENS,
    large_tool_results_prefix: str = VIRTUAL_PATH_LARGE_TOOL_RESULTS,
) -> list[AnyMessage]:
    """Build a compact summary/offload view by replacing only ToolMessage content."""
    sanitized: list[AnyMessage] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            if getattr(message, "additional_kwargs", {}).get(_TOOL_RESULT_SAVED_MARKER) is True:
                sanitized.append(message)
                continue
            sanitized.append(
                _replace_tool_message_content(
                    message,
                    backend=backend,
                    tool_result_offload_token_limit=tool_result_offload_token_limit,
                    large_tool_results_prefix=large_tool_results_prefix,
                )
            )
            continue
        sanitized.append(message)
    return sanitized


class YuxiSummarizationMiddleware(SummarizationMiddleware):
    """DeepAgents summarization middleware with Yuxi-specific tool-call sanitization."""

    def __init__(
        self,
        *args,
        tool_result_offload_token_limit: int | None = _DEFAULT_SUMMARY_TOOL_RESULT_LIMIT_TOKENS,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tool_result_offload_token_limit = tool_result_offload_token_limit

    def _should_summarize(self, messages: list[AnyMessage], total_tokens: int) -> bool:
        if not self._lc_helper._trigger_clauses:
            return False

        for clause in self._lc_helper._trigger_clauses:
            clause_met = True
            for kind, value in clause.items():
                if kind == "messages":
                    if len(messages) < value:
                        clause_met = False
                        break
                elif kind == "tokens":
                    if total_tokens < value:
                        clause_met = False
                        break
                elif kind == "fraction":
                    max_input_tokens = self._get_profile_limits()
                    if max_input_tokens is None:
                        clause_met = False
                        break
                    threshold = int(max_input_tokens * value)
                    if threshold <= 0:
                        threshold = 1
                    if total_tokens < threshold:
                        clause_met = False
                        break
            if clause_met:
                return True
        return False

    def _sanitize_messages_for_summary(
        self,
        messages: list[AnyMessage],
        *,
        backend,
    ) -> list[AnyMessage]:
        cache = _SUMMARY_SANITIZED_MESSAGES.get()
        cache_key = tuple(id(message) for message in messages)
        if cache is not None and cache_key in cache:
            return cache[cache_key]

        sanitized = sanitize_messages_for_summary(
            messages,
            backend=backend,
            tool_result_offload_token_limit=self.tool_result_offload_token_limit,
            large_tool_results_prefix=self._large_tool_results_prefix,
        )
        if cache is not None:
            cache[cache_key] = sanitized
        return sanitized

    def _backend_for_request(self, request: ModelRequest):
        try:
            return self._get_backend(request.state, request.runtime)
        except Exception:
            return None

    @staticmethod
    def _summarization_event_from_result(result: Any) -> dict | None:
        if not isinstance(result, ExtendedModelResponse):
            return None
        command = getattr(result, "command", None)
        update = getattr(command, "update", None) if command is not None else None
        if not isinstance(update, dict):
            return None
        event = update.get("_summarization_event")
        return event if isinstance(event, dict) else None

    def _emit_completed(self, result: Any) -> None:
        event = self._summarization_event_from_result(result)
        if event is not None:
            _emit_compression(
                "completed",
                cutoff_index=event.get("cutoff_index"),
                file_path=event.get("file_path"),
            )

    # 重写 _create_summary/_acreate_summary 以在摘要 LLM 调用上挂 TAG_NOSTREAM：父类
    # 的 model.invoke 带 lc_source 元数据但无 nostream 标记，其 token 流会被 LangGraph
    # messages stream 捕获并广播到前端，形成 phantom 摘要消息。带 TAG_NOSTREAM 后流式
    # 层在源头跳过该调用，无需 chat_service 下游过滤，主 messages 流天然只含用户可见回复。
    # 父类硬编码 invoke config 且无 tags 钩子（self.model 为中间件实例共享属性，并发下不能
    # 临时换绑 bind(tags=...)），故只能重写；trim/format 是纯同步逻辑，抽到 _build_summary_prompt
    # 供 sync/async 两条路径共用，避免逐字重复。
    _SUMMARY_INVOKE_CONFIG = {"metadata": {"lc_source": "summarization"}, "tags": [TAG_NOSTREAM]}

    def _build_summary_prompt(self, sanitized: list[AnyMessage]) -> str | None:
        trimmed = self._lc_helper._trim_messages_for_summary(sanitized)
        if not trimmed:
            return None
        return self._lc_helper.summary_prompt.format(
            messages=get_buffer_string(trimmed, format="xml")
        ).rstrip()

    def _create_summary(self, messages_to_summarize: list[AnyMessage]) -> str:
        sanitized = self._sanitize_messages_for_summary(
            messages_to_summarize,
            backend=_SUMMARY_BACKEND.get(),
        )
        if not sanitized:
            return "No previous conversation history."
        prompt = self._build_summary_prompt(sanitized)
        if prompt is None:
            return "Previous conversation was too long to summarize."
        try:
            return self.model.invoke(prompt, config=self._SUMMARY_INVOKE_CONFIG).text.strip()
        except Exception as e:
            return f"Error generating summary: {e!s}"

    async def _acreate_summary(self, messages_to_summarize: list[AnyMessage]) -> str:
        sanitized = self._sanitize_messages_for_summary(
            messages_to_summarize,
            backend=_SUMMARY_BACKEND.get(),
        )
        if not sanitized:
            return "No previous conversation history."
        prompt = self._build_summary_prompt(sanitized)
        if prompt is None:
            return "Previous conversation was too long to summarize."
        try:
            response = await self.model.ainvoke(prompt, config=self._SUMMARY_INVOKE_CONFIG)
            return response.text.strip()
        except Exception as e:
            return f"Error generating summary: {e!s}"

    def _offload_to_backend(self, backend, messages: list[AnyMessage]) -> str | None:
        _emit_compression_started_once()
        return super()._offload_to_backend(
            backend,
            self._sanitize_messages_for_summary(
                messages,
                backend=backend,
            ),
        )

    async def _aoffload_to_backend(self, backend, messages: list[AnyMessage]) -> str | None:
        _emit_compression_started_once()
        return await super()._aoffload_to_backend(
            backend,
            self._sanitize_messages_for_summary(
                messages,
                backend=backend,
            ),
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        backend_token = _SUMMARY_BACKEND.set(self._backend_for_request(request))
        sanitized_token = _SUMMARY_SANITIZED_MESSAGES.set({})
        compression_state: dict[str, bool] = {"started": False}
        compression_token = _SUMMARY_COMPRESSION_STATE.set(compression_state)
        try:
            try:
                result = super().wrap_model_call(request, handler)
            except Exception as exc:
                if compression_state.get("started"):
                    _emit_compression("failed", error=repr(exc))
                raise
            self._emit_completed(result)
            return result
        finally:
            _SUMMARY_COMPRESSION_STATE.reset(compression_token)
            _SUMMARY_SANITIZED_MESSAGES.reset(sanitized_token)
            _SUMMARY_BACKEND.reset(backend_token)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        backend_token = _SUMMARY_BACKEND.set(self._backend_for_request(request))
        sanitized_token = _SUMMARY_SANITIZED_MESSAGES.set({})
        compression_state: dict[str, bool] = {"started": False}
        compression_token = _SUMMARY_COMPRESSION_STATE.set(compression_state)
        try:
            try:
                result = await super().awrap_model_call(request, handler)
            except Exception as exc:
                if compression_state.get("started"):
                    _emit_compression("failed", error=repr(exc))
                raise
            self._emit_completed(result)
            return result
        finally:
            _SUMMARY_COMPRESSION_STATE.reset(compression_token)
            _SUMMARY_SANITIZED_MESSAGES.reset(sanitized_token)
            _SUMMARY_BACKEND.reset(backend_token)


def create_summary_middleware(
    model: str | BaseChatModel,
    *,
    trigger: ContextSize | list[ContextSize] | None,
    keep: ContextSize | list[ContextSize] | None,
    summary_prompt: str | None = None,
    trim_tokens_to_summarize: int | None = None,
    tool_result_offload_token_limit: int | None = _DEFAULT_SUMMARY_TOOL_RESULT_LIMIT_TOKENS,
) -> SummarizationMiddleware:
    """Create DeepAgents summarization middleware using Yuxi's virtual outputs root."""
    middleware_kwargs = {
        "model": model,
        "backend": create_agent_composite_backend,
        "trigger": trigger,
        "keep": keep,
        "token_counter": _count_tokens_for_summary_trigger,
        "trim_tokens_to_summarize": trim_tokens_to_summarize,
        "tool_result_offload_token_limit": tool_result_offload_token_limit,
    }
    if summary_prompt and summary_prompt.strip():
        middleware_kwargs["summary_prompt"] = summary_prompt
    middleware = YuxiSummarizationMiddleware(**middleware_kwargs)
    middleware._history_path_prefix = VIRTUAL_PATH_CONVERSATION_HISTORY
    middleware._large_tool_results_prefix = VIRTUAL_PATH_LARGE_TOOL_RESULTS
    return middleware
