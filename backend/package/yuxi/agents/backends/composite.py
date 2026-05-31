from __future__ import annotations

from deepagents.backends.composite import (
    CompositeBackend,
    _remap_file_info_path,
    _route_for_path,
    _strip_route_from_pattern,
)
from deepagents.backends.protocol import FileInfo, GlobResult
from deepagents.middleware.filesystem import FilesystemMiddleware

from yuxi.agents.skills.service import normalize_string_list
from yuxi.utils.paths import VIRTUAL_PATH_CONVERSATION_HISTORY, VIRTUAL_PATH_LARGE_TOOL_RESULTS, VIRTUAL_PATH_OUTPUTS

from .sandbox import ProvisionerSandboxBackend
from .skills_backend import SelectedSkillsReadonlyBackend


def _coerce_glob_result(result) -> GlobResult:
    if isinstance(result, GlobResult):
        return result
    return GlobResult(matches=result or [])


class CustomCompositeBackend(CompositeBackend):
    """修复 glob 路由逻辑的 CompositeBackend。

    修复内容：当 path 不匹配任何路由时应该只搜索 default 后端，
    而不是错误地遍历所有路由后端搜索。
    """

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        backend, backend_path, route_prefix = _route_for_path(
            default=self.default,
            sorted_routes=self.sorted_routes,
            path=path,
        )
        if route_prefix is not None:
            result = _coerce_glob_result(backend.glob(pattern, backend_path))
            if result.error:
                return result
            return GlobResult(matches=[_remap_file_info_path(fi, route_prefix) for fi in (result.matches or [])])

        if path is None or path == "/":
            results: list[FileInfo] = []
            default_result = _coerce_glob_result(self.default.glob(pattern, path))
            if default_result.error:
                return default_result
            results.extend(default_result.matches or [])
            for route_prefix, backend in self.routes.items():
                route_pattern = _strip_route_from_pattern(pattern, route_prefix)
                result = _coerce_glob_result(backend.glob(route_pattern, "/"))
                if result.error:
                    return result
                results.extend(_remap_file_info_path(fi, route_prefix) for fi in (result.matches or []))
            results.sort(key=lambda x: x.get("path", ""))
            return GlobResult(matches=results)

        return _coerce_glob_result(self.default.glob(pattern, path))

    async def aglob(self, pattern: str, path: str = "/") -> GlobResult:
        backend, backend_path, route_prefix = _route_for_path(
            default=self.default,
            sorted_routes=self.sorted_routes,
            path=path,
        )
        if route_prefix is not None:
            result = _coerce_glob_result(await backend.aglob(pattern, backend_path))
            if result.error:
                return result
            return GlobResult(matches=[_remap_file_info_path(fi, route_prefix) for fi in (result.matches or [])])

        if path is None or path == "/":
            results: list[FileInfo] = []
            default_result = _coerce_glob_result(await self.default.aglob(pattern, path))
            if default_result.error:
                return default_result
            results.extend(default_result.matches or [])
            for route_prefix, backend in self.routes.items():
                route_pattern = _strip_route_from_pattern(pattern, route_prefix)
                result = _coerce_glob_result(await backend.aglob(route_pattern, "/"))
                if result.error:
                    return result
                results.extend(_remap_file_info_path(fi, route_prefix) for fi in (result.matches or []))
            results.sort(key=lambda x: x.get("path", ""))
            return GlobResult(matches=results)

        return _coerce_glob_result(await self.default.aglob(pattern, path))


def _get_readable_skills_from_runtime(runtime) -> list[str]:
    context = getattr(runtime, "context", None)
    selected = getattr(context, "_readable_skills", [])
    return normalize_string_list(selected if isinstance(selected, list) else [])


def _extract_thread_id(runtime) -> str:
    config = getattr(runtime, "config", None)
    if isinstance(config, dict):
        configurable = config.get("configurable", {})
        if isinstance(configurable, dict):
            thread_id = configurable.get("thread_id")
            if isinstance(thread_id, str) and thread_id.strip():
                return thread_id.strip()

    context = getattr(runtime, "context", None)
    thread_id = getattr(context, "thread_id", None)
    if isinstance(thread_id, str) and thread_id.strip():
        return thread_id.strip()

    raise ValueError("thread_id is required in runtime configurable context")


def _extract_uid(runtime) -> str:
    config = getattr(runtime, "config", None)
    if isinstance(config, dict):
        configurable = config.get("configurable", {})
        if isinstance(configurable, dict):
            uid = configurable.get("uid")
            if isinstance(uid, str) and uid.strip():
                return uid.strip()

    context = getattr(runtime, "context", None)
    uid = getattr(context, "uid", None)
    if isinstance(uid, str) and uid.strip():
        return uid.strip()

    raise ValueError("uid is required in runtime configurable context")


def create_agent_composite_backend(runtime) -> CompositeBackend:
    readable_skills = _get_readable_skills_from_runtime(runtime)
    thread_id = _extract_thread_id(runtime)
    uid = _extract_uid(runtime)
    return CustomCompositeBackend(
        default=ProvisionerSandboxBackend(thread_id=thread_id, uid=uid, readable_skills=readable_skills),
        routes={
            "/skills/": SelectedSkillsReadonlyBackend(selected_slugs=readable_skills),
        },
        artifacts_root=VIRTUAL_PATH_OUTPUTS,
    )


def create_agent_filesystem_middleware(tool_token_limit_before_evict: int | None = None) -> FilesystemMiddleware:
    middleware = FilesystemMiddleware(
        backend=create_agent_composite_backend,
        tool_token_limit_before_evict=tool_token_limit_before_evict,
    )
    middleware._large_tool_results_prefix = VIRTUAL_PATH_LARGE_TOOL_RESULTS
    middleware._conversation_history_prefix = VIRTUAL_PATH_CONVERSATION_HISTORY
    return middleware
