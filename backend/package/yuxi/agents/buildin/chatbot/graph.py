from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, TodoListMiddleware

from yuxi.agents import BaseAgent, BaseState, load_chat_model
from yuxi.agents.backends import create_agent_composite_backend, create_agent_filesystem_middleware
from yuxi.agents.context import prepare_agent_runtime_context
from yuxi.agents.middlewares import (
    create_summary_middleware,
    save_attachments_to_fs,
)
from yuxi.agents.middlewares.knowledge_base_middleware import KnowledgeBaseMiddleware
from yuxi.agents.middlewares.skills_middleware import SkillsMiddleware
from yuxi.agents.subagents.service import build_subagent_middleware_specs, get_subagents_from_slugs
from yuxi.agents.toolkits.service import resolve_configured_runtime_tools

from .prompt import TODO_MID_PROMPT, build_prompt_with_context


async def _build_middlewares(context):
    """构建中间件列表"""
    # summary middleware
    # 主 Agent 上下文优化：默认 100k tokens 触发压缩，保留最近 50%
    summary_trigger_tokens = getattr(context, "summary_threshold", 100) * 1024
    summary_middleware = create_summary_middleware(
        model=load_chat_model(fully_specified_name=context.model),
        trigger=("tokens", summary_trigger_tokens),
        keep=("tokens", summary_trigger_tokens // 2),
        trim_tokens_to_summarize=4000,
    )

    # subagents
    subagents = await get_subagents_from_slugs(context.subagents)
    default_subagent_middleware = [
        create_agent_filesystem_middleware(tool_token_limit_before_evict=500),  # 文件系统后端
        PatchToolCallsMiddleware(),
        summary_middleware,
    ]
    subagents_middleware = SubAgentMiddleware(
        backend=create_agent_composite_backend,
        subagents=build_subagent_middleware_specs(
            subagents,
            default_model=load_chat_model(fully_specified_name=context.subagents_model),
            default_middleware=default_subagent_middleware,
            model_loader=load_chat_model,
        ),
        state_schema=BaseState,
    )
    # all middlewares
    middlewares = [
        create_agent_filesystem_middleware(tool_token_limit_before_evict=500),  # 文件系统后端
        save_attachments_to_fs,  # 附件注入提示词
        KnowledgeBaseMiddleware(),  # 知识库工具
        SkillsMiddleware(),  # Skills 中间件（提示词注入、依赖展开、动态激活）
        subagents_middleware,
        summary_middleware,
        TodoListMiddleware(system_prompt=TODO_MID_PROMPT),  # 待办事项中间件
        PatchToolCallsMiddleware(),
        ModelRetryMiddleware(),  # 模型重试中间件
    ]

    return middlewares


class ChatbotAgent(BaseAgent):
    name = "智能助手"
    description = "基础的对话机器人，可以回答问题，可在配置中启用需要的工具。"
    capabilities = ["file_upload", "files"]  # 支持文件上传功能

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def get_graph(self, context=None, **kwargs):

        context = await prepare_agent_runtime_context(
            context or self.context_schema(),
            context_schema=self.context_schema,
        )

        # 使用 create_agent 创建智能体
        graph = create_agent(
            model=load_chat_model(fully_specified_name=context.model),
            tools=await resolve_configured_runtime_tools(context),
            system_prompt=build_prompt_with_context(context),
            middleware=await _build_middlewares(context),
            state_schema=BaseState,
            checkpointer=await self._get_checkpointer(),
        )

        return graph


def main():
    pass


if __name__ == "__main__":
    main()
    # asyncio.run(main())
