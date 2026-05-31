import os

from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import (
    TodoListMiddleware,
    ToolCallLimitMiddleware,
)

from yuxi.agents import BaseAgent, BaseState, load_chat_model
from yuxi.agents.backends import create_agent_composite_backend, create_agent_filesystem_middleware
from yuxi.agents.context import prepare_agent_runtime_context
from yuxi.agents.middlewares import (
    create_summary_middleware,
    save_attachments_to_fs,
)
from yuxi.agents.middlewares.knowledge_base_middleware import KnowledgeBaseMiddleware
from yuxi.agents.middlewares.skills_middleware import SkillsMiddleware
from yuxi.agents.toolkits.buildin.tools import _create_tavily_search
from yuxi.agents.subagents.service import build_subagent_middleware_specs, get_subagents_from_slugs
from yuxi.agents.toolkits.service import resolve_configured_runtime_tools
from yuxi.utils import logger
from yuxi.utils.datetime_utils import shanghai_now

from .prompt import DEEP_PROMPT


class DeepAgent(BaseAgent):
    name = "深度分析"
    description = "具备规划、深度分析和子智能体协作能力的智能体，可以处理复杂的多步骤任务"
    capabilities = ["file_upload", "files"]  # 支持文件上传功能

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.graph = None
        self.checkpointer = None

    async def get_tools(self):
        """返回 Deep Agent 的专用工具"""
        tools = []
        if os.getenv("TAVILY_API_KEY"):
            tavily = _create_tavily_search()
            if tavily:
                tools.append(tavily)

        if not tools:
            logger.warning("No search tools configured, DeepAgent will work without web search")
        return tools

    async def get_graph(self, context=None, **kwargs):

        context = await prepare_agent_runtime_context(
            context or self.context_schema(),
            context_schema=self.context_schema,
        )
        current_date = f"当前日期：{shanghai_now().strftime('%Y-%m-%d')}"
        system_prompt = f"{current_date}\n\n{DEEP_PROMPT.strip()}\n\n{context.system_prompt or ''}"

        model = load_chat_model(context.model)
        sub_model = load_chat_model(context.subagents_model)
        search_tools = await self.get_tools()

        # 从数据库加载 subagent specs（工具名称已解析）
        user_subagents = await get_subagents_from_slugs(context.subagents)

        # 主 Agent 上下文优化：90k tokens 触发压缩，保留最近 50%
        summary_middleware = create_summary_middleware(
            model=model,
            trigger=("tokens", 90000),
            keep=("tokens", 45000),
            trim_tokens_to_summarize=4000,
        )

        default_subagent_middleware = [
            create_agent_filesystem_middleware(tool_token_limit_before_evict=500),  # 文件系统后端
            PatchToolCallsMiddleware(),
            summary_middleware,
            # 子 Agent 搜索工具限制：tavily_search 最多 8 次
            ToolCallLimitMiddleware(
                tool_name="tavily_search",
                run_limit=8,
                exit_behavior="continue",
            ),
        ]
        subagents_middleware = SubAgentMiddleware(
            backend=create_agent_composite_backend,
            subagents=build_subagent_middleware_specs(
                user_subagents,
                default_model=sub_model,
                default_tools=search_tools,
                default_middleware=default_subagent_middleware,
                model_loader=load_chat_model,
            ),
            state_schema=BaseState,
        )

        # 使用 create_deep_agent 创建深度智能体
        graph = create_agent(
            model=model,
            tools=await resolve_configured_runtime_tools(context),
            system_prompt=system_prompt,
            middleware=[
                create_agent_filesystem_middleware(tool_token_limit_before_evict=500),  # 文件系统后端
                SkillsMiddleware(),  # Skills 中间件（提示词注入、依赖展开、动态激活）
                save_attachments_to_fs,  # 附件注入提示词
                TodoListMiddleware(system_prompt="任务结束前，应该检查维护的待办事项列表是否结束。"),
                PatchToolCallsMiddleware(),
                KnowledgeBaseMiddleware(),  # 知识库工具
                subagents_middleware,
                summary_middleware,
                # 工具调用限制：tavily_search 总调用最多 20 次
                ToolCallLimitMiddleware(
                    tool_name="tavily_search",
                    thread_limit=20,
                    exit_behavior="continue",
                ),
                # 总工具调用轮次限制：防止单次运行无限循环
                ToolCallLimitMiddleware(
                    run_limit=50,
                    exit_behavior="end",
                ),
            ],
            state_schema=BaseState,
            checkpointer=await self._get_checkpointer(),
        )

        return graph
