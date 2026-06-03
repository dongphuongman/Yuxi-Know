
# 兼容性报告

无法兼容历史的数据库、知识库、配置项，只能重新创建项目

# 功能性报告

1. 将 LightRAG 从调用 package 修改为手动实现，不依赖原本仓库，更好管理。
2. 移除 legacy 参数，简化代码逻辑，减少维护成本。
3. 权限结构重构

# 待办事项

- [x] 移除所有对于 v1 的模型的代码的支持、移除所有对于 legacy 参数的支持
- [x] 移除 lightrag
- [x] 新增图谱构建，支持自定义 Schema、支持自定义并发数，知识库与向量检索解耦
- [x] 知识图谱抽取要求支持并发处理
- [ ] UV 的版本
- [x] Agent 的 user_id 使用的有歧义，从数据库表到代码中都修改为统一使用 yuxi_id 来代替
- [ ] 权限新增一个 guest 的预设字段，暂无任何权限
- [ ] rename database table name, such as skills -> agent_skills, subagents -> agent_subagents, mcp, tool_call, 等等
- [x] department 的 id 也不能使用那个索引的 id 来使用了，应该是一个独立的 dept_id，需要确认
- [x] allow user config envs
- [x] add model retry times to agent context config
- [x] 添加用户级别的 Skills 的安装
- [x] Skill 卡片优化
- [ ] 拓展 Skill 安装方法
- [x] MCP 部分，未添加情况下无法获取工具
- [ ] 添加 MCP 移除 JSON 模式
- [ ] 工作区允许上传多个文件
- [x] 子智能体的消息渲染与可视化
- [x] 子智能体的优化，参考 PR 的方案。
- [x] 附件上传能够支持转换为 PDF，待办：查看 OCR 模型的状态，样式优化，保存的文件名不对
- [x] 参考 PR，实现内置 Dashscope 的 Embedding 和 rerank 的方法
- [x] 优化知识库的 API 接口设计，使用 /{db_id}/xxx 的形式，整合 mindmap / eval 接口
- [x] 前端新增基于 ID 的随机像素头像生成机制（类似 GitHub 方块头像），用于替代默认图片
- [x] 子智能体右上角功能增强：任务按钮常驻，展开后展示已激活子智能体、附件、state 等信息
- [x] 流式输出混排修复：ongoing 流式返回时按 thread ID 区分主/子智能体内容，子智能体内容单独可视化渲染
- [x] 子智能体中间过程可查看：流式结束后，通过子智能体自身 thread ID 获取并渲染中间调用过程
- [x] allow multi-hop qa generate
- [x] 在工作区的文件编辑的时候，保存和取消的按钮应该是悬浮在编辑框的右上角，而不是在 header 上面
- [x] default enable all build in tools / kbs / skills / mcps / subagents
- [x] 链接 Notion 和 feishu 目前来看，都是支持的
- [x] 知识库的权限调整，修改为三个等级，全局共享、部门共享、指定人可访问
- [x] 优化评估结构
- [ ] 考虑如何将知识库更好的挂载到沙盒，是不是可以使用一个别的后端，但是使用别的后端是否还能读取到数据？应该不能
- [x] 智能体体系改进。改进子智能体，我觉得子智能体并不是一个子级的智能体，应该是
- [ ] RAG 中的文件的 metadata 包含那些内容？然后 Find 和 Read 的时候要支持展示
- [x] 检查前端的这些 store 的使用，是否有需要调整优化收敛的地方
- [x] install skill 这个工具需要在主智能体启用，子智能体无法使用
- [x] parser 从plugins 移动到 knowledge 里面，guard 移动到services 里面
- [x] neo4j 相关的服务，可以移动到 storage 里面
- [ ] 点开对话的时候要能够自动定位到尾部，而不是最开始。
- [x] 将 Qwen-Image 修改为 Skill
- [x] 现在输入区域对于不同 mention 的渲染的 ICON 和 human-message 里面的渲染的 ICON;
