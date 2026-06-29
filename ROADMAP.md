# PaperPilot 项目路线图

> 目标：打造一个可放在简历上的、有量化数据支撑的 RAG 文献知识库项目
> 仓库地址：https://github.com/FSSLI/PaperPilot

---

## 第一步：架构图 + README 优化

让 GitHub 仓库第一眼看起来专业、完整。

### 待办

- [x] 绘制系统架构图（Mermaid 流程图，GitHub 原生渲染）
- [x] README 重构为 GitHub 项目页面（功能亮点、架构图、技术栈、快速开始）
- [x] 添加 LICENSE 文件（MIT）
- [x] 添加演示截图/GIF（发 PDF → 微信提问 → 得到回答的完整流程）
- [x] 添加 GitHub Topics 标签

---

## 第二步：搭建 RAG 评测体系 + 引用溯源

用数据说话，为后续优化提供基准。

### 评测体系

- [x] 准备评测数据集：从已入库文献中构造 20-30 组 QA 对
  - 格式：问题 + 标准答案 + 出处（论文名 + 段落位置）
  - 37 篇文献，70 组 QA（事实性/方法细节/概念性/比较性/综合性/数值型）
- [x] 引入 Ragas 评测框架（https://github.com/explodinggradients/ragas）
- [x] 评测维度：
  - **检索召回率（Context Recall）**：检索到的段落是否包含正确答案
  - **回答忠实度（Faithfulness）**：LLM 回答是否基于检索内容，有无幻觉
  - **回答相关性（Answer Relevancy）**：回答是否切题
  - **上下文精确度（Context Precision）**：相关块是否排在前面
- [x] 建立自动化评测脚本，改参数后可一键跑分
  - eval_rag.py（数据采集）+ compute_ragas.py（指标计算）
  - 当前结果：Faithfulness=0.85, Answer Relevancy=0.74, Context Recall=0.77, Context Precision=0.63

### 引用溯源

- [x] 回答中标注来源：LLM 回答时使用【来源：《文件名》】格式标注信息来源，context 构建时每个 chunk 前加文档名标记
- [x] 支持跨文献对比问题：检索多个文档并在 Prompt 中要求引用多个来源
- [x] 评测中加入引用准确率指标：新增 Citation Rate / Citation Recall / Citation Precision / Citation Accuracy 四项指标

---

## 第三步：基于评测结果优化 RAG

每改一个参数跑一轮评测，积累对比数据。

### 优化方向

- [ ] 文本分块策略：调整 chunk size / overlap
- [ ] 检索方式：加入混合检索（关键词 BM25 + 向量相似度）
- [ ] Rerank 调优：调整 rerank 权重、top-k 参数
- [ ] Prompt 工程：限制只用检索内容回答，减少幻觉
- [ ] Embedding 模型对比：text-embedding-v4 vs 其他模型

### 期望产出

- 优化前后对比表格（检索召回率、忠实度、相关性的变化）
- 简历描述示例："通过优化分块策略和检索排序，检索召回率从 65% 提升至 89%，回答忠实度从 72% 提升至 94%"

---

## 第四步：用户反馈闭环

建立数据驱动的持续优化机制。

- [ ] 每次回答后加"有用/没用"反馈按钮
- [ ] 记录反馈数据：哪些回答没用 → 分析检索失败原因
- [ ] 简历描述示例："建立用户反馈驱动的 RAG 迭代优化机制，基于 200+ 条反馈数据定向优化检索策略"

---

## 第五步：文献自动摘要

上传文献后自动生成结构化摘要。

- [ ] 上传文献时自动调用 LLM 生成中文摘要 + 关键词
- [ ] 摘要展示在管理后台的文献详情页
- [ ] 支持用户手动刷新/重新生成摘要

---

## 第六步：全栈 Docker Compose 编排

体现容器化和工程化能力。

- [ ] 为 mildoc_index、mildoc_admin、mildoc_wxkf 编写 Dockerfile
- [ ] 编写统一的 docker-compose.yml，一键启动全部服务（Milvus + etcd + MinIO + 3个Python服务）
- [ ] README 补充 Docker 部署方式
- [ ] 简历描述示例："全栈容器化部署，docker compose 一键启动 6 个服务"

---

## 进度记录

| 日期 | 事项 | 状态 |
|------|------|------|
| 2026-06-28 | 项目初始化、README、.gitignore、.env.example | ✅ 完成 |
| 2026-06-28 | 微信发送文件自动入库功能 | ✅ 完成 |
| 2026-06-28 | 修复微信文件名丢失问题 | ✅ 完成 |
| 2026-06-28 | 制定项目路线图 | ✅ 完成 |
| 2026-06-29 | README 重构 + Mermaid 架构图 + MIT LICENSE | ✅ 完成 |
| 2026-06-29 | RAG 评测体系（Ragas 框架 + 4项指标 + 37篇文献70组QA） | ✅ 完成 |
| 2026-06-29 | 文件哈希去重（避免重复 embedding 调用） | ✅ 完成 |
| 2026-06-29 | README 快速开始优化（补充 conda 环境说明，精简重复步骤） | ✅ 完成 |
| 2026-06-29 | 引用溯源：Prompt 要求标注来源 + Context 加文档名标记 + 4项引用评测指标 | ✅ 完成 |
