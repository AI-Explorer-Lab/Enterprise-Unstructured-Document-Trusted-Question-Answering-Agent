# Enterprise Unstructured Document Trusted QA Agent

面向企业 PDF 文档的可信问答服务。系统负责将 PDF 解析、切分并建立索引，再通过混合检索、重排、证据门控和带引用的回答生成完成问答。

当前运行链路是：

```text
PDF 文件
  -> 文档解析与结构化切分
  -> Embedding / pgvector + BM25 混合检索
  -> Cross-Encoder 重排
  -> Agent / skill / planner 编排
  -> 证据门控
  -> 带引用的 SSE 问答响应
```

## 主要能力

- PDF 本地路径索引或文件上传索引。
- 同步索引和后台异步索引，并提供任务进度查询。
- Dense retrieval 与 BM25 sparse retrieval 的混合检索。
- Cross-Encoder 候选重排、缓存和近重复结果处理。
- 意图与槽位理解、技能路由、查询扩展、会话上下文和受约束的 Agent workflow。
- 证据不足时拒答或重试，并返回文档、页码、标题路径和引用片段。

## 环境要求

- Python 3.11；Docker 镜像也以 Python 3.11 为基础。
- PostgreSQL，以及项目需要的 pgvector 能力；默认配置使用 `trusted_qa` 数据库。
- 可用的 LLM、Embedding 和 MinerU 配置。默认配置位于 `config/app.yaml`，API Key 建议放在环境变量中。
- Cross-Encoder 模型首次使用时可能需要下载模型文件；是否预加载由配置决定。

项目依赖见 [`requirements.txt`](requirements.txt)。

## 本地启动

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.template .env
```

然后根据实际环境填写 `.env`，至少检查以下配置：

- `LLM_PROVIDER`、`LLM_MODEL`、对应的 LLM API Key。
- Embedding API Key。
- `MINERU_API_KEY`（使用远程 MinerU 时）。
- `PGVECTOR_DATABASE_URL` 和 `PGVECTOR_EMBEDDING_DIM`。

启动开发服务：

```bash
python -m uvicorn main:app --reload
```

服务默认监听 `http://127.0.0.1:8000`。启动后可访问：

```bash
curl http://127.0.0.1:8000/health
```

生产或容器启动方式：

```bash
docker build -t trusted-qa .
docker run --env-file .env -p 8000:8000 trusted-qa
```

容器只启动应用本身；PostgreSQL、pgvector、LLM、Embedding 和 MinerU 等外部依赖仍需单独提供。

## API

### 健康检查

```text
GET /health
```

### 文档管理与索引

```text
GET  /documents/companies
GET  /documents/collections
POST /documents/index
POST /documents/index/start
POST /documents/upload
POST /documents/upload/start
GET  /documents/tasks/{task_id}
```

`/documents/index` 接收 JSON，常用字段如下：

```json
{
  "pdf_path": "/absolute/path/report.pdf",
  "collection_name": "annual-report",
  "company_id": "example",
  "year": 2025,
  "force_rebuild": false
}
```

`/documents/upload` 和 `/documents/upload/start` 使用 multipart 文件上传；后台版本返回任务信息，再通过 `/documents/tasks/{task_id}` 查询进度。

### 问答

```text
POST /qa/ask/stream
```

当前问答接口通过 Server-Sent Events（SSE）返回进度和最终结果，不是普通的 `POST /qa/ask` JSON 接口。请求体示例：

```json
{
  "question": "2025 年的营业收入是多少？",
  "collection_name": "annual-report",
  "session_id": "",
  "top_k": 5,
  "expand_query_num": 3,
  "enable_cache": true,
  "include_debug": false
}
```

问答前，目标 collection 必须已经完成索引并包含可检索的文档。最终事件包含回答、决策、查询类型、置信度、会话 ID 和 citations；证据不足时可能返回拒答决策。

### 会话

```text
GET    /qa/sessions?collection_name=annual-report&limit=30&offset=0
GET    /qa/sessions/{session_id}
DELETE /qa/sessions/{session_id}
```

## 配置

主配置文件是 [`config/app.yaml`](config/app.yaml)，环境变量模板是 [`.env.template`](.env.template)。主要配置区段包括：

| 区段 | 作用 |
| --- | --- |
| `llm` | LLM provider、模型、endpoint、超时和重试 |
| `embedding` | Embedding provider、模型和向量维度 |
| `storage` / `vector` | pgvector 与本地开发存储配置 |
| `pdf` | PDF 解析器、MinerU 和页数限制 |
| `retrieval` | 混合检索、查询扩展和并发控制 |
| `reranker` | Dense/BM25 权重和 Cross-Encoder |
| `agent` / `workflow` / `planner` / `skills` | Agent 编排、规划和技能路由 |
| `guardrails` | 最低证据数量、分数阈值和拒答策略 |

默认配置的存储后端是 `pgvector`。`local_dev` 配置可用于本地开发场景，但不等同于生产数据库部署方案。

## 测试与验收

项目的 pytest 测试目录由 [`pytest.ini`](pytest.ini) 指定为 `qa_tests`：

```bash
python -m pytest -q
```

常用的定向验证脚本：

```bash
# 工作流级验收：使用脚本内置的测试文档块
python scripts/workflow_acceptance.py

# ASGI API 端到端验收：覆盖索引和 SSE 问答链路
python scripts/e2e_acceptance.py

# 数据库/向量 schema 冒烟检查
python scripts/pgvector_smoke.py
```

这些命令的验证范围不同。通过 pytest 或脚本只表示对应的本地验证通过，不代表外部 LLM、Embedding、MinerU 或生产数据库已经可用。

## 目录结构

```text
main.py                 FastAPI 应用和生命周期管理
controller/apis/        HTTP 路由
domain/                 请求、响应、会话和引用模型
service/pdf/            PDF 解析、切分、索引队列和进度
service/retrieval/      混合检索、重排、缓存和存储访问
service/agent/          Agent workflow、planner、技能和证据门控
service/embedding/      Embedding 服务与缓存
service/llm/            LLM 客户端
config/                 YAML 配置和领域配置
qa_tests/               pytest 测试
scripts/                工作流、API 和数据库验收脚本
skills/                 可路由的问答技能定义
```

## 常见排查

- 启动时报数据库连接错误：先检查 PostgreSQL 是否运行，以及 `.env` 中的数据库 URL 是否覆盖了 `config/app.yaml` 的默认值。
- 问答返回 collection 不存在或没有证据：先完成文档索引，并确认问答请求中的 `collection_name` 与索引时一致。
- SSE 没有最终事件：查看服务日志和响应中的错误事件；同时确认索引任务没有仍处于运行状态。
- Cross-Encoder 启动失败：检查模型下载、网络、磁盘空间和 `reranker` 相关配置；本地测试可按实际环境关闭预加载。
- 外部模型服务不可用：不要把本地路由、解析或 schema 测试的通过结果当作真实模型调用成功。
