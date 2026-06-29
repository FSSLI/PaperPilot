# PaperPilot - 文献知识库问答系统

> 上传文献，微信提问，AI 基于你的文献库智能回答。

## 项目简介

读研期间，下载的 PDF 文献往往散落在各个文件夹中，遇到技术细节需要回顾时不得不挨个打开论文查找，面对大量英文文献还要频繁切换翻译软件——费时费力。

PaperPilot 为解决这一痛点而设计：将所有文献上传到知识库，系统自动解析并向量化存储。研究中遇到问题时，直接在微信提问，系统基于你的全部文献智能检索并回答。

![screenshot](images/screenshot_00.png)

## 功能亮点

- **微信直接提问**：在微信客服中发送问题，基于 LangChain RAG 从文献库中检索相关内容，LLM 生成回答
- **微信发送文件入库**：直接在微信聊天中发送 PDF/Word 等文件，自动上传到 MinIO 并触发索引
- **事件驱动索引**：基于 MinIO Bucket Notification，文献上传/删除自动触发解析和向量化
- **多格式解析**：支持 PDF、Word、Excel、PPT、Markdown、纯文本，内置高精度 MinerU 解析器
- **两阶段检索**：向量相似度检索 + Rerank 重排序，提升回答准确性
- **共享上传**：组内成员通过口令上传文献，无需管理员账号
- **可选 Langfuse 可观测性**：追踪 LLM 调用的 token 消耗和检索效果

## 系统架构

```mermaid
graph LR
    subgraph Users["用户端"]
        WU["微信用户"]
        AU["管理员"]
    end

    subgraph App["应用服务 (Python / Flask)"]
        WXKF["mildoc_wxkf<br/>微信问答接口<br/>:8890"]
        IDX["mildoc_index<br/>文献索引服务"]
        ADM["mildoc_admin<br/>文献管理后台<br/>:8870"]
    end

    subgraph Infra["基础设施 (Docker)"]
        MINIO["MinIO<br/>对象存储"]
        MILVUS["Milvus<br/>向量数据库"]
        ETCD["etcd<br/>元数据"]
    end

    subgraph External["外部服务"]
        LLM["DashScope LLM<br/>通义千问"]
    end

    WU -->|"① 发送 PDF 文件"| WXKF
    WXKF -->|"② 下载并上传"| MINIO
    MINIO -->|"③ Bucket Notification"| IDX
    IDX -->|"④ 解析 + 向量化"| MILVUS

    WU -->|"⑤ 发送问题"| WXKF
    WXKF -->|"⑥ Retrieve + Rerank"| MILVUS
    WXKF -->|"⑦ 生成回答"| LLM
    LLM -->|"⑧ 返回答案"| WU

    AU -->|"管理文献"| ADM
    ADM -->|"上传/删除"| MINIO
    ADM -->|"查看向量"| MILVUS
```

## 技术栈

| 层面 | 技术 |
|------|------|
| 应用框架 | Python 3.12, Flask, uv |
| RAG 框架 | LangChain (Retrieve + Rerank) |
| 向量数据库 | Milvus 2.6.7 |
| 对象存储 | MinIO |
| 文献解析 | MinerU, pdfplumber, python-docx, openpyxl |
| Embedding | DashScope text-embedding-v4 |
| LLM | DashScope qwen-plus |
| 可观测性 | Langfuse（可选） |
| 容器化 | Docker Compose (Milvus + etcd + MinIO) |
| 内网穿透 | frp |
| 微信集成 | 企业微信客服 API |

## 使用流程

```
上传文献                    微信提问                    得到回答
   │                          │                          │
   ▼                          ▼                          ▼
┌──────┐   自动解析    ┌──────────┐   RAG 检索    ┌──────────┐
│ PDF/ │──────────────▶│ 知识库   │◀──────────────│ 微信     │
│ Word │   自动向量化  │ (Milvus) │   LLM 生成    │ 客服     │
└──────┘              └──────────┘              └──────────┘
  3种方式上传：                                   回答基于你的
  管理后台 / 共享页面 / 微信直接发文件               全部文献内容
```

## 部署指南

详细的部署步骤请参考 [部署文档](#部署文档)，支持两种部署架构：

- **架构 A**：全云服务器部署（≥ 4核8G）
- **架构 B**：云服务器 + 本地 + frp 穿透（2核2G 可用）

### 快速开始

```bash
# 1. 克隆项目
git clone https://github.com/FSSLI/PaperPilot.git
cd PaperPilot

# 2. 配置环境变量
cp mildoc_index/.env.example mildoc_index/.env
cp mildoc_admin/.env.example mildoc_admin/.env
cp mildoc_wxkf/.env.example mildoc_wxkf/.env
# 编辑各 .env 文件，填入你的 API Key 和配置

# 3. 启动基础设施（Docker）
cd mildoc_milvus/milvus_local/
docker compose up -d

# 4. 启动应用服务
cd ../../mildoc_index && uv sync && uv run main.py --provider minio --mode listen
cd ../mildoc_admin && uv sync && uv run admin_app.py
cd ../mildoc_wxkf && uv sync && uv run wxkf_callback_app.py
```

---

# 部署文档

以下是完整的分步部署教程，包含截图和详细说明。

## 第一步：环境准备

### 1. 安装 Python 3.12 和 uv 包管理器

```bash
conda create -n mildoc python=3.12
conda activate mildoc
pip install uv
```

### 2. 下载 frp 内网穿透工具

从 GitHub 下载 frp（推荐 v0.61.1）：https://github.com/fatedier/frp/releases

解压后包含 frpc.exe（客户端）和 frpc.toml（配置文件）。

> **注意**：Windows Defender 可能误删 frpc.exe，需在"病毒和威胁防护 → 排除项"中添加排除文件夹。

## 第二步：Milvus 向量数据库部署

### 1. 克隆项目代码

```bash
git clone https://github.com/FSSLI/PaperPilot.git
cd PaperPilot
```

> **注意**：项目的 `.env` 配置文件不会被提交到 Git 仓库中。克隆项目后，请将各服务目录下的 `.env.example` 复制为 `.env`，然后填入你自己的配置值：
> ```bash
> cp mildoc_index/.env.example mildoc_index/.env
> cp mildoc_admin/.env.example mildoc_admin/.env
> cp mildoc_wxkf/.env.example mildoc_wxkf/.env
> ```

### 2. 设置 Docker 存储卷

创建 `volumes` 文件夹用于持久化 Milvus 数据，并在 docker-compose.yml 同级目录下配置 `.env` 文件。

**Windows 本地部署（推荐）**：

```bash
# 创建存储目录（数据持久化到本地，重启不丢失）
mkdir volumes/milvus
mkdir volumes/etcd
mkdir volumes/minio
```

`.env` 文件内容：

```properties
DOCKER_VOLUME_DIRECTORY=.
```

**Linux / WSL2 部署**：

```bash
mkdir -p /docker_data_volume/milvus_local
```

`.env` 文件内容：

```properties
DOCKER_VOLUME_DIRECTORY=/docker_data_volume/milvus_local
```

> **注意**：`volumes` 目录必须存在，否则 Milvus 数据仅在容器内存储，重启容器后数据会丢失。

### 3. 运行 Milvus

```bash
# 进入项目目录
cd mildoc_milvus/milvus_local/

# 拉取镜像文件
docker pull quay.io/coreos/etcd:v3.5.25
docker pull minio/minio:RELEASE.2024-12-18T13-15-44Z
docker pull milvusdb/milvus:v2.6.7
docker pull zilliz/attu:v2.6

# 启动（后台运行）
docker compose up -d

# 查看日志
docker compose logs -f

# 停止
# docker compose down
```

### 4. MinIO 管理后台

访问 http://localhost:9001

用户名：`minioadmin` 密码：`minioadmin`

![screenshot](images/screenshot_02.png)

### 5. 创建 mildoc 桶

![screenshot](images/screenshot_03.png)

![screenshot](images/screenshot_04.png)

### 6. Milvus 管理后台（Attu）

启动 Attu（**必须加 `--network milvus`**，否则无法连接 Milvus）：

```bash
docker run -d --rm --network milvus -p 8000:3000 --name attu26 -e MILVUS_HOST=milvus-standalone zilliz/attu:v2.6
```

> **注意**：`docker compose up -d` 启动的 Milvus 容器在 `milvus` 网络中。Attu 如果不在同一网络，即使填写 `milvus-standalone:19530` 也会报 DNS 解析失败。`--network milvus` 参数让 Attu 加入同一网络。

访问 http://localhost:8000/

使用 Milvus 启动时设置的用户名密码登录：比如 root / admin123

在连接页面，Milvus 地址填写 `milvus-standalone:19530`（**不要填 127.0.0.1**，因为 Attu 也是容器，127.0.0.1 指向的是 Attu 容器自身）。

![screenshot](images/screenshot_05.png)

![screenshot](images/screenshot_06.png)

![screenshot](images/screenshot_07.png)

### 7. 创建 mildoc 向量数据库

> **重要**：此步骤必须在启动 `mildoc_index` 之前完成，否则服务启动时会报 `database not found[database=mildoc]` 错误。

**方法一：通过 Attu 图形界面**

进入 Attu（http://localhost:8000），在首页点击创建。

填写名字：`mildoc`

选择时区：`beijing`

![screenshot](images/screenshot_08.png)

**方法二：通过命令行（推荐）**

```python
pip install pymilvus
python -c "
from pymilvus import connections, db
connections.connect(host='127.0.0.1', port='19530', user='root', password='admin123')
db.create_database('mildoc')
print('数据库列表:', db.list_database())
"
```

### 8. 其他部署模式

除上述 milvus_local 模式外，项目还提供了两种部署模式：

#### 8.1 milvus_minio 模式（本地 Milvus + 外部 MinIO）

适用场景：已有独立的 MinIO 服务器，希望将 Milvus 和 MinIO 分离部署。

此模式下，docker-compose.yml 仅包含 etcd 和 Milvus，MinIO 地址指向外部服务器（如 172.31.154.203:9000）。

docker-compose.yml 中的关键配置差异：

```yaml
MINIO_ADDRESS: 172.31.154.203
MINIO_PORT: 9000
MINIO_ACCESS_KEY_ID: <你的 access key>
MINIO_SECRET_ACCESS_KEY: <你的 secret key>
MINIO_USE_SSL: false
MINIO_BUCKET_NAME: milvus
```

`.env` 配置：

```properties
# Windows 本地部署用 . ，Linux/WSL2 用 /docker_data_volume/milvus_minio
DOCKER_VOLUME_DIRECTORY=.
```

#### 8.2 milvus_oss 模式（本地 Milvus + 阿里云 OSS）

适用场景：使用阿里云 OSS 作为对象存储，无需自建 MinIO。

此模式下，Milvus 的对象存储指向阿里云 OSS，需要配置 OSS 的 AccessKey 和 Bucket。

docker-compose.yml 中的关键配置差异：

```yaml
MINIO_ADDRESS: oss-cn-hangzhou.aliyuncs.com
MINIO_PORT: 443
MINIO_ACCESS_KEY_ID: <你的阿里云 AccessKey ID>
MINIO_SECRET_ACCESS_KEY: <你的阿里云 AccessKey Secret>
MINIO_USE_VIRTUAL_HOST: true
MINIO_USE_SSL: true
MINIO_BUCKET_NAME: milvus-alioss
MINIO_ROOT_PATH: milvus_db
MINIO_REGION: cn-hangzhou
```

`.env` 配置：

```properties
# Windows 本地部署用 . ，Linux/WSL2 用 /docker_data_volume/milvus_oss
DOCKER_VOLUME_DIRECTORY=.
```

## 补充步骤：frp 内网穿透配置（架构 B 需要）

如果你的云服务器配置较低（如 2 核 2G），无法同时运行 Docker 基础设施和 Python 应用服务，可以使用 frp 内网穿透方案：

云服务器仅运行 Milvus + etcd + MinIO（Docker），Python 服务运行在本地，通过 frp 将本地端口穿透到云服务器公网。

### 1. 云服务器安全组配置

在云服务器的安全组中开放以下端口：

- 7000（frp 通信端口）
- 8870（mildoc_admin 管理后台）
- 8890（mildoc_wxkf 微信客服接口）

### 2. 云服务器端配置（frps）

将 frps 部署在云服务器上，配置文件 `frps.toml`：

```toml
bindPort = 7000
```

启动命令：

```bash
./frps -c frps.toml
```

### 3. 本地客户端配置（frpc）

在本地开发机上配置 `frpc.toml`：

```toml
[common]
server_addr = <你的云服务器公网IP>
server_port = 7000

[admin]
type = tcp
local_ip = 127.0.0.1
local_port = 8870
remote_port = 8870

[wxkf]
type = tcp
local_ip = 127.0.0.1
local_port = 8890
remote_port = 8890
```

### 4. 启动 frpc

在本地运行：

```bash
./frpc -c frpc.toml
```

启动成功后，外部可通过 `http://<云服务器公网IP>:8870` 访问管理后台，`http://<云服务器公网IP>:8890` 访问微信客服接口。

### 5. 注意事项

- Windows Defender 可能误删 frpc.exe，需在"病毒和威胁防护 → 管理设置 → 排除项 → 添加排除文件夹"中排除 frp 所在目录。
- frpc 需要在本地 Python 服务启动后再运行，否则穿透的端口无服务响应。
- 企业微信回调 URL 使用穿透后的公网地址：`http://<云服务器公网IP>:8890/callback/command`

## 第三步：获取千问云 API Key（原百炼平台）

### 1. 手机号登录/注册千问云

官网地址：https://www.qianwenai.com/

进入工作台：https://platform.qianwenai.com/home/

### 2. 免费额度说明

https://platform.qianwenai.com/home/benefits

![screenshot](images/screenshot_09.png)

### 3. 创建 API Key

https://platform.qianwenai.com/home/api-keys

![screenshot](images/screenshot_10.png)

### 4. 保存好 API Key

保存好你的 API Key，后续项目中会用到。

## 第四步：运行文献索引服务 mildoc_index

### 1. 进入项目目录

```bash
cd mildoc_index
```

### 2. 修改配置文件 .env

补充密钥：

```properties
OPENAI_API_KEY=你的API密钥
```

### 3. 运行服务

```bash
# 安装依赖
uv sync

# 调试运行
uv run main.py --provider minio --mode listen

# 后台运行
nohup uv run main.py --provider minio --mode listen >> mildoc_index.log 2>&1 &

# 查看日志
tail -f mildoc_index.log
```

## 第五步：注册企业微信

### 1. 注册企业微信

官网网址：https://work.weixin.qq.com/

注册地址：https://work.weixin.qq.com/wework_admin/register_wx

![screenshot](images/screenshot_11.png)

![screenshot](images/screenshot_12.png)

### 2. 熟悉了解管理后台

注册完成后，登录并熟悉管理后台。

![screenshot](images/screenshot_13.png)

## 第六步：运行文献管理后台 mildoc_admin

### 1. 进入项目目录

```bash
cd mildoc_admin
```

### 2. 修改配置文件 .env

```properties
# 共享上传口令（组内同学共用）
SHARE_UPLOAD_PASSPHRASE=<自定义口令>
```

### 3. 运行服务

```bash
# 安装依赖
uv sync

# 调试运行
uv run admin_app.py

# 后台运行
nohup uv run gunicorn --workers 1 --bind 0.0.0.0:8870 admin_app:app >> mildoc_admin.log 2>&1 &
```

### 4. 登录管理后台

访问 http://127.0.0.1:8870

默认账户密码（可在 `.env` 中修改）：

```properties
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin123
```

### 5. 共享上传功能

管理后台内置了共享上传页面，允许组内同学通过口令（passphrase）上传文档，无需管理员账号登录。

访问地址：`http://<管理后台地址>:8870/share/upload`

在 `.env` 中配置共享上传口令：

```properties
SHARE_UPLOAD_PASSPHRASE=<自定义口令>
```

## 第七步：运行微信问答接口 mildoc_wxkf

### 1. 在企业微信中创建自建应用

1.1 登录企微管理后台，创建自建应用

![screenshot](images/screenshot_14.png)

1.2 进入自建应用，获取 AgentId、Secret 密钥

![screenshot](images/screenshot_15.png)

1.3 查看企业微信 CorpID

![screenshot](images/screenshot_16.png)

### 2. 准备配置参数

修改项目配置文件 `.env`，填入企微自建应用相关参数：

```properties
# 微信应用配置
CORP_ID=<你的企业微信 CorpID>
AGENT_ID=<你的自建应用 AgentId>
APP_SECRET=<你的自建应用 Secret>

# 自行设置
TOKEN=<自定义一个随机字符串>
ENCODING_AES_KEY=<自定义一个43位随机字符串>
```

阿里云百炼平台 API Key：

```properties
# LLM配置 百炼
LLM_API_KEY=

# LLM Embedding配置 百炼
LLM_EMBEDDING_API_KEY=

# LLM rerank 配置 百炼
RERANK_API_KEY=
```

### 3. 进入项目目录，运行服务

```bash
cd mildoc_wxkf
uv sync

# 调试运行
uv run wxkf_callback_app.py

# 后台运行
nohup uv run gunicorn --workers 1 --bind 0.0.0.0:8890 wxkf_callback_app:app >> mildoc_wxkf.log 2>&1 &
```

### 4. 开放公网访问端口

4.1 在阿里云 ECS 控制台，找到服务器，再找到"安全组"标签。

![screenshot](images/screenshot_17.png)

4.2 在安全组详情页，找到"入方向"、"添加规则"

访问来源：`0.0.0.0`

![screenshot](images/screenshot_18.png)

### 5. 测试页面

访问 `http://{ECS的公网IP}:8890/`

### 6. 配置应用回调接口

6.1 在企微管理后台，打开应用详情页

![screenshot](images/screenshot_19.png)

6.2 填写回调地址和相关加密 token 信息

加密 token 可以从项目配置文件中找到相关配置。

回调地址：`http://{ECS服务器的公网IP}:8890/callback/command`

![screenshot](images/screenshot_20.png)

### 7. 配置企业可信 IP

![screenshot](images/screenshot_21.png)

### 8. 创建并配置客服账号

8.1 在应用管理中，进入"微信客服"

![screenshot](images/screenshot_22.png)

8.2 创建账号 & 设置可调用应用

![screenshot](images/screenshot_23.png)

![screenshot](images/screenshot_24.png)

![screenshot](images/screenshot_25.png)

8.3 将自建应用和客服账号建立连接

![screenshot](images/screenshot_26.png)

![screenshot](images/screenshot_27.png)

![screenshot](images/screenshot_28.png)

8.4 查看/分享微信客服

![screenshot](images/screenshot_29.png)

### 9. 企业微信 IP 白名单

使用 frp 穿透时，企业微信 API 主动调用（如发送消息）的出口 IP 是你本地的公网 IP（非云服务器 IP）。

需要在企业微信管理后台 → 应用管理 → 自建应用 → 企业可信 IP 中，添加你本地的出口 IP。

查看本地出口 IP：浏览器访问 https://ipinfo.io 或命令行执行 `curl ipinfo.io`

如果报错提示 `from ip: xxx.xxx.xxx.xxx`，将该 IP 添加到白名单即可。

### 10. 可选配置：Langfuse 可观测性

项目集成了 Langfuse，用于追踪 LLM 调用的 token 消耗和检索效果。如需启用，在 `.env` 中配置：

```properties
LANGFUSE_ENABLE=true
LANGFUSE_SECRET_KEY=<你的 secret key>
LANGFUSE_PUBLIC_KEY=<你的 public key>
LANGFUSE_BASE_URL=<你的 Langfuse 服务地址>
```

---

## 项目结构

```
mildoc_202601/
├── mildoc_milvus/             # Milvus 部署配置（3种模式）
│   ├── milvus_local/          # 全本地部署：Milvus + etcd + MinIO
│   ├── milvus_minio/          # 混合部署：本地 Milvus + 外部 MinIO
│   └── milvus_oss/            # 云存储部署：本地 Milvus + 阿里云 OSS
├── mildoc_index/              # 文献索引服务
├── mildoc_admin/              # 文献管理后台
├── mildoc_wxkf/               # 微信问答接口
├── images/                    # 文档截图
├── pyproject.toml             # Python 项目配置
├── requirements-all.txt       # 全部依赖列表
├── ROADMAP.md                 # 项目路线图
└── README.md                  # 本文档
```

## 常见问题

### Milvus 报 `database not found[database=mildoc]`

**原因**：Milvus 启动后默认只有 `default` 数据库，项目配置使用 `mildoc` 数据库，但未创建。

**解决**：参考"第二步 7. 创建 mildoc 向量数据库"，通过 Attu 或命令行创建该数据库。

### Attu 连接 Milvus 报 DNS 解析失败

**报错**：`Name resolution failed for target dns:milvus-standalone:19530`

**原因**：Attu 容器和 Milvus 容器不在同一个 Docker 网络中。`docker compose up -d` 会创建一个名为 `milvus` 的独立网络，而单独 `docker run` 启动的 Attu 默认在 `bridge` 网络中。

**解决**：启动 Attu 时加 `--network milvus` 参数。如果 Attu 已经在运行，可以手动加入网络：

```bash
docker network connect milvus attu26
```

### Milvus 重启后数据丢失（数据库/Collection 消失）

**原因**：`volumes` 目录不存在，Docker 挂载路径时创建了空目录但数据仅在容器内。容器重启后数据丢失。

**解决**：
1. 确保 `DOCKER_VOLUME_DIRECTORY=.` 指向本地目录
2. 在 `docker-compose.yml` 同级目录下创建 `volumes/milvus`、`volumes/etcd`、`volumes/minio`
3. 重启 Milvus 后数据将持久化到这些目录

### Attu 页面中 Milvus 地址填什么

- Attu 是 Docker 容器时：填 `milvus-standalone:19530`（容器名），**不能填 `127.0.0.1`**（指向 Attu 容器自身）
- Attu 直接运行在宿主机时：填 `127.0.0.1:19530`

### Windows Defender 误删 frpc.exe

**解决**：在"Windows 安全中心 → 病毒和威胁防护 → 管理设置 → 排除项 → 添加排除文件夹"中排除 frp 所在目录，然后重新下载。

### 企业微信报 `from ip: xxx.xxx.xxx.xxx` 不在白名单

**原因**：使用 frp 穿透时，企业微信 API 调用的出口 IP 是你本地的公网 IP，不是云服务器 IP。

**解决**：在企微管理后台 → 应用管理 → 自建应用 → 企业可信 IP 中，添加该出口 IP。查看出口 IP：浏览器访问 https://ipinfo.io

---

## License

MIT License
