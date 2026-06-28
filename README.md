# PaperPilot - 文献知识库问答系统

## 第一步：系统介绍

### 1. 项目简介

读研期间，阅读文献是日常工作的核心环节。然而，下载的 PDF 文献往往散落在各个文件夹中，没有系统整理。遇到某个技术细节需要回顾时，不得不挨个打开论文查找；面对大量英文文献，还需要频繁切换翻译软件辅助阅读——整个过程费时费力，效率很低。

PaperPilot 正是为解决这一痛点而设计的智能文献知识库系统。你可以将所有下载的文献上传到知识库，系统会自动解析文档内容并进行向量化存储。当你在阅读或研究中遇到问题时，只需通过微信直接提问，系统会基于你上传的全部文献进行智能检索和问答，帮你快速找到答案。相比传统的手动翻阅 + 翻译的方式，效率提升显著。

![screenshot](images/screenshot_00.png)

### 2. 系统概述

本系统基于 MinIO 对象存储和 Milvus 向量数据库，构建了一套面向科研场景的文献知识库问答解决方案。

系统包含三个核心模块：

- **文献索引服务**（mildoc_index）
- **文献管理系统**（mildoc_admin）
- **微信问答接口**（mildoc_wxkf）

![screenshot](images/screenshot_01.png)

### 3. 业务系统

**mildoc_admin（文献管理系统）**

- 创建/删除文献目录分类
- 上传/删除文献文档
- 查看文献元信息（文件名、MD5、创建时间）
- 查看文献解析状态和切片信息

**mildoc_index（文献索引服务）**

- 监听 MinIO 指定桶的文献上传事件
- 调用文档解析器自动解析文献内容并分片
- 生成文献向量并存储到 Milvus
- 处理文献删除时的向量清理

**mildoc_wxkf（微信问答接口）**

- 接收微信客服转发的问题
- 使用 LangChain 从文献知识库中智能检索相关内容
- 调用 LLM 基于文献内容生成回答
- 返回答案给微信客服系统

### 4. 技术实现

**文档解析器**

- PDF解析器：处理 PDF 文档
- Office解析器：处理 Word、Excel、PowerPoint 文档
- MinerU解析器：高精度文档解析
- Markdown解析器：处理 Markdown 文档
- Text解析器：处理纯文本文档

**Embedding**

- 将文本分片转换为向量表示
- 支持多种向量模型

**LangChain**

- Retrieve：基于向量相似度的文档检索
- Rerank：对检索结果进行重新排序优化

**LLM服务**

- 基于检索到的文献上下文生成智能解答
- 支持多种大语言模型

### 5. 技术特点

**事件驱动架构**

- 基于 MinIO 对象事件的自动化文献处理
- 实时响应文献的上传和删除操作
- 确保存储和向量数据的一致性

**多格式文献支持**

- 支持 PDF、Word、Excel、PowerPoint、Markdown、Text 等格式
- 内置多种专业解析器，包括高精度的 MinerU 解析器
- 统一的文本分片和向量化处理流程

**智能检索问答**

- 基于向量相似度的语义搜索
- LangChain 框架实现 Retrieve + Rerank 优化
- LLM 驱动的上下文感知智能解答

### 6. 数据流转说明

- **文献上传流程**：上传文献 → MinIO 存储 → 触发事件 → 文献解析 → 向量生成 → 存储到 Milvus
- **智能问答流程**：微信提问 → 文献知识库检索 → 匹配相关段落 → LLM 分析 → 返回回答
- **文献管理流程**：管理后台操作 → 查看文献元信息 → 展示解析状态和切片信息
- **数据同步流程**：删除文献 → MinIO 清理 → 事件触发 → Milvus 向量清理

### 7. 项目目录结构

```
mildoc_202601/
├── mildoc_milvus/             # Milvus 部署配置（3种模式）
│   ├── milvus_local/          # 全本地部署：Milvus + etcd + MinIO
│   ├── milvus_minio/          # 混合部署：本地 Milvus + 外部 MinIO
│   └── milvus_oss/            # 云存储部署：本地 Milvus + 阿里云 OSS
├── mildoc_index/              # 文档索引服务
├── mildoc_admin/              # 文档管理后台
├── mildoc_wxkf/               # 微信客服接口
├── frp_0.61.1_windows_amd64/  # frp 内网穿透工具
├── pyproject.toml             # Python 项目配置
└── requirements-all.txt       # 全部依赖列表
```

### 8. 部署架构说明

本项目支持两种部署架构，可根据服务器配置灵活选择：

**架构 A：全云服务器部署（服务器配置 ≥ 4核8G）**

所有 Docker 基础设施（Milvus、etcd、MinIO）和 Python 应用服务（mildoc_index、mildoc_admin、mildoc_wxkf）均运行在同一台云服务器上。适合服务器资源充足的场景。

**架构 B：云服务器 + 本地 + frp 穿透（服务器配置 2核2G 等低配场景）**

云服务器仅运行 Docker 基础设施（Milvus + etcd + MinIO），3 个 Python 应用服务运行在本地开发机上，通过 frp 内网穿透将本地端口映射到云服务器公网，供企业微信回调等外部访问。

各服务端口说明：

| 服务 | 端口 |
|------|------|
| MinIO API | 9000 |
| MinIO 管理后台 | 9001 |
| Milvus API | 19530 |
| Milvus 管理后台 | 9091 |
| mildoc_admin | 8870 |
| mildoc_wxkf | 8890 |
| frp 服务端 | 7000 |

## 第二步：环境准备

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

## 第三步：Milvus 向量数据库部署

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

创建 docker volume 文件夹，并在 docker-compose.yml 同级目录下设置 `.env` 文件：

```bash
mkdir -p /docker_data_volume/milvus_local
```

`.env` 文件内容：

```properties
DOCKER_VOLUME_DIRECTORY=/docker_data_volume/milvus_local
```

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

启动 Attu：

```bash
docker run -d --rm --network milvus -p 8000:3000 --name attu26 -e MILVUS_HOST=milvus-standalone zilliz/attu:v2.6
```

访问 http://localhost:8000/

使用 Milvus 启动时设置的用户名密码登录：比如 root / admin123

> **注意**：Milvus 地址需要使用 `milvus-standalone:19530`

![screenshot](images/screenshot_05.png)

![screenshot](images/screenshot_06.png)

![screenshot](images/screenshot_07.png)

### 7. 创建 mildoc 向量数据库

进入 Attu，在首页点击创建。

填写名字：`mildoc`

选择时区：`beijing`

![screenshot](images/screenshot_08.png)

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
DOCKER_VOLUME_DIRECTORY=/docker_data_volume/milvus_minio
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
DOCKER_VOLUME_DIRECTORY=/docker_data_volume/milvus_oss
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

## 第四步：获取千问云 API Key（原百炼平台）

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

## 第五步：运行文档索引服务 mildoc_index

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

## 第六步：注册企业微信

### 1. 注册企业微信

官网网址：https://work.weixin.qq.com/

注册地址：https://work.weixin.qq.com/wework_admin/register_wx

![screenshot](images/screenshot_11.png)

![screenshot](images/screenshot_12.png)

### 2. 熟悉了解管理后台

注册完成后，登录并熟悉管理后台。

![screenshot](images/screenshot_13.png)

## 第七步：运行文档管理后台 mildoc_admin

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

## 第八步：运行微信客服服务接口 mildoc_wxkf

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
