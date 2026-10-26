# Docker 部署与服务器准备

2026-10-26。用户已恢复 Docker 部署准备，服务器尚未租用。

当前镜像运行**基础 Add/Search 服务**：BM25／规则检索、原始图文持久化、用户范围隔离。它尚未包含 V7 实验的 Dense＋重排在线流水线。健康接口明确报告 dense 和 reranker 禁用；不能把 94.62% 实验 Recall@20 作为该容器的线上成绩。设置不支持的模式会拒绝启动。

## 服务器选择

初期单机试用建议 Ubuntu 24.04 LTS x86_64、4 vCPU、8 GB 内存、80 GB SSD。这是工程起步估算，尚无容量或并发 SLA；媒体数据量增加时需要扩容磁盘。当前基础服务无需 GPU。

真正部署 Dense＋重排还需要在线 embedding 索引和常驻重排模型。可把 CPU API 与 GPU scorer 分开；GPU 显存应由实际服务批量、输入长度和负载验证决定，本包暂不要求特定 GPU。服务器需要能访问所选 provider 官方端点；本次准备不自动调用模型。

先保留宿主端口 127.0.0.1:8765，用 SSH 隧道验收。公网 API 需后续域名／TLS 网关。共享服务 token 适用于可信上游代理各个 user_id，不提供不可信用户直接接入所需的 token 与用户范围授权。

## 镜像与验证

Python 3.11 slim Bookworm 的 OCI index 已固定到 sha256:f06d01d4ba3f8ad5af45a3ec87df19c1d317d8e8961edc0fabda61811af6f7cf，2026-10-26 从 Docker 官方 registry 核实；Gunicorn 26.2.0 wheel 也固定 SHA256。实际验证平台以 CI 摘要为准。

一个进程、四线程，SQLite 持久卷位于 /data。容器 UID/GID 10001、只读根目录、临时 /tmp、无 capabilities、禁止新增权限。默认 CPU 上限 2 核、内存 1 GB、128 PID；日志每文件 10 MB，保留 3 个。这些是限制，不是实测容量保证。构建允许列表排除密钥、数据、研究结果、账本和 Git 历史。

本机 Windows 没有 Docker CLI／可用 Linux Engine，本地预检因此失败，这不等于镜像失败或成功。GitHub 工作流 [.github/workflows/docker.yml](../.github/workflows/docker.yml) 在 Ubuntu 24.04 执行真实验证。对应提交的成功运行与 JSON 摘要才是镜像通过的证据。

在已有 Docker Engine 和 Compose v2 的 Linux 或 Docker Desktop Linux containers 主机执行：

~~~bash
python3 scripts/verify_container.py --output artifacts/container-validation.json
~~~

它使用随机项目、随机测试 token、合成数据和独立卷，依次验证构建、健康、认证、非 root／只读配置、资源限制、回环端口、幂等冲突、用户隔离和图文顺序；重建容器后查询原数据；再导出备份、恢复到另一新卷并核对查询。最后只删除自己的测试容器、卷和临时镜像，不做全局 prune，不使用已有服务数据，不调用模型或官方评测。

## 新服务器启动

按 [Docker 官方 Ubuntu 安装文档](https://docs.docker.com/engine/install/ubuntu/)安装 Engine、Buildx 和 Compose plugin。先确认 docker version 和 docker compose version 可用。以下命令使用有 Docker 权限的同一登录用户。

获取私有仓库部署分支。服务器须先配置有权读取该仓库的 Git 认证；不要把 token 填进命令 URL。

~~~bash
git clone --branch codex/v6-research-docker git@github.com:cenzihan/trace-mem-mm.git
cd trace-mem-mm
git rev-parse HEAD
python3 scripts/verify_container.py --output artifacts/server-container-validation.json
~~~

确认该提交验证通过，再启动运行实例。所有命令在仓库根目录执行，固定项目名 tracemem，避免目录名变化导致使用另一套卷：

~~~bash
python3 scripts/init_container_secret.py
docker compose -p tracemem config --quiet
docker compose -p tracemem build --pull
docker compose -p tracemem up -d --wait --wait-timeout 90
curl --fail http://127.0.0.1:8765/health
~~~

初始化 secret 仅首次执行，拒绝覆盖既存 token，不输出值。POSIX 父目录 0700，文件 0644，供容器单文件挂载读取。此服务 token 与 OpenAI／百炼密钥无关；Add/Search 客户端使用 Authorization: Bearer 加该值，不写进 URL、仓库或日志。

从自己的电脑建立 SSH 隧道：

~~~bash
ssh -N -L 8765:127.0.0.1:8765 ubuntu@SERVER_IP
~~~

此后本机 http://127.0.0.1:8765/health 指向服务器，Add/Search 仍需认证。本机 8765 已占用时修改隧道左侧端口即可。实例不会自动开放公网端口。

支持非敏感环境变量：TRACEMEM_PORT（8765）、TRACEMEM_IMAGE（trace-mem-mm:deterministic-preview）、TRACEMEM_MEMORY_LIMIT（1g）、TRACEMEM_CPUS（2.0）、TRACEMEM_SECRET_FILE（./.secrets/service_token）。模型 key 不应放进这些变量。

## 备份、恢复和更新

不要直接复制仍在写入的 SQLite 文件。工具通过 SQLite backup API 导出，检查完整性并生成 SHA256 清单；输出目录必须全新，父目录必须存在：

~~~bash
mkdir -p backups
python3 scripts/container_backup.py backup --project tracemem --destination backups/before-update-001
~~~

目录含 database.sqlite 和 manifest.json，记录哈希、大小、实际镜像 ID 和操作者 checkout 版本。备份可能含真实用户内容，需私有保存并复制到独立存储；Git 忽略规则不等于加密。

恢复验证哈希、清单和 SQLite 完整性，只允许没有已有容器或卷的新项目，且不自动启动 API。保留旧服务和旧卷，先把恢复实例放在另一个端口：

~~~bash
python3 scripts/container_backup.py restore --backup backups/before-update-001 --project tracemem-restored
TRACEMEM_PORT=8766 docker compose -p tracemem-restored up -d --wait --wait-timeout 90
curl --fail http://127.0.0.1:8766/health
~~~

生产恢复后用调用方自己的已知请求核对内容，不要向生产卷运行会写入合成记录的 smoke_test.py seed。

更新前先备份、记录旧 commit 和镜像 ID。获取已审查提交、构建，然后同一项目 up -d --wait 会保留卷。停止用 docker compose -p tracemem down。**运行实例不要使用 down -v**；验证工具的 --volumes 仅作用于自己的随机测试项目。尚无跨版本数据库迁移回滚保证，恢复需兼容备份的镜像并保留旧卷。

## Dense＋重排服务化后续

1. 在线 Add 持久化原文原图，以 Linux secret 注入 embedding provider，维护每用户动态向量索引；完成必要索引后才返回成功。
2. Search 生成查询向量，在对应用户索引取 Top100，再交 GPU scorer 排序。冻结实验缓存不能替代新用户在线索引。
3. GPU 服务固定模型、常驻加载，增加批处理、队列、超时与故障策略；每次 HTTP 请求不应启动实验 worker。
4. 验证未知 provider 结果恢复、并发和重启一致性，再用独立场景验收。已关闭实验 guard 和账本不复用为服务状态。

平台调用 Add/Search；Reader 与 Judge 属实验评估，不自动加入每次 Search。Gunicorn 活性和停机参数不是逐请求截止时间，也不构成 30 分钟 SLA；公网网关、64 MiB HTTP 上限和负载能力需服务器到位后验证。

依据：[Compose secrets](https://docs.docker.com/compose/how-tos/use-secrets/)、[Compose 服务配置](https://docs.docker.com/reference/compose-file/services/)、[Docker volumes](https://docs.docker.com/engine/storage/volumes/)、[GitHub runner](https://docs.github.com/en/actions/how-tos/write-workflows/choose-where-workflows-run/choose-the-runner-for-a-job)、[V7/V7.1 结果](v7-independent-scene.md)、[协议](protocol.md)。
