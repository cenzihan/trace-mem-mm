# 有限额度的真实 embedding 冒烟验收

后续更新（2026-10-26）：用户调整连接配置后，已部署容器内的一次真实 GPT 卡片调用通过，另有独立 GPU 上的 Qwen 重排 smoke 通过。见[最新模型结果与准入准备](competition-readiness-20261026.md)。下文保留三次 embedding 阶段及当时 GPT 地区预检失败的历史记录，不代表最新连接状态。

这份流程验证已部署容器里的真实 `text-embedding-v4`：新资料入库生成向量、查询生成向量，以及认证、幂等、用户隔离和容器重建后的持久化。它使用独立 Compose 项目、固定合成文字和独立持久卷，保留原有基础服务。

这是 **embedding-only 功能验收**。检索方式为 BM25＋dense RRF；没有 GPT 卡片生成、图片识别或 GPU 重排。当前 GPT 路径存在 OpenAI 地区 HTTP 403，处理方式仍待确定，本流程不重试或替换该模型，也不宣称 V7 Dense＋重排已经上线。本文是执行流程，不能代替成功收据或实际部署记录。

## 已验证结果：2026-10-26

用户服务器已在提交 `0ade655b2db5c3db987cec967b03925039cd07d4` 上完成本流程：3 次真实 `text-embedding-v4` 调用均返回 HTTP 200，256 维，输入用量分别为 32、9、13 token，共 54 token。写入、认证拒绝、重复写入、用户隔离和容器重建后的检索均通过；两种查询均将目标原文排第一。只读审计核对了全部 3 份原始收据，并确认数据库中两条来源向量与第一份收据逐项一致。

父端已将原来的 3 项预算预留核对为合计 **¥0.000028 使用量估算**（按每次请求向上取整至微元），没有新增请求或预留，历史账本记录保持不变。这是依据 usage 和已核实单价的估算，不是供应商最终账单。测试容器已停止并移除，私有数据卷保留，临时上传的百炼 key 已删除；基础服务继续健康运行。

[Linux Docker CI](https://github.com/cenzihan/trace-mem-mm/actions/runs/40311915449) 在同一提交上通过，包含实际镜像导入检查、相关离线测试和基础容器生命周期验证。最初镜像启动时发现缺少被间接引用的 Pillow；补齐固定版本及 wheel hash 后重新构建成功，该启动失败未发出模型请求。

GPT 部分仅做过一次未携带凭据的直连预检，收到 `unsupported_country_region_territory` HTTP 403，**没有 GPT 推理请求**。本记录不证明 GPT、视觉理解或重排已上线，也不是官方 Smoke、榜单成绩或正式版本。完整收据、向量和账本只保存在私有运行目录。

## 三次请求与预算

`deploy/embedding_smoke_test.py` 固定如下输入，`expected_embedding_payloads()` 给出完整 provider payload，维度均为 256：

| 顺序 | 操作 | embedding 请求 |
|---|---|---:|
| 1 | 用户 A 一次 Add：钥匙存放位置与无关会议，两条原始文字 | 1，batch=2 |
| 2 | seed 阶段查询钥匙位置 | 1，batch=1 |
| 3 | 重建容器后，用另一问法查询钥匙位置 | 1，batch=1 |

缺失认证、重复 Add、同 request_id 改写、另一个空用户的查询都应增加 **0 次** provider 请求。来源索引保留现有确定性 cards 派生逻辑，包括无事实字段时的尾部空格；预授权 payload 与原生服务实际发出的 payload 有离线回归测试。

准备工具仅在父端**已存在的 CNY 500 项目账本**中预留三个 `.05` 元 slot，共 `.15` 元，不发送请求。每个 slot 固定父端 attempt ID、payload SHA256、模型、北京官方 endpoint 和最大 token 数。服务器不会创建另一份预算账本，不会重置旧实验 guard。grant 自生成起有效两小时；过期、输入变化、乱序、未知结果或任一失败都会阻止继续派发。没有自动重试；不要通过新建 grant 绕过失败。

预算预留与实际使用费不同。必须以私有收据及父端 reconciliation 记录确认调用和费用；客户端的 `state=passed` 只表示 HTTP 功能验收通过，`provider_receipts_verified=false` 提醒收据尚需审计。

## 父端准备 grant

以下为 Bash 示例；在 Windows 父端使用 Python 执行相同脚本参数，并按 PowerShell 语法设置变量。父端应位于该项目代码根目录，且只能指定已有真实项目账本。先实际核对[百炼官方 embedding 价格](https://help.aliyun.com/zh/model-studio/embedding-interfaces-compatible-with-openai)，记录当天 UTC 核对时间；不能仅填入当前时间代替核查。

```bash
export PARENT_LEDGER='/absolute/path/to/existing/runtime/development-budget.sqlite'
export PRICE_CHECKED_AT='REPLACE_WITH_ACTUAL_PRICE_CHECK_UTC_ISO_TIMESTAMP'
mkdir -p runtime
export GRANT_DIR="$PWD/runtime/server-embedding-smoke-grant-001"
python3 scripts/prepare_embedding_smoke_grant.py prepare \
  --ledger "$PARENT_LEDGER" --output "$GRANT_DIR" \
  --price-checked-at "$PRICE_CHECKED_AT" --reserve-live
```

输出目录必须全新，其父目录必须存在。工具输出 `grant_file`、`grant_sha256` 和三个已预留 slot；完整准备记录留在父端。即使中途失败，也保留原目录和已产生的预留，不能换目录重跑。仅将 `grant.json` 经已授权的安全渠道传到服务器，原始账本留在父端；将父端输出的 SHA256 独立保存为服务器的可信配置。

## 服务器私有文件与启动

在服务器代码根目录执行，使用有 Docker 权限的同一用户。以下目录在仓库之外，父目录权限为 0700。需要三个文件：`grant.json`、`service_token` 和已授权的 `bailian_token`。不得写入 Git、镜像、URL、命令参数或日志。

```bash
export TRACEMEM_SMOKE_SECRET_DIR="$HOME/.local/share/tracemem-embedding-smoke"
install -d -m 0700 "$TRACEMEM_SMOKE_SECRET_DIR"
export TRACEMEM_GRANT_SHA256='REPLACE_WITH_64_HEX_HASH_FROM_PARENT_PREPARATION'
```

先将父端 `grant.json` 安全传入该目录。`service_token` 是此隔离实例的新认证 token；`bailian_token` 是已有百炼凭据。若通过交互终端初始化这两个文件，可用下面的一次性工具；输入不回显，已有文件会被拒绝覆盖：

```bash
python3 - <<'PY'
import getpass
import os
from pathlib import Path
import secrets

directory = Path(os.environ['TRACEMEM_SMOKE_SECRET_DIR'])
service = directory / 'service_token'
bailian = directory / 'bailian_token'
if service.exists() or bailian.exists():
    raise SystemExit('Existing secret files retained; initialization stopped.')
key = getpass.getpass('Authorized Bailian key (hidden): ')
if not 16 <= len(key) <= 4096 or any(not 33 <= ord(c) <= 126 for c in key):
    raise SystemExit('Invalid key format; no files written.')
for path, value in ((service, secrets.token_urlsafe(48)), (bailian, key)):
    with path.open('x', encoding='ascii') as output:
        output.write(value + '\n')
    path.chmod(0o644)
PY
chmod 0644 "$TRACEMEM_SMOKE_SECRET_DIR/grant.json"
docker compose -f compose.model-smoke.yaml -p tracemem-model-smoke config --quiet
docker compose -f compose.model-smoke.yaml -p tracemem-model-smoke build --pull
docker compose -f compose.model-smoke.yaml -p tracemem-model-smoke up -d --wait --wait-timeout 90
curl --fail http://127.0.0.1:8766/health
```

文件使用 0644 是为了使 UID 10001 的容器通过单文件 secret bind mount 读取；宿主私有父目录仍为 0700。这不等于把密钥目录开放给其他用户。不要将模型 key 本身放进环境变量。

此独立 Compose 文件只绑定宿主 `127.0.0.1:8766`，限制 1 CPU、512 MB 内存；一个 Gunicorn worker、四线程，transport 通过持久 guard 串行准入。原基础服务及其 8765 端口不变。健康接口应为 `mode=live_prepaid_embedding_smoke`、`dense=text-embedding-v4`、`reranker=disabled`、`llm_enabled=false`，但这些配置标识仍不能代替真实收据。

## seed、重建与 verify

仅对这个全新隔离实例依次执行一次；任何步骤失败即停止检查原始状态，不自动重试：

```bash
docker compose -f compose.model-smoke.yaml -p tracemem-model-smoke exec -T api \
  python deploy/embedding_smoke_test.py seed
docker compose -f compose.model-smoke.yaml -p tracemem-model-smoke up \
  -d --no-build --force-recreate --wait --wait-timeout 90
docker compose -f compose.model-smoke.yaml -p tracemem-model-smoke exec -T api \
  python deploy/embedding_smoke_test.py verify
```

seed 预期使用前两个 slot；verify 的重复 Add 应直接返回既存结果，只为第二种查询使用最后一个 slot。两个阶段都核对目标排名第一、两条原文完整、不同用户没有结果。合成资料不能用来宣称 benchmark 得分或线上容量。

## 私有收据审计、停机与父端核对

审计命令不调用 provider。它核对已持久化的 grant、三个完整收据、256 维向量、usage 和业务数据库，并输出脱敏摘要；完整私有审计 JSON 留在卷中，不进入 Git 或 CI artifact：

```bash
docker compose -f compose.model-smoke.yaml -p tracemem-model-smoke exec -T api \
  python deploy/audit_embedding_smoke.py \
  --grant /run/secrets/embedding_grant \
  --expected-grant-sha256 "$TRACEMEM_GRANT_SHA256" \
  --state /app/runtime/deployment-embedding-smoke \
  --private-output /app/runtime/deployment-embedding-smoke/private-audit.json
docker compose -f compose.model-smoke.yaml -p tracemem-model-smoke stop
docker compose -f compose.model-smoke.yaml -p tracemem-model-smoke cp \
  api:/app/runtime/deployment-embedding-smoke/private-audit.json \
  "$TRACEMEM_SMOKE_SECRET_DIR/private-audit.json"
chmod 0600 "$TRACEMEM_SMOKE_SECRET_DIR/private-audit.json"
rm -- "$TRACEMEM_SMOKE_SECRET_DIR/bailian_token"
```

保留已停止的实例及 `smoke-runtime` 卷，不执行 `down -v`。删除的只是此次上传的宿主百炼 key 文件；父端凭据不受影响。将 `private-audit.json` 经安全渠道取回父端私有目录。若审计失败，先停机并保留现有 journal、grant 和预留，不能清空状态重新调用。

在父端代码根目录执行，`PRIVATE_AUDIT_JSON` 指向刚取回的原始私有审计文件。先只读核对，再显式写入父端已有账本：

```bash
export PRIVATE_AUDIT_JSON='/absolute/private/path/private-audit.json'
python3 scripts/reconcile_embedding_smoke_grant.py \
  --ledger "$PARENT_LEDGER" --grant "$GRANT_DIR/grant.json" \
  --receipts "$PRIVATE_AUDIT_JSON" --output "$GRANT_DIR/reconciliation"
python3 scripts/reconcile_embedding_smoke_grant.py \
  --ledger "$PARENT_LEDGER" --grant "$GRANT_DIR/grant.json" \
  --receipts "$PRIVATE_AUDIT_JSON" --output "$GRANT_DIR/reconciliation" --apply
```

核对工具只接受三个全部 HTTP 200、模型／向量／usage 合同通过且对应原 attempt ID 的收据；失败或未知结果不改变账本，预留继续保留。它不会新增预留或发请求。若应用核对中断，只能对相同 grant、收据及输出目录继续核对原 ID；已核对项不得再次计费或派发。

## CI 与证据边界

Docker CI 运行 `test_embedding_smoke*.py` 和 `test_granted_embedding_smoke.py`，使用临时合成账本、fake HTTP 和合成 token；现有基础容器的真实 Docker 生命周期验证继续保留。CI 不准备真实 grant、不注入 provider key、不启动 live smoke。只有服务器实际收据审计和父端核对完成后，才能记录“三次真实 embedding 调用成功”。
