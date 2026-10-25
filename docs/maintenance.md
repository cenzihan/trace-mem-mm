# 本地迁移、恢复与计数

当前记忆库 schema 3 新增模型 Add 操作 journal；版本 1/2 可在服务初始化时非破坏升级。模型请求的 preparing/uncertain 状态不自动过期，重启、备份恢复或关闭模型开关均不能绕过；repair-index 遇到所选用户的未决状态会拒绝。来源、索引与 complete 同事务提交。purge 清理用户 journal，旧在途 owner 随即失去提交权限。未知计费状态需先核账，当前没有自动解锁入口。旧版本进程不应再打开 schema 3 数据库；恢复旧代码时使用其配套快照数据库，勿覆盖活跃数据。

切换 dense 模式前需完成该用户全部指定模型向量迁移。已有卡片但缺向量时 Add 不返回假成功，也不会自动付费补建；显式迁移工具尚待真实 provider 联调。以下 repair-index 仅能修复来源派生的确定性索引，不能解除这些门槛。最新测试与证据见 [失败注入审查](../reports/milestone-failure-review-20261025.md)。

操作对象必须是本项目的数据库。先停止写入并备份；不要用恢复覆盖活跃数据库。CLI 的 backup 使用 SQLite online backup，目标必须不存在，原始来源不变。

```powershell
python -m agent_memory --db runtime/system.sqlite integrity
python -m agent_memory --db runtime/system.sqlite backup runtime/snapshot-new.sqlite
python -m agent_memory --db runtime/system.sqlite repair-index
python -m agent_memory --db runtime/system.sqlite repair-index --user-id synthetic:user:alice
python -m agent_memory --db runtime/snapshot-new.sqlite backup runtime/restored-new.sqlite
```

版本 1 是来源层，版本 2 增加卡片、显式事实和链接。缺派生索引时 Search 明确 503，可重放原 Add 或使用 repair-index；恢复只从已有来源派生，不调用 LLM/embedding，不制造缺失向量。真实 dense 开启时仍要求完整的 text-embedding-v4 索引，不能用本命令冒充完成。高于已知版本的数据库拒绝自动降级。integrity 检查 SQLite、外键及卡片来源作用域，不会修复损坏的原始图像。

backup 创建失败时保留目标供诊断，不自动覆盖重试。恢复文件检查通过后，手动让 CLI 指向新路径验证 Add/Search，再决定切换；不要把旧 DB 覆盖掉。故障机器、磁盘损坏、系统级快照/安全擦除不在本实现的保证范围。

实际预算唯一账本为 `runtime/development-budget.sqlite`（CNY 500）；Smoke 唯一账本为 `runtime/evaluation-ledger.sqlite`（全任务 5 次）。每次实际尝试之前持久化预占，结果不明保持占用；人工直接调用外部 API 不受代码约束。不能复制旧账本恢复额度，不能用测试临时账本替代实际账本。EvaluationLedger 没有网络提交功能，仅提供权限和计数门槛；当前没有 AML Key/公网端点，Full 始终拒绝。

`python scripts/run_checks.py` 包含真实临时 loopback HTTP 测试，finally 关闭；不绑定公网。`scripts/operational_checks.py` 可单独运行 6 个运维测试。测试 token 和消息全部原创合成，不取系统凭据。服务开发启动器关闭访问日志，测试确认正文/查询 URL/token 不出现；代理层与正式托管日志尚未配置，不能把本地检查视为生产全链路日志证明。

服务运行和契约测试使用标准库；原创图形的反证生成脚本另需 Pillow，本机已经安装 13.0.0，`pyproject.toml` 的 `audit` 可选依赖记录了这个实际验证版本。本轮未为本机额外安装包。
