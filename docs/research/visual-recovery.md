# 八图响应持久化与恢复

2026-10-25，本轮只验证本地可靠性。没有读取真实凭据、发送 provider 请求、创建真实 guard、GPU、push 或部署。
冻结 N0/Mem-Gallery 实验不改；本改动仅用于新的显式八图路径。

`agent_memory/visual_recovery.py` 增加私有 SQLite 操作 journal 和不可变图像 blob：

1. 在本地保存原图、请求派生图及有序 occurrence manifest，分别校验 SHA256。
2. 原子领取操作后才允许一次传输。预算传输层预占并标记 attempt，再把 attempt ID 绑定操作。
3. HTTP 返回后，先以 SQLite `synchronous=FULL` 保存状态码和响应正文，再交给 provider/八图解析器。
4. 成功可重启复用；有完整响应但解析失败，可离线重解析。未知/inflight 状态只保留，不自动重发。

指纹覆盖 namespace、模型/prompt/版本、图像顺序及原图/请求图 hash。
相同图片在不同 occurrence 保留来源；不同 namespace 不复用缓存。
当前只支持同一固定请求版本的恢复；prompt/版本发生变化会使指纹不匹配，不能暗中跨版本复用。

## 故障处理

| 故障 | 持久状态与恢复行为 |
|---|---|
| HTTP 200 后本地解析器错误 | 先保存响应；修复解析器后本地重解析，不新增请求 |
| 返回内容缺字段/截断/usage 越界 | 保存已知响应、保持无效状态，不放宽事实校验，不转为成功缓存 |
| 响应已落盘后进程终止 | 重启读取 receipt；测试用子进程 `os._exit(44)` 验证 |
| 网络超时或响应保存失败 | 结果未知，保留预算预占，不自动重放 |
| 成功操作再次执行 | 校验 blob/响应完整性并复用，不建立 transport 或再次预占 |
| 并发执行同一操作 | SQLite 原子 claim 仅允许一方发送 |
| blob、manifest 或响应被改动 | hash/指纹失败，阻止恢复与发送 |

若进程在收到响应与成功落盘之间中断，仍可能无法恢复正文；这种窗口按未知处理，不能保证端到端 exactly-once。
恢复不会自动释放历史预占、补写实际账单或把历史 run guard 改成完成。
发生“有 receipt、账本观察尚未写入”的崩溃时，可取得结果，但该历史 attempt 仍须明确核对后才能推进旧运行。
禁止没有预算 attempt 绑定的传输，也禁止 factory 开启自动重试；live factory 还必须使用现有 `ScopedTransport` 运行范围。
本轮没有 live factory 或新运行器实例。

## 存储与秘密边界

生产数据目录必须位于项目 `runtime` 之下，例如 `runtime/visual-private/`；测试仅使用隔离临时目录。
SQLite 中可含数据集响应，只能本地保存。`runtime/` 被 Git 忽略，既有源码 allowlist 导出不复制该目录。
没有执行导出或 Git 操作。

不持久化请求 header、credential reader 返回值或异常原文。
响应保存前会去除 header/key 字段，并掩去当前请求凭据的明文/JSON Unicode 转义回显；日志只给类别。
如果发生掩去操作，会记录标志，保存的是脱敏响应而非声称字节完全相同的原始响应。
本轮测试覆盖明文和转义回显未进入文件；这不是对任意恶意编码的秘密探测保证。
文件不额外声称静态加密，依赖本机目录访问控制；不暴露下载地址，不上传。

原始大图与请求图分开存储：新的 `VisualInput.original_bytes` 接受最多 64 MiB 的本地 PNG/JPEG/WebP 原字节，
这个数是工程存储上限，不是比赛限制。请求派生图仍按单图 10 MiB、整批 30 MiB、最长边 512 的准入检查。
超过 10 MiB 的原图测试验证了原字节回读完全一致，发送的只是独立小图；没有通过重编码覆盖原图。
blob 先写临时文件、flush/fsync 后原子替换，再写 journal 引用；完整性失败一律阻止继续。
这不等同已经验证断电、磁盘硬件故障或真实媒体全覆盖。

## 本轮验证与剩余前提

新增 15 项恢复/故障注入测试，全量回归快照 268 项通过。
初次沙箱拒绝临时目录原子重命名，经正常审批后执行通过；没有修改防火墙或安全设置。
复现命令：`python -m unittest discover -s tests -p test_visual_recovery.py -v`；
完整验证：`python scripts/run_checks.py`。
Windows 受限沙箱可能需要正常审批才能执行原子文件操作和旧的 loopback HTTP 测试。

剩余前提：父线程决定 ATM 拟传输内容/服务商留存范围与有界付费运行；
八图实际输出成功率、真实费用及新描述分块数仍待验证，不能从 fake 响应推断。
当前 PATH 没有发现 `ffmpeg`/`ffprobe`，未下载原视频或安装解码器；音轨数量与帧采样遗漏仍未知。
六帧不覆盖音频和完整时间事件，不删除这些候选或 gold 题，也不把此路径称完整音视频支持。
