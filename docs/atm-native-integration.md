# ATM-Bench 原生接入状态

读取日期：2026-10-25 UTC。数据固定 revision
`78e826dc07e97466b2f54443831ef9a83ab8b27c`，作者代码固定
`ef4e5dff1a47ec71213a06e359f02753defa8fb1`。

官方[数据卡](https://huggingface.co/datasets/Jingbiao/ATM-Bench/blob/78e826dc07e97466b2f54443831ef9a83ab8b27c/README.md)
将数据标为 **CC BY-NC 4.0**；作者代码是 MIT。两者不能混写成 MIT 数据许可。
本轮仅取公开元数据、文档和六个小型原生 QA 文件，未下载个人图像、视频、邮件或 GPS 内容，
未向模型发送这些数据。原生数据不进入 GitHub，也不重新分发。

元数据清单含 3759 张图、533 段视频、一个邮件 bundle、4292 个地理缓存文件，
另有作者预生成的 processed_memory。数据卡声明邮件 6742 条；当前未读取邮件内容
逐项核实。完整九页元数据现已核完，共 8597 文件、3,329,285,464 字节，
其中原图 1,276,954,223 字节、视频 2,013,856,006 字节、邮件 bundle 5,037,231 字节。
这些是远端清单大小，未下载原始内容；首个 tree 页面不是全库。
作者 processed_memory 的 Qwen SGM 不属于指定模型从 raw 构建的严格条件。

原生 QA loader 与隔离准备流程已实现。公共问题只包含 id/question/qtype，
answer/evidence_ids/NIAH 固定 pool 留在 scorer-only，不能作为记忆内容。
regular 有 1013 题（514 open_end、360 number、139 list_recall），Hard 有 31 题。
实际检查 regular 与 Hard 的 ID 和规范化问题文本均无精确交集，
但两者有 59 个相同 gold evidence ID，共用原始 corpus，不能据此宣称完全独立泛化。
四份 NIAH 分别为 25/50/100/200 evidence pool，均使用同一组 31 道 Hard 题，
它们不是四份独立测试集。检索分母按来源证据计算，不按生成卡片数计算。

当前接入是 **schema/清单/数据隔离完成，ATM 实际检索与答题未运行**。
新增本地视频帧适配代码，支持来源 SHA 校验、受限本地路径、固定均匀中点采样、
单进程超时/字节/尺寸上限，以及 frame occurrence 到原始 video evidence ID 的映射。
多帧仍只算一个 native evidence，不用帧数虚增召回。存在音轨时当前严格路径直接拒绝，
不会静默丢音轨后称完整支持。只通过注入 fake media process 的单元验证。
本机未发现 ffmpeg/ffprobe，因此还没有真实视频解码验证；未安装新二进制。
调用参数依据 FFmpeg 官方文档，不执行上游研究仓库代码。

原始媒体内容与外部 API 处理边界审核仍待完成。以暂定每视频六帧、三帧一批、
每批0.25元固定预占计算，3759图+3198帧需要2319批，仅视觉预占579.75元，
超过当前剩余预算，尚不含embedding/reader。这是保守预占方案，**不是实际账单估价**。
不能在当前方案下直接全量处理，也不能省略视频却声称完整ATM；应先冻结可承受的
采样和成本方案或取得有依据的更紧输入费用上界，不按gold挑媒体。
后续固定与 Mem-Gallery 相同的 A/B/C 配置，不根据公开答案改提示词或训练。

来源还包括作者的[原生格式文档](https://github.com/JingbiaoMei/ATM-Bench/blob/ef4e5dff1a47ec71213a06e359f02753defa8fb1/docs/data.md)。
没有复制或执行上游项目代码。
本地帧处理参数参考 [ffprobe](https://ffmpeg.org/ffprobe.html) 和
[ffmpeg](https://ffmpeg.org/ffmpeg.html) 官方文档，读取日期同上。
