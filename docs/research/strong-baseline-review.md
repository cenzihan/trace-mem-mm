# 强基线静态核查与适配边界

读取日期：2026-10-25 UTC。仅缓存并静态阅读官方源码/配置/许可，未导入执行，
未安装依赖、下载模型、运行 GPU 或发起 provider 请求。
逐文件来源、大小和 SHA256 保存在`reports/baseline-source-review.json`。

## 原版与指定模型适配版

| 条件 | 已核实的原版 | 指定模型适配需要改变什么 | 当前状态 |
|---|---|---|---|
| V-Mem 原版 | Session/Round/Unit，细粒度检索后返回 round；MiniLM 文本、SigLIP2 视觉通道，另有 query anchor | 文本改 text-embedding-v4；LLM 固定 gpt-4o-mini；不能保留 SigLIP 视觉 embedding | 未运行，不称学术榜合规原版 |
| V-Mem-style 适配 | 借 unit→原生 parent 聚合、RRF、模态路由和 transient anchor | 文本桥代替 V→V 像素 embedding，属于方法适配，不是等价复现；ATM 本身无对话 round，不造 session | source view/父证据聚合契约已写，encoder/anchor/reader 未联调 |
| ATM-RAG 发布主表 | run.sh 默认 MiniLM、Qwen3-VL-8B-Instruct-FP8 answerer，Qwen3-VL-2B 生成 SGM | embedding/answerer 换指定模型，统一证据预算；发布原分不能直接横比 | 未运行上游原版 |
| ATM-RAG 通用配置 | config.py 默认 Qwen3-VL embedding/reranker，且有多种 retriever 可选 | 不应把通用默认配置误当发布主表运行参数 | 仅静态核查 |
| ATM-SGM 指定模型研究条件 | 复用作者预计算 SGM，再用 text-embedding-v4/gpt-4o-mini | 必须披露 Qwen 生成来源，不能叫全程指定模型从 raw 构建 | 解析/来源/预算契约已写，未下载 SGM payload 或跑新分 |
| strict-from-raw | 原始媒体经 gpt-4o-mini 表示、text-embedding-v4 索引 | 保留原字节，视频/音轨不可静默遗漏；当前全库 6 帧预占不适配余预算 | 不启动 |

V-Mem 固定 commit`36916b14dc5241e04acbf5cd0f3c40799bc09550`。
其[数据模型](https://github.com/Dingyi-Kang/V-Mem/blob/36916b14dc5241e04acbf5cd0f3c40799bc09550/model.py)
与[检索实现](https://github.com/Dingyi-Kang/V-Mem/blob/36916b14dc5241e04acbf5cd0f3c40799bc09550/retrieve.py)
支持“search fine, return coarse”；不是知识图谱方法。
[文本编码](https://github.com/Dingyi-Kang/V-Mem/blob/36916b14dc5241e04acbf5cd0f3c40799bc09550/embeddings.py)
默认 all-MiniLM-L6-v2（384 维），
[视觉编码](https://github.com/Dingyi-Kang/V-Mem/blob/36916b14dc5241e04acbf5cd0f3c40799bc09550/visual_embeddings.py)
默认 SigLIP2（768 维）。这些 embedding 不能原样带入指定模型条件。
[HyDE/anchor](https://github.com/Dingyi-Kang/V-Mem/blob/36916b14dc5241e04acbf5cd0f3c40799bc09550/hyde.py)
仅作查询假设；适配版不得把假设写成记忆事实。
其[Mem-Gallery loader](https://github.com/Dingyi-Kang/V-Mem/blob/36916b14dc5241e04acbf5cd0f3c40799bc09550/load_memgallery.py)
只取首个 caption/image；本项目不能据此丢弃同一来源的其余图片。

ATM 固定 commit`ef4e5dff1a47ec71213a06e359f02753defa8fb1`。
发布主表的[run.sh](https://github.com/JingbiaoMei/ATM-Bench/blob/ef4e5dff1a47ec71213a06e359f02753defa8fb1/scripts/QA_Agent/MMRAG/run.sh)
与[通用 config.py](https://github.com/JingbiaoMei/ATM-Bench/blob/ef4e5dff1a47ec71213a06e359f02753defa8fb1/memqa/qa_agent_baselines/MMRag/config.py)
有不同默认模型；原版引用必须明确 entrypoint。
[发布 README](https://github.com/JingbiaoMei/ATM-Bench/blob/ef4e5dff1a47ec71213a06e359f02753defa8fb1/README.md)
声明 SGM processor/captioner 为 Qwen3-VL-2B，Hard 使用 release set，不能混用 preprint 表。
当前 HF processed_memory/readme.md 为空，不能把它当逐条生成元数据验证；
SGM 来源目前按作者声明披露，payload 尚未检查。

[原文 formatter](https://github.com/JingbiaoMei/ATM-Bench/blob/ef4e5dff1a47ec71213a06e359f02753defa8fb1/memqa/retrieve/utils.py)
使用 caption/OCR/tags/time/location 等，不含 entities 字段。
加入 entities 属于额外消融，不能假称原版默认。
[reader](https://github.com/JingbiaoMei/ATM-Bench/blob/ef4e5dff1a47ec71213a06e359f02753defa8fb1/memqa/qa_agent_baselines/MMRag/mmrag_utils.py)
可提供文本、原图和视频帧，但原版有独立帧预算，受控比较须显式统一。
其 compute_recall 实现是命中占 gold 数量的 fractional 平均，不能把同名文档的 any-hit 混算。

## 许可、依赖与安全边界

两份代码均为 MIT：[V-Mem 许可](https://github.com/Dingyi-Kang/V-Mem/blob/36916b14dc5241e04acbf5cd0f3c40799bc09550/LICENSE)、
[ATM 许可](https://github.com/JingbiaoMei/ATM-Bench/blob/ef4e5dff1a47ec71213a06e359f02753defa8fb1/LICENSE)。
ATM 数据独立标为[CC BY-NC 4.0](https://huggingface.co/datasets/Jingbiao/ATM-Bench/blob/78e826dc07e97466b2f54443831ef9a83ab8b27c/README.md)，不能外推成 MIT 媒体许可。
数据卡说明个人数据经 owner consent 和部分脱敏，仍保留时间/GPS；
公开可下载不等于已完成每条数据向服务商传输的隐私核查。

V-Mem 依赖 numpy/openai/sentence-transformers，视觉服务另需 vLLM；
ATM 依赖 Torch/Transformers/OpenCV/decord 等。当前只静态读 requirements，不整套安装。
ATM 通用 config 导入会读凭据/建目录，索引 utility 支持 pickle；本轮都不执行，
使用自有 JSON/SQLite 边界，不读取上游 pickle。
没有复制上游框架代码；新模块是 source-only 契约准备，不是完整 baseline 复现。
