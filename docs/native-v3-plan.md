# 第二场景独立复核：冻结计划

采用 Mem-Gallery 固定数据版本 `af912daba984e896e253016b7c7e334ef92c2a6f`。
在读取新场景前记录排序规则：对原生 dialogue 文件名排序，排除已经评估的
AI_Robotics 场景，取首个未读文件 Academic_Animal_Pet_Research_Life。
完整索引 177 轮、354 条消息、14 个 session、45 张媒体；不依据 gold 筛选 distractors。
全部 77 道问题做检索评估。此前场景的实现、权重、视觉提示词和结果保持原样。

A 为原文和全局图像描述的 BM25+dense/RRF；B 并联有来源的 OCR/区域/关系卡片；
C 在 B 上加有限来源邻居。三者沿用 `1c156783` 的核心代码及视觉提示词。
Embedding 固定 text-embedding-v4、256 维，视觉、reader、代理 judge 固定
gpt-4o-mini；Qwen 关闭。卡片不替代原文。全部原图有 occurrence、原始哈希、
协议文件哈希和模型缩放文件哈希映射。

固定 Top20。reader 只比较 A/C，各 10 道问题：按原始题序取每个 point 类别的
首题，再按题序补足，覆盖九类，不读取答案选择题目。每题两组共用相同最多两图
实际数量、8192 文本 token、24000 UTF-8 字节和 256 输出 token 上限。
按整条证据前缀装包，不用 gold 补图。图片来自经过哈希校验的本地媒体，
模型输入由原图缩放至最长边 1024，detail=high。

45 张原始媒体全部保留，其中一张原文件超过 10 MiB；为满足单图协议上限，
按预先登记策略生成同尺寸 JPEG95。44 张接口文件原字节不变，一张经过有损编码。
这项转换不隐藏，不能称全部原字节回传。原始文件及转换不进入 GitHub。

memory-source 不包含 QA/profile；reader 只读取问题与已检索上下文。
答案全部落盘后，另一个评分进程才读取隔离 gold。检索分别报告
fractional recall、any-hit 和 all-evidence coverage；reader 报告支持证据、
引用有效性、拒答、独立实现的 EM/F1 和 gpt-4o-mini 代理判分。
代理 judge 隐藏 A/C 标签，与赛事官方 judge 不同；检索召回不作为答题准确率。
不看结果改提示词或重跑追分。

计划 83 次提供商请求，累计预占上界 11.95 元；本轮硬上限 100 次/20 元，
共享总预算 500 元。未知请求结果保留预占并停止，不自动重试。
已完成的两家连通测试不重做；不运行官方 Smoke/Full。

ATM-Bench regular 是紧接本轮的独立数据集接入工作。先核对官方固定版本、
原生 schema、完整 corpus、媒体许可和 regular/Hard/NIAH 关系，再做隔离 loader。
作者 Qwen 生成的 SGM 与指定 gpt-4o-mini 从 raw 重建的条件必须分开。
音频/视频不能静默丢弃；目前尚未运行 ATM 评估，不宣称已有跨数据集结果。
