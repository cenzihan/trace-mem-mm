# ATM N0：完整 SGM 来源库的本地 BM25

读取/运行日期：2026-10-25 UTC。本次 **0 provider 请求、0 新增 API 费用**。
没有 reader 答题、dense embedding、reranker、GPU 或原媒体解码。
它是作者 Qwen SGM 外部预处理研究条件，不是严格 gpt-4o-mini 从 raw 端到端条件，
也不是官方比赛分数。

## 来源与完整性

仅从官方 Hugging Face 固定 revision
`78e826dc07e97466b2f54443831ef9a83ab8b27c` 下载 image/video SGM 和邮件文本三个文件，
共 34,311,246 字节。分别通过官方 LFS SHA256 或 Git blob SHA1 与大小校验，
本地另记 SHA256。没有镜像、登录或新凭据，没有下载本阶段不需要的原图/视频。

[数据卡](https://huggingface.co/datasets/Jingbiao/ATM-Bench/blob/78e826dc07e97466b2f54443831ef9a83ab8b27c/README.md)
标为 CC BY-NC 4.0，代码 MIT 不是媒体许可。
原生文件只留在本项目授权目录，不推 GitHub、不打包分发、不发送模型。
数据卡说明 owner consent 和部分脱敏，但保留时间/地点等，仍不能当作已完成向服务商传输的隐私审查。

| 来源 | 实际记录 | 唯一 ID | 缺失/额外媒体 ID | 仅元数据、无描述 |
|---|---:|---:|---:|---:|
| 图像 SGM | 3,759 | 3,759 | 0/0 | 0 |
| 视频 SGM | 533 | 533 | 0/0 | 0 |
| 邮件 | 6,742 | 6,742 | 不适用 | 0 |
| 合计 | 11,034 | 11,034 | 无缺失 | 0 |

图/视频 ID 与完整 raw 媒体清单逐项对应。无重复 ID、无无对象行、无混入的 QA/gold 字段。
regular 的 1,325 个不同 gold source ID 均能找到；另 9,709 个未被这套 gold 标注的来源仍保留，
并非只索引相关记录。未被 gold 标注不等于语义上必然无关。

## 固定索引和隔离评分

在读取问题之前冻结全部来源和参数：BM25 k1=1.2、b=0.75，项目现有 tokenizer，
查询词去重并排序，无停止词调优、无 stemming，平分按原生 ID 字典序。
媒体字段为 ID/type/time/location/short caption/caption/OCR/tags，邮件为 ID/time/summary/detail。
与 ATM 选定文本字段兼容，不含额外 entities 字段；本地文本完全保留，不做 8192 字节截断。
预计算 postings 与原有 BM25 公式通过合成契约验证，未调整检索参数追分。

完整库的 source-index.sqlite 只含来源记录，没有问题/答案/gold 列。
检索进程只读取 public questions，输出 native source ID 排名；全部 1,013 题排名保存后，
另一个只读进程才读取原始 gold。评分不回写索引，无 gold ID 归一化或针对样本的 remap。

## 检索结果

所有 1,013 题都有非空 gold，无重复 gold label，无未解析 gold ID。
以原生 artifact ID 计，不把 view、card 或 video frame 当作新的证据单位。
令 G 为去重 gold ID 集，Rk 为去重 TopK：

- Fractional Recall：各题 `|Rk ∩ G| / |G|` 的宏平均。
- Any-hit：各题 `1[Rk ∩ G 非空]` 的宏平均。
- All-evidence：各题 `1[G ⊆ Rk]` 的宏平均。

| K | Fractional Recall | Any-hit | All-evidence |
|---|---:|---:|---:|
| 1 | 0.4128 | 0.4630 | 0.3791 |
| 5 | 0.6142 | 0.6545 | 0.5785 |
| 10 | 0.6886 | 0.7246 | 0.6525 |
| 20 | 0.7383 | 0.7730 | 0.6989 |
| 50 | 0.7932 | 0.8253 | 0.7581 |
| 100 | 0.8299 | 0.8598 | 0.7947 |

| 原生类别 | 题数 | Fractional@10 | Fractional@20 | All-evidence@20 |
|---|---:|---:|---:|---:|
| open_end | 514 | 0.6973 | 0.7570 | 0.7160 |
| number | 360 | 0.7458 | 0.7840 | 0.7528 |
| list_recall | 139 | 0.5082 | 0.5510 | 0.4964 |

| Gold 所属模态 | 题数 | Fractional@10 | Fractional@20 | All-evidence@20 |
|---|---:|---:|---:|---:|
| email | 381 | 0.8863 | 0.9108 | 0.9081 |
| image | 582 | 0.5676 | 0.6350 | 0.5687 |
| video | 50 | 0.5900 | 0.6267 | 0.6200 |

该 regular release 的标注没有混合模态 gold 集，不能据此证明跨模态联合推理。
全部查询返回 100 个正分来源，无空结果。建库和检索共约 10.83 秒；
查询中位约 15ms、P95 约 16ms，Windows 计时有量化，本地单次观察不是 SLA。

## 边界与后续成本依据

完整 SGM 库不等于完整原媒体理解：原图、视频帧、音轨没有处理，视频仅用作者 SGM。
没有 native answer score 或官方 judge，不能把 Recall 当答题正确率，
也不能拿它直接宣称超过发布版 MiniLM/LLM 的 ATM-RAG。

选定文本共 12,999,278 UTF-8 字节。只有一条图像描述超过现有 embedding 8192 字节准入，
无损按 UTF-8 分块后共 11,035 块，不是预估的三倍 33,102 份向量输入。
这只是规划，当前 **ATM 文本向量缓存为 0**；本地词项计数不是供应商计费 token。
经济计划见 `atm-after-n0-economical-plan.md`，不自动发起下一次调用。

运行记录：reports/atm-n0-download.json、atm-n0-schema.json、atm-n0-run.json、atm-n0-scores.json。
原问题、gold、SGM、邮件及逐题结果均只在本地隔离目录，不随说明公开。
