# 脚本入口

以下路径以仓库根目录为起点。

| 用途 | 入口 |
| --- | --- |
| 运行本地演示或离线测试 | `python -m agent_memory` 演示合成对话。运行 `python scripts/run_checks.py` 检查回归测试。 |
| 检查 Add/Search 接口 | [离线交付检查](verify_offline_delivery.py) 使用临时数据库，通过本机 HTTP 验证图文写入、重复请求和用户检索。 |
| 验证容器 | [容器检查](verify_container.py) 构建本地演示镜像并验证重启与恢复。运行条件见[容器指南](../docs/operations/docker-deployment.md)。 |
| 追溯 ATM 实验 | 从 [ATM 研究记录](../docs/research/atm/) 查阅对应版本，再查看同版本的 `prepare_atm_*`、`run_atm_*` 和 `score_atm_*` 脚本。 |
| 追溯 Mem-Gallery 实验 | [V7/V7.1 方法与结果](../docs/research/gallery/v7-independent-scene.md) 对应 `*memgallery_v7*` 脚本。 |
