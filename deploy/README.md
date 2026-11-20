# 部署文件

| 用途 | 文件与说明 |
| --- | --- |
| 本地容器演示 | 从仓库根目录使用 [compose.yaml](../compose.yaml) 和 [Dockerfile](../Dockerfile)。启动、健康检查和备份步骤见[本地容器部署](../docs/operations/docker-deployment.md)。 |
| 构建原生 Mem0 镜像 | 在仓库根目录执行 `docker build -f deploy/docker/Dockerfile.mem0 -t tracemem-native .`。[docker/](docker/) 包含镜像定义和按哈希固定的依赖。 |
| 配置当前专用服务 | [v4.3.15-rc7 运行说明](../docs/service/gpu6-full-migration.md) 列出了入口、已有状态目录和运行条件。 |
| 模型接入检查 | [embedding Smoke 指南](../docs/operations/embedding-model-smoke.md) 使用 [Compose 文件](compose/compose.model-smoke.yaml)。 |
| 复查历史服务 | [compose/](compose/) 中的 service、bailian 和 context 文件是独立部署模板。使用时将模板置于对应部署目录，与其引用的 `source/`、`config/` 等目录并列。 |
