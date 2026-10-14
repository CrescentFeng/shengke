# 声刻 GitHub 仓库归档说明

本仓库保存声刻 1.0.0 工程验证版及其共享音频底座，适合代码审查、版本追踪、测试和后续开发。本次发布采用 Public 可见性，原创内容按根目录 LICENSE 保留权利；第三方材料继续遵循各自许可。公开说明更新于 2026-10-14。

## 已纳入版本控制

- 声刻工作台、展示页、平台 API、worker、四模型适配器与听评工具源码；
- JSON Schema、数据库迁移、示例配置、依赖锁定清单和自动化测试；
- MVP、运行、GPU、验收、演示、网站部署和 v1.0 发布文档；
- 小型公开演示音频及 `artifacts/shengke/release-v1/` 中经审阅的发布证据；
- 共享音频底座原有源码、测试和文档。

## 未上传的大型或本地资产

以下内容由 `.gitignore` 排除，不应直接提交到普通 Git 仓库：

- `models/` 下的模型权重、第三方模型仓库与独立 Python 环境；
- 除 v1.0 发布证据外的 `artifacts/` 运行日志、基准音频和临时输出；
- PostgreSQL、MinIO 数据卷及用户上传的参考音频、生成音频和评分数据库；
- `.env.local`、token、密钥、本机缓存、构建目录和临时文件；
- 未确认可再分发许可的第三方模型或素材。

完整本机文件快照保存在开发机桌面 `声刻/` 文件夹。它用于本机备份和演示，不等同于可公开分发的 GitHub 内容。

## 恢复开发环境

1. 克隆仓库并按 `docs/shengke-mvp.md` 安装平台环境。
2. 根据 `configs/shengke/models.json` 和各锁定清单准备四模型环境与权重。
3. 在 `configs/shengke/local-macos.json` 中填写本机 Python、模型源码和权重路径。
4. 准备 Docker Desktop、FFmpeg、PostgreSQL 与 MinIO，再运行 `scripts/shengke_stack.py --check`。
5. 通过 `scripts/启动声刻.command` 启动；数据库和对象存储中的历史需从单独备份恢复。

仓库中的哈希、版本和测试结果用于追溯当前工程状态，不代替模型许可证、数据许可或多人 MOS 结论。

## 许可与历史版本

请结合 [LICENSE](../LICENSE)、[COPYRIGHT](../COPYRIGHT.md)、[THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md) 和 [素材登记](ASSET_PROVENANCE.md) 阅读。原 `shengke-v1.0.0` 标签保留不变；本次补齐公开说明没有重新运行或改写历史测试、音频、评分和哈希证据。早期 PDF 是规划稿而非当前验收结果。
