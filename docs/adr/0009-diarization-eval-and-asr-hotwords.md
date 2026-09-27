# ADR 0009：DER/JER 工程评测与逐作业 ASR 热词

- 状态：已采纳
- 日期：2026-09-24

## 背景

说话人分离接入后，需要可重复度量 speaker 数量、漏检、误检和混淆，而不能只看单条时间轴。另一方面，客服产品名和领域术语是 ASR 的主要错误来源，热词必须成为可审计的逐作业参数，而不是修改共享模型的隐式状态。

## 决策

新增 `diarization_eval`，用固定毫秒帧构建 speaker 活跃集合，先执行最优一对一 speaker 映射，再计算 false alarm、missed detection、confusion、DER 和 JER。报告固化帧长、collar、重叠处理和 exclusive 设置。

新增严格 `AsrRequestOptions`。热词在请求级清洗、去重和限长后，每次调用都显式传给后端：Faster-Whisper 使用 `hotwords`，FunASR 使用 `hotword`。ASR 基准清单也支持逐样本热词。

## 后果

- speaker 标签随机编号不影响 DER/JER，失败样本不会中断整份报告。
- 工程 DER/JER 使用离散帧近似；外部基准发布需用官方评测工具复核。
- 热词不会污染后续请求，但会完整进入 ToolRun，调用方应避免把秘密作为热词。
- FunASR checkpoint 对热词支持不一致，必须通过同数据集 A/B 评测确认收益。
