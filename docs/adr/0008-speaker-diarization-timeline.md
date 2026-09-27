# ADR 0008：说话人分离与 speaker-attributed transcript

- 状态：已采纳
- 日期：2026-09-24

## 背景

会议、客服和播客分析不仅需要文本，还需要回答“谁在什么时候说了什么”。仅在 ASR 之后按文本猜测 speaker 会丢失重叠说话和原始边界，也难以更换模型或重新对齐。

## 决策

引入独立 `DiarizationBackend` 与 `DiarizationDocument`。模型直接处理原始音频并输出全局毫秒 `SpeakerTurn`；首个实现适配 Pyannote 4.x Community-1，并在可用时优先使用 exclusive diarization。

完整 turn 时间轴作为独立 S3 制品和 PostgreSQL 记录保存。VAD segment 根据各 speaker 的重叠时长选择主 speaker，ASR transcript span 继承该 speaker。分离运行记录模型、版本、人数约束、turn 数量与片段归属数量。

## 后果

- 原始说话人时间轴与下游简化归属同时保留，可重新对齐而无需重新推理。
- `diarize` 是显式作业操作，未配置模型时请求会明确失败，而不是静默跳过。
- Pyannote 为重量级可选依赖，默认镜像不安装；模型授权和 token 由部署环境管理。
- 单一 segment 的主 speaker 归属不表达重叠说话；使用方需要重叠信息时应读取 `DiarizationDocument.turns`。
- 后续需要引入 DER/JER 评测与跨录音 speaker linking。
