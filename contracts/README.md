# Contracts

这里保存跨服务稳定的数据契约：

- `AudioAsset`：原始或派生音频与探测结果；
- `AudioSegment`：统一毫秒时间轴、原音频/切片制品关联、说话人、转写和标签；
- `AudioQualityMetrics`：LUFS、峰值、削波与静音指标；
- `ToolRun`：工具、版本、参数、输入输出和指标；
- `AudioJob`：异步作业与状态事件；
- `AudioReport`：资产、片段、运行记录与发现项的可追溯聚合。
- `TranscriptDocument`：ASR 后端、模型、语种、全文与全局毫秒时间戳。
- `AsrBenchmarkReport`：ASR 样本级 WER/CER、RTF、语种、时间戳和资源指标报告。
- `VadBenchmarkReport`：VAD 候选参数、时长级检测指标、边界误差和最佳候选报告。
- `SpeakerTurn`：全局毫秒时间轴上的说话人区间。
- `DiarizationDocument`：说话人分离后端、模型、speaker 数量和完整 turn 时间轴。
- `DiarizationBenchmarkReport`：DER/JER、人数准确率、边界误差和运行性能报告。
- `DiarizationBenchmarkMatrixReport`：regular/exclusive 与人数约束矩阵、失败分析及生产门槛建议。

JSON Schema 由 Pydantic 模型生成并纳入版本控制，服务变更时必须检查兼容性。
