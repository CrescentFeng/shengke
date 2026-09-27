# ADR-0003：基础音质分析使用 FFmpeg

状态：已接受

## 决定

ingest worker 对显式包含 `loudness` operation 的作业执行三类 FFmpeg 分析：

- `loudnorm` 获取综合响度、响度范围和真峰值；
- `astats` 获取样本峰值、RMS 和削波比例；
- `silencedetect` 获取长静音时长和占比。

指标写入 `ToolRun.metrics` 和 `AudioReport.summary.quality`，超过作业策略阈值时生成可追溯 `Finding`。

## 原因

- 不增加新模型依赖，与现有解码和归一化环境一致；
- 先建立可回归、可配置的质量基线，供「清听·音审通」直接复用；
- 输出是通用数据合约，后续可在不改 API 的情况下替换或叠加模型。

## 限制

`silencedetect` 只识别低能量区间，不是语音 VAD；削波比例是根据接近 0 dBFS 的绝对峰值样本估算。后续需用真实业务音频校准默认阈值。
