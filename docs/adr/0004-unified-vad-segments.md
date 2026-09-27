# ADR-0004：统一 VAD 分段与切片制品

状态：已接受

## 决定

共享底座使用统一毫秒时间轴表示语音活动区间。每个区间同时产生：

- 确定性 `AudioSegment.id`；
- 指向原始音频的 `asset_id`；
- 指向独立 WAV 制品的 `slice_asset_id`；
- PostgreSQL `audio_segments` 记录；
- S3/MinIO 切片制品；
- VAD ToolRun 和报告指标。

首个后端为 `adaptive_energy_v1`：FFmpeg 解码到 16 kHz 单声道 PCM，按帧计算 dBFS，利用噪声底和动态范围自适应生成阈值，再执行短静音合并、最小语音时长和边界 padding。

## 原因

- 先稳定 ASR、降噪、分类和 TTS 参考音频共用的分段合约；
- 无需新增模型依赖即可建立可测试基线；
- 确定性 ID 与 S3 key 使 worker 重放保持幂等。

## 限制与后续

能量 VAD 对稳态背景噪声、音乐和远场语音的区分能力有限。后续将在相同 `VadResult` 边界下增加 Silero/WebRTC 后端，并用实际客服、会议和播客数据评估语音召回率、误检率和边界偏差。
