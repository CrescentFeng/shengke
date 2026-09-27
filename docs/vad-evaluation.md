# VAD 后端与参数校准

共享底座使用统一 `VadSpec` 和 `VadResult` 支持两个后端：

- `adaptive_energy_v1`：零额外模型依赖，利用音频噪声底、动态范围和最低 dBFS 生成自适应阈值。
- `webrtc_v1`：使用 WebRTC VAD 的 0–3 激进度模式；数值越高越倾向过滤非语音。该实现要求 16-bit 单声道 PCM、8/16/32/48 kHz 采样率，以及 10/20/30 ms 帧，底座会通过 FFmpeg 完成统一解码。

WebRTC 可选依赖：

```bash
pip install -e ".[vad-webrtc]"
audio-core vad sample.wav --backend webrtc_v1 --webrtc-aggressiveness 2
```

生产作业仍默认使用兼容旧版本的 `adaptive_energy_v1`，只有显式设置作业参数才切换后端。

## 标注清单

校准清单为 JSONL，每行一个音频。路径相对于清单目录解析：

```json
{"id":"call-001","audio":"audio/call-001.wav","reference_segments":[{"start_ms":320,"end_ms":1840},{"start_ms":2250,"end_ms":4970}],"tags":{"domain":"customer-service","snr":"low"}}
```

参考区间应由人工审校并使用原始音频全局毫秒时间轴。同一清单中的 `id` 必须唯一。

## 参数扫描

能量阈值：

```bash
audio-vad-eval data/vad-eval/manifest.jsonl \
  --backend adaptive_energy_v1 \
  --energy-threshold -48 \
  --energy-threshold -42 \
  --energy-threshold -36
```

WebRTC 激进度：

```bash
audio-vad-eval data/vad-eval/manifest.jsonl \
  --backend webrtc_v1 \
  --webrtc-modes 0,1,2,3
```

使用 `--backend all` 可在已安装 WebRTC 依赖时同时比较两类后端。输出为 `report.json` 和 `report.md`。

## 指标口径

- Precision / Recall / F1：把标注区间和预测区间视为毫秒集合，按重叠时长计算 TP、FP、FN。
- IoU：语音区间交集时长除以并集时长。
- 边界 MAE：标注片段与预测片段按时间顺序配对后，开始和结束边界的平均绝对误差。
- RTF：检测耗时除以音频时长。

最佳候选首先要求所有样本成功，再依次按 F1、IoU 和速度选择。边界按顺序配对是基线方法；严重过分段或漏分段的数据应结合样本明细人工复核。
