# ASR 可重复评测

`audio-asr-eval` 使用 JSONL 清单驱动同一数据集上的模型评测，输出 JSON 契约和 Markdown 报告。每行代表一个样本：

```json
{"id":"meeting-001","audio":"audio/meeting-001.wav","reference_text":"今天讨论音频智能发布计划","language":"zh-CN","hotwords":["音频智能"],"reference_spans":[{"start_ms":120,"end_ms":1860,"text":"今天讨论音频智能发布计划"}],"tags":{"domain":"meeting","noise":"office"}}
```

`audio` 相对于清单所在目录解析。`id` 在单份清单中必须唯一；音频文件必须在运行前存在。`reference_spans` 可省略，省略时不会计算时间戳边界误差。

`hotwords` 可省略。比较热词收益时，应为同一音频分别准备无热词和有热词样本，保持模型、设备、beam size 和其他参数一致。

## 运行

Faster-Whisper：

```bash
pip install -e ".[asr-whisper]"
audio-asr-eval data/eval/manifest.jsonl \
  --backend faster-whisper \
  --model small \
  --device cpu \
  --compute-type int8 \
  --output artifacts/asr-eval/faster-whisper-small
```

FunASR：

```bash
pip install -e ".[asr-funasr]"
audio-asr-eval data/eval/manifest.jsonl \
  --backend funasr \
  --model paraformer-zh \
  --device cpu \
  --output artifacts/asr-eval/funasr-paraformer-zh
```

若未指定 `--output`，报告会写入带 UTC 时间戳的 `artifacts/asr-eval/` 子目录。任一样本失败时，评测仍会完成并记录失败原因，但命令以状态码 `2` 退出，便于 CI 识别不完整结果。

## 指标口径

- WER：Unicode NFKC、大小写折叠、去标点后，按词计算编辑距离；适合有自然空格分词的文本。
- CER：同样规范化后去空白、按 Unicode 字符计算编辑距离；中文等语言应优先看 CER。
- RTF：单条推理耗时除以音频时长；小于 1 表示推理快于实时播放。
- 语种准确率：仅统计清单给出语种且后端返回检测语种的成功样本，地区后缀会归一化。
- 时间戳边界 MAE：参考片段与预测片段按顺序配对，对开始和结束边界计算平均绝对误差。
- 内存：进程值来自操作系统的生命周期峰值 RSS；GPU 值在 NVIDIA 环境中通过 `nvidia-smi` 采样推理期当前进程显存。它们用于同机同环境相对比较，不应跨机器直接比较。

正式比较前应固定数据集版本、模型版本、设备、精度、beam size 和预热策略。当前 runner 不主动预热模型，首次推理开销会进入结果；这符合冷启动评测口径。
