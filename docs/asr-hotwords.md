# ASR 术语热词

热词用于提示解码器关注产品名、人名、专业缩写等低频术语。它不是转写后的字符串替换：模型仍可能忽略、误识别或过度偏向热词，因此必须配合 ASR 评测使用。

## 作业参数

```json
{
  "operations": ["vad", "transcribe"],
  "parameters": {
    "asr": {
      "language": "zh",
      "hotwords": ["音频智能", "客服质检", "DeepFilterNet"]
    }
  }
}
```

约束：

- 最多 100 个热词；
- 单个术语最多 128 个字符；
- 总字符数最多 4096；
- 禁止空白术语；
- 去除首尾空白，并按大小写折叠去重，保留第一次出现的写法。

最终热词列表会写入 ASR `ToolRun.parameters`，`hotword_count` 写入 metrics，便于回溯。

## 后端映射

- Faster-Whisper：使用 `transcribe(..., hotwords="term1 term2")`。
- FunASR：使用 `generate(..., hotword="term1 term2")`。

FunASR 的热词行为取决于具体模型。SeACo/Contextual Paraformer 会消费 `hotword`，但 SenseVoice 等模型不应被假定支持通用热词。选型报告必须记录完整模型 ID 和版本。

## 评测

ASR JSONL 清单的每个样本可设置 `hotwords`。推荐为同一批音频运行两份清单：基线清单不带热词，实验清单只增加热词，随后比较整体 WER/CER 和术语子集错误。

当前报告提供整体 WER/CER；术语召回率、误触发率和热词权重扫描将在后续增量中实现。
