# 声刻评测素材

- `mandarin-v1.jsonl`：12 条项目自编文本，用于同硬件、同配置的合成基准。
- `emotion-v1.jsonl`：同一句话的中性、开心、悲伤、愤怒对照。
- `public-references.json`：两条未经裁剪的 THCHS-30 公开参考音频，7.192 秒和 4.730 秒。
- `reference-crops.json`：两位说话人各 3、5、10 秒的裁剪配方，保存原始词级时间标注及逐字稿。
- `public-references-3-5-10.json`：准备脚本生成的六条素材清单，包含原始与裁剪音频的 SHA-256。

## 来源与许可

原始语料：[THCHS-30，OpenSLR 18](https://www.openslr.org/18/)，由清华大学发布，页面标注 Apache-2.0。音频取自 [FluidInference/THCHS-30-tests](https://huggingface.co/datasets/FluidInference/THCHS-30-tests)，固定版本 `cb125be05c3059f5ef7ad796dfd460a026508dff`。词级标注来自 [anyspeech/THCHS-30-alignments](https://huggingface.co/datasets/anyspeech/THCHS-30-alignments)，固定版本 `9bd41492e5dbc9d69a4cbea0acbc03b9b20375a4`，原 Parquet SHA-256 为 `4686a0448eebc4177203c601d38bc27da152bfaa6e17875f25248cf80a778194`。

准备脚本校验固定版本的音频文件哈希，在词语边界外的原有静音区间裁剪 PCM 帧，保留原采样率；不拉伸、不重采样、不补静音。初始逐字稿取自原始公开逐字稿的连续片段。

用户已于 2026-10-11 完成六份参考音频核对：五份为 `verified`；D12_991_10s 按用户反馈把“溅落”更正为“降落”，标记为 `reviewed_with_correction`。用户同时指出该处可能有发音问题，原因尚未确认；原始词级标注与更正前文本保留供追溯。复核记录绑定音频 SHA-256；重新准备后只有哈希一致才继承复核状态，否则回到 `pending`。

```bash
python scripts/shengke_prepare_references.py
python scripts/shengke_import_voices.py \
  --manifest data/tts-eval/public-references-3-5-10.json \
  --output artifacts/shengke/public-voices-3-5-10.json
```

原始及裁剪音频存放在 Git 忽略的数据目录，仓库保留来源和复现配方。重新准备素材前不要手工修改生成文件；调整边界应先改配方并重新生成、校验和导入。

## 对照的局限

分档素材来自同一说话人的不同句子；5 秒档的两位说话人使用相同文本，10 秒档原为相同文本，但 D12 的“降落”更正与 D11 保留的“溅落”现在不同，3 秒档使用不同短句。因此它适合验证 3–10 秒输入范围及展示克隆效果，不能单独证明“参考时间越长，克隆效果越好”。音频长度、内容、语速和录音条件的影响不能混为一个因果结论。

机器质检、合成成功率与人工 MOS 分开报告。没有真实评分时，MOS 保持未执行；N=1 自听只作为个人预评测。
