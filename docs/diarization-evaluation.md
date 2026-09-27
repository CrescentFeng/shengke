# 说话人分离 DER/JER 评测

`audio-diarization-eval` 复用生产 `DiarizationBackend`，读取人工标注 speaker turn，生成
JSON 与 Markdown 报告。准确率矩阵只接受 `audio_kind=real`、
`annotation_kind=human`、原始 RTTM、音频/RTTM SHA-256，以及完整数据集来源标签的样本；
合成、来源不明或不可复现的样本会被拒绝，不能用于生产准确率结论。

## 数据集与许可

优先使用业务域内、已取得授权并由人工复核的真实多人音频。公开回归集可以采用
[VoxConverse 0.3](https://github.com/joonson/voxconverse)：官方提供 WAV 与 RTTM，许可为
CC BY 4.0，但视频原始版权仍归原权利人。公开集只能衡量通用域表现，不能替代客服、会议、
播客等目标域验收。

真实音频放入 `data/diarization-eval/audio/`，RTTM 放入
`data/diarization-eval/rttm/`。两者默认被 `.gitignore` 排除；只提交不含音频内容的 JSONL
清单、数据集版本和许可记录。原始下载包放在
`data/diarization-eval/sources/`，同样被 `.gitignore` 排除。

### VoxConverse 0.3 dev 固定版本

本项目的可复现公开回归集使用 216 条 dev 录音与人工 RTTM：

- WAV 包 SHA-256：`e83a68b5df3bc945a3cf4544102038792ae79972753c585769e58ea677c523a8`；
- 标注固定到 VoxConverse commit
  `24bf60be297701cd7e4ef18550c6d390c1b87365`，归档 SHA-256：
  `c340d2cd9da2f19eff8a3adc55b0818ec45cd0770ef72c6926c4de6efe9e9a1d`；
- 来源与许可审计：`data/diarization-eval/voxconverse-dev-v0.3-source.json`；
- 完整清单：`data/diarization-eval/voxconverse-dev-v0.3-full.jsonl`。

用确定性分层算法选取验收集，覆盖 speaker 人数和时长四分位：

```bash
conda run -n audio-ai python scripts/select_diarization_subset.py \
  data/diarization-eval/voxconverse-dev-v0.3-full.jsonl \
  --output data/diarization-eval/voxconverse-dev-v0.3-acceptance.jsonl \
  --minimum-samples 20 \
  --minimum-duration-seconds 1800 \
  --minimum-speakers 2 \
  --seed voxconverse-0.3-acceptance-v1
```

输出的 `.selection.json` 记录算法、seed、样本 ID、speaker 人数分布和时长分布；
源清单使用相对路径，不依赖生成机器的绝对目录。

## RTTM 与 JSONL 清单

标准 RTTM 行：

```text
SPEAKER meeting-001 1 0.120 1.720 <NA> <NA> ALICE <NA> <NA>
SPEAKER meeting-001 1 2.010 2.310 <NA> <NA> BOB <NA> <NA>
```

JSONL 可以直接引用 RTTM；SHA-256 在加载时强制校验：

```json
{"id":"meeting-001","audio":"audio/meeting-001.wav","rttm":"rttm/meeting-001.rttm","recording_id":"meeting-001","audio_sha256":"<64位小写SHA-256>","rttm_sha256":"<64位小写SHA-256>","audio_kind":"real","annotation_kind":"human","tags":{"dataset":"internal-meetings","dataset_version":"2026-09-27","split":"test","license":"internal-approved"}}
```

使用准备脚本生成确定性清单：

```bash
python scripts/prepare_diarization_manifest.py \
  --audio-root data/diarization-eval/audio \
  --rttm-root data/diarization-eval/rttm \
  --output data/diarization-eval/manifest.jsonl \
  --dataset voxconverse --dataset-version 0.3 --split dev \
  --license CC-BY-4.0
```

脚本按 recording id 排序、解析并校验 RTTM，同时固定音频和标注哈希。内联
`reference_turns` 仍用于小型单元测试；正式验收应保留原始 RTTM。

## 运行

```bash
docker compose --env-file .env.local -f infra/compose.yaml run --rm ingest-worker \
  audio-diarization-eval /data/diarization-eval/manifest.jsonl \
  --model pyannote/speaker-diarization-community-1 \
  --matrix \
  --frame-ms 10 \
  --collar-ms 0 \
  --boundary-tolerance-ms 250 \
  --speaker-bound-padding 1 \
  --output /artifacts/diarization-eval/community-1
```

不要把 token 写入清单、命令参数、已跟踪的环境文件或日志。Compose 使用 `.env.local` 中的
`AUDIO_PLATFORM_HF_TOKEN_HOST_PATH` 只读挂载 token 文件，并复用持久化 `model-cache`。
真实评测数据以只读方式挂载，报告写入 `artifacts/diarization-eval/`。

矩阵固定运行六种配置：

- regular / exclusive 两类时间轴；
- `oracle_num`：使用人工标注人数，仅表示诊断上限；
- `bounded`：使用参考人数 ± `speaker-bound-padding`；
- `unconstrained`：不提供人数信息，最接近完全自动的生产场景。

Community-1 每次推理会同时返回 regular 与 exclusive，因此每个人数约束只执行一次模型
推理，再分别对两条时间轴计分。两份报告记录同一次真实推理耗时，避免缓存命中被误记为
接近零的 RTF；六份报告共对应三次模型推理。

单次评测默认使用 exclusive；`--regular` 改为常规、允许重叠的时间轴；
`--skip-overlap` 从计分区域排除参考重叠语音。正式报告必须明确 collar、是否跳过重叠，
不能混合比较不同计分口径。

## 指标口径

评测以固定帧长近似连续时间：

- 使用最大权重一对一匹配消除参考与预测 speaker 标签名称差异；
- DER = `(false alarm + missed detection + speaker confusion) / reference speaker time`；
- JER 对每个参考 speaker 计算最优匹配后的 `1 - intersection / union`，再取平均；
- RTF 为模型推理耗时除以音频时长；
- `collar_ms` 表示在每个参考边界两侧各排除的毫秒数。
- speaker-count accuracy 为预测人数与参考人数完全相等的样本比例；
- 边界误差在最优 speaker 映射后分别匹配 start/end，报告 MAE、P95 和指定容差内召回率。

默认初始工程门槛为 DER ≤ 25%、JER ≤ 35%、speaker-count accuracy ≥ 90%、边界
P95 ≤ 500ms、平均 RTF ≤ 0.75；至少需要 20 个样本且总时长不少于 30 分钟才可能给出
`accepted`。这些是保守的起始门槛，必须结合目标业务的漏检成本、重叠比例和硬件 SLA
调整。样本不足时报告状态为 `insufficient_data`，即使指标很好也不能宣布生产验收通过。
只有不依赖参考人数的 `unconstrained` 配置能够触发 `accepted`；`oracle_num` 和由参考人数
派生的 `bounded` 即使通过门槛，也只作为诊断对照，不能成为生产推荐。

报告会列出每个配置的推理失败、speaker 数错误以及 DER 最高的样本。排查顺序建议：音频
损坏或解码失败、标注与音频错位、近场/远场域差异、重叠语音、极短 turn、人数估计偏差，
最后才考虑更换人数约束或模型。

帧长、collar、重叠处理和 exclusive 选择都会写入报告。该实现用于稳定的工程回归；对外发表或与论文数字对比时，应同时用官方 `pyannote.metrics` 在相同 RTTM 和计分口径下复核。
