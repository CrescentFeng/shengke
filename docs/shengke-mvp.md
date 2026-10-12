# 声刻 1.0.0 · 工程验证版

当前四种模型均已在 **Mac CPU 上完成真实合成**：CosyVoice2 克隆与情绪指令、GPT-SoVITS v2 克隆、ESPnet 中文 VITS 及 Transformer + Parallel WaveGAN 基线。上传、质检、排队、保存、播放、下载和盲听评分工具均已实现。不能把接口成功当成听感验证成功；N=1 八样本真人自听已完成；RTX 4090D 上四模型各 36 次正式测试均通过，共 144/144，实际消费 2.34 元并已关机。Mac 本地工作台已恢复；第四轮真人复核已完成；第五轮 GPT 单句节奏获用户接受，CosyVoice 机械感减轻，GPT 可复用停顿试用功能已接入，开心表达仍有改进空间，见 [验收记录](shengke-acceptance.md)。

## 启动

### 本机日常启动（环境已安装）

双击 [`scripts/启动声刻.command`](../scripts/启动声刻.command)，后台就绪后自动打开工作台，可关闭启动终端。重复启动复用健康实例；Docker 未运行时尝试打开并等待。双击 [`停止声刻.command`](../scripts/停止声刻.command) 结束模型与工作台，数据库和历史保留。完整日常说明及故障处理见 [v1.0 使用与发布说明](shengke-v1-release.md)。

```bash
/opt/anaconda3/envs/audio-ai/bin/python scripts/shengke_service.py start --open
/opt/anaconda3/envs/audio-ai/bin/python scripts/shengke_service.py status
/opt/anaconda3/envs/audio-ai/bin/python scripts/shengke_service.py stop
```

配置在 `configs/shengke/local-macos.json`；前台开发命令 `scripts/shengke_stack.py --check` 可检查路径。现有环境与模型复用，不自动安装或下载，不增加登录项。

### 首次安装与分开启动

平台使用 Python 3.11+；每种模型使用独立环境。不要把各模型的 Torch、Transformers、NumPy 依赖安装进同一个环境。模型服务只监听本机端口，平台通过 HTTP 调用。

```bash
# 在项目根目录执行；首次启动需要 Docker 构建基础服务。
docker compose -f infra/compose.yaml up -d postgres minio migrate
python -m pip install -e '.[dev,shengke-ui]'
python scripts/shengke_local.py --ui-python "$(command -v python)"
```

本机现有环境可直接运行：

```bash
PATH=/opt/anaconda3/bin:$PATH /opt/anaconda3/envs/audio-ai/bin/python \
  scripts/shengke_local.py --ui-python /opt/anaconda3/envs/game_tts/bin/python
```

工作台：<http://127.0.0.1:7860>；API 文档：<http://127.0.0.1:8000/docs>。日志写入 `artifacts/shengke/logs/`。终止启动器会同时停止其 API、worker 和 UI 子进程；模型服务需单独停止。默认数据库和对象存储账号仅用于本地开发。

另开终端启动已有的 CosyVoice2：

```bash
export SHENGKE_MODEL=cosyvoice2
export SHENGKE_DEVICE=cpu
export SHENGKE_FP16=0
export SHENGKE_REPO_ROOT="$HOME/Models/CosyVoice"
export SHENGKE_MODEL_DIR="$SHENGKE_REPO_ROOT/pretrained_models/CosyVoice2-0.5B"
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"
/opt/anaconda3/envs/game_tts/bin/python -m uvicorn server:app \
  --app-dir services/tts-runtime --host 127.0.0.1 --port 9881
```

启动会计算权重 SHA-256 并加载模型；`GET /healthz` 返回 ready 后才能合成。CosyVoice 当前关闭可选文本前端；请把数字和英文缩写写成实际读法。12 条评测文本已按此方式整理。

## 使用流程

v1.0 日常界面为“合成工作台、历史与下载、专业 · 对比与评测”。原专业能力保留在第三个入口，旧链接兼容。历史可恢复回听、下载 WAV/JSON 并载入参数，详见 [版本说明](shengke-v1-release.md)。

1. 添加 3–10 秒单人普通话 WAV/FLAC（最多 10 MiB），填写准确逐字稿、来源与许可。保留原始采样率；不套用 ASR 的 16 kHz/4 秒预处理。质检拒绝时长越界与近乎静音的素材，提示削波、静音比例和立体声风险；是否单人、文本是否准确仍需人工核对。
2. 选择已保存音色，输入最多 200 字。CosyVoice2 支持中性、开心、悲伤、愤怒；GPT-SoVITS v2 首版仅中性克隆。未部署的引擎会明确失败，不会偷偷切换成其他模型。
   GPT-SoVITS 可选“按标点停顿（试用）”：默认关闭，适合每段至少八个汉字的普通话长分句；短分句保持整句朗读。此选项只作用于工作台本次生成，“对比试听”仍使用基线配置。
   某些参考录音仍可能导致分句提前结束。系统会丢弃异常结果并整句重试一次，状态栏明确说明；重试也失败则报错。生成后仍需核对实际朗读内容。
   GPT-SoVITS 另可选“舒缓语速与停顿（试用）”：适用于中文及标点，0.9 倍速；按实际音频定位停顿，定位不可靠的标点会跳过并提示。生成会增加本地识别耗时。开启后旧“按标点停顿”开关禁用，避免叠加。
   CosyVoice2 开心模式可选“舒缓节奏（试用）”：使用短开心指令并在生成后以 0.85 倍速拉伸，默认关闭。单句试听反馈机械感有所减轻；开心强度及跨文本效果尚未验收。
3. 点击生成后，页面显示排队与生成状态；成功后播放、下载 WAV。“任务详情”可刷新或取消。取消不能中断已进入模型内部的推理，但迟到结果不会发布为成功作业。
4. “对比试听”沿用工作台音色与文本，可生成中性/情绪或两引擎对照。
5. 将已完成的作业编号粘贴至“评测中心”，创建盲听集合，使用稳定匿名编号加载随机顺序样本，听后打分并导出 CSV。相同评测人对同一样本的再次提交会更新原记录。

## 模型边界与复现

| 模型 | 用途 | 状态 |
|---|---|---|
| CosyVoice2-0.5B | 3–10 秒克隆、情绪指令 | Mac CPU 与 RTX 4090D FP32 验证通过 |
| GPT-SoVITS v2 | 第二克隆引擎 | Mac CPU 与 RTX 4090D 各 36 次基准通过 |
| ESPnet CSMSC VITS | 固定音色中文 TTS 基线 | Mac CPU 与 RTX 4090D 各 36 次基准通过 |
| ESPnet CSMSC Transformer + 匹配声码器 | Transformer TTS 基线 | Mac CPU 与 RTX 4090D 各 36 次基准通过 |

模型目录和版本记录于 `configs/shengke/models.json`；每次成功任务报告保留代码版本、权重哈希、硬件、精度、Torch/Python 版本、模型加载时间、含声码器的推理耗时、音频时长和 RTF。CPU RSS 与 CUDA allocated/reserved 分开记录；CPU 上 CUDA 指标为 null。固定随机种子不能保证跨硬件逐字节相同。

平台的 `SHENGKE_MODEL_ENDPOINTS` 是模型 ID 到地址的 JSON；默认仅 CosyVoice2 的 9881 端口。不同模型独占一个服务进程和端口。GPU 部署配置见 [模型服务部署](shengke-runtime.md)。

## 基准与听评

```bash
PYTHONPATH=packages/audio-core/src:apps/tts-eval/src python -m tts_eval.cli \
  data/tts-eval/mandarin-v1.jsonl --voice-id '<已上传音色 UUID>' \
  --models cosyvoice2 --repeats 3 --output artifacts/shengke/benchmark-cpu
```

每模型先预热一次，再对每条文本重复三次。输出必须为空目录以保护旧证据。保存原始 `runs.jsonl`、独立的预热记录、WAV、P50/P95、成功率和可用于听评的作业列表。失败项留在分母；下载证据失败也算失败。正式跨模型比较必须使用同一硬件、精度策略、文本和参考素材，并单独说明固定音色基线不支持克隆。

第一轮可以只做 N=1 自听预评测，记录自然度、克隆相似度、目标情绪匹配度与问题。自然度 1=很不自然，5=非常自然；相似度和情绪匹配度分别按很差至很好评分。没有真人评分就没有听评结论；自动质检不能替代 MOS。系统在至少五名评测人完成全部样本后才标记为多人听评，这个门槛本身不保证统计代表性。

工作台已导入 THCHS-30 两条原始短句及六条分档参考：D11、D12 两位说话人各 3/5/10 秒。原始文件来源与哈希见 `data/tts-eval/public-references.json`，分档来源、裁剪与限制见 [评测素材说明](../data/tts-eval/README.md)。素材均已通过上传质检，用户已完成核对；D12 十秒逐字稿的更正与发音疑点见素材说明。

公开参考素材优先使用有明确来源与许可的语料。[THCHS-30](https://www.openslr.org/18/) 页面标注 Apache-2.0，计划选取两个说话人的短样本，保留原文件哈希、截取区间与对应逐字稿。现有 CosyVoice 仓库 demo 仅用于技术冒烟验证，其说话人单独授权尚未核实，不作为可再分发的产品音色包。

## 验证证据与待完成项

2026-10-11 本机验证：83 项常规测试及 3 项独立 PostgreSQL 租约/取消并发测试通过。`artifacts/shengke/smoke-cosyvoice2-cpu-v2/` 保存中性合成 WAV、作业、音色与报告；6.52 秒音频，CPU 推理 12.70 秒、RTF 1.95。`artifacts/shengke/smoke-cosyvoice2-happy-cpu/` 保存开心指令结果；5.04 秒音频，CPU 推理 10.03 秒、RTF 1.99。这些是单次功能验证，不是经过预热和重复的正式性能结论；两次早于固定种子补丁。

同一 RTX 4090D 上的四模型基准已完成，GPU 证据已纳入验收记录。参考核对、两轮八样本与一轮两样本评分已完成；第四轮三样本复核也已完成。后续第五轮 GPT 单句节奏获用户接受、CosyVoice 舒缓节奏已作为可选功能接入；GPT 可复用停顿处理已作为试用功能接入并完成五例实测，开心表达质量仍有改进空间。

首轮开发机基准已完成：`artifacts/shengke/benchmark-cosyvoice2-cpu-demo/` 使用官方 demo 参考声、12 条普通话文本，每条重复 3 次，排除 1 次预热，36/36 成功。平均 RTF 2.42，RTF P50/P95 为 2.38/2.75；推理耗时 P50/P95 为 11.21/13.94 秒；最大采样 RSS 约 3.77 GiB。此结果仅描述当前 Mac CPU FP32 环境，运行期间有开发与依赖下载活动，不能用于承诺部署 SLA，也不能推出 GPU 性能或主观听感质量。

## 五分钟演示顺序

- 0:00–0:40：展示参考音色的来源、时长与准确逐字稿。
- 0:40–1:40：生成中性短句，播放并下载，展示任务状态。
- 1:40–2:40：同音色、同文本的情绪对照，让听众自行判断差异。
- 2:40–3:40：展示已实测模型的原始耗时、RTF、内存与失败项；未实测项明确留空。
- 3:40–5:00：盲听评分、CSV 导出与真实局限。不要把个人预评测写成多人 MOS。

Transformer 开发机基准（公开参考集 D12，固定音色模型实际不使用参考声）：36/36 成功，平均 RTF 1.12；RTF P50/P95 为 0.97/1.91；推理耗时 P50/P95 为 3.75/6.68 秒。原始记录位于 `artifacts/shengke/benchmark-transformer-cpu-public-D12/`。首轮个人试听中该模型一份样本自然度为 3/5；不能据此或依据速度判定其整体音质优于克隆模型。

VITS 开发机基准（与 Transformer 使用相同公开 D12 参考配置和 12 条文本）：36/36 成功，平均 RTF 0.25；RTF P50/P95 为 0.22/0.34；推理耗时 P50/P95 为 0.78/1.51 秒。证据位于 `artifacts/shengke/benchmark-vits-cpu-public-D12/`。VITS 是固定音色基线，不能把更快的生成速度解释为具备克隆或情绪控制能力。

公开分档参考集已完成：D11、D12 两位说话人各 3/5/10 秒，六份音频均通过真实上传及质检，无削波/高静音比提示。配方见 `data/tts-eval/reference-crops.json`，哈希与来源见 `data/tts-eval/public-references-3-5-10.json`。这些裁剪基于公开词级标注，用户已核对并更正 D12 十秒逐字稿；不同长度使用不同句子，不能单独归因于参考时长。

CosyVoice2 使用公开 D12 音色的同文本基准也已完成：36/36 成功，平均 RTF 2.85，RTF P50/P95 2.84/3.34；推理耗时 P50/P95 13.37/16.71 秒。证据位于 `artifacts/shengke/benchmark-cosyvoice2-cpu-public-D12/`。与两种 ESPnet 基线使用同一开发机、FP32 和 12 条文本，各排除一次预热后重复三次。开发机同时有下载等后台活动，这些数据不代表独占硬件的部署 SLA。

参考范围验证完成：六份公开 3/5/10 秒素材均完成 CosyVoice2 的一次预热和一次正式克隆，12 次全部成功；音频与指标位于 `artifacts/shengke/reference-range-cosyvoice2/`。工作台的 10 秒参考输入也已实际生成可播放、可下载的结果。六份参考素材已由用户核对，D12_991_10s 文本更正已记录；各时长条件的克隆相似度仍需专门听评。

本次 CPU 基准硬件为 Apple M5 Pro、24 GiB 内存、15 个 CPU 核心，模型进程各设置 4 个 Torch CPU 线程；系统读取证据见 `artifacts/shengke/comparison-cpu/hardware.json`，汇总见该目录 `report.md`。

GPT-SoVITS v2 整句模式基准：36/36 成功，平均 RTF 0.40，RTF P50/P95 0.39/0.44；推理耗时 P50/P95 1.63/2.07 秒。完整证据位于 `artifacts/shengke/benchmark-gpt-v2-cpu-public-D12/`，工作台界面克隆与下载已验证。四种模型共 144 次正式 CPU 基准请求全部成功，另有四次排除在统计外的预热。

GPT-SoVITS 同样完成六份 3/5/10 秒参考的预热与正式克隆，12 次全部成功；两个克隆引擎的分档功能验证合计 24 次成功。逐项证据见 `artifacts/shengke/reference-range-summary.json` 及对应引擎目录。

D12 十秒逐字稿更正为“降落”后，两引擎另行各完成一次预热和一次正式合成，4/4 成功；实际作业已核验更正文本及不变的参考音频哈希。证据独立保存在 `artifacts/shengke/reference-correction-D12-10s/`，旧 24 次范围验证不覆盖，听感改善未作结论。

已准备八样本盲听集合“声刻 MVP · 四模型与情绪 · 八样本自听预评测”，编号 `aa3b864b-3131-4e83-9626-6e01380192ac`。用户已完成该集合 8/8 份评分（N=1），结果见 [验收记录](shengke-acceptance.md)。六份克隆样本相似度均为 5 分；开心、愤怒、悲伤匹配度分别为 1、3、4 分；中性语音存在断句和机械感反馈。

根据首轮反馈，第二轮对照集合 `bae8964a-1641-4cb1-8dbb-de7cb564e2fc` 已完成 8/8 份 N=1 真实评分：四份原始音频加四份候选，检验开心指令和中性语速调整。CosyVoice 中性两句自然度一升一降；开心匹配度 2→3、自然度 4→3；GPT-SoVITS 降速后自然度 3→2、相似度 5→4。结果不支持整体推广，实验配置 `pilot_v2` 保留供复现，默认继续使用 `baseline`。完整配对结果和原始证据见验收记录。评分页支持进度显示、恢复已有分数及“保存并继续下一个”。

第三轮 `6f8c71ac-463a-4aea-84ca-642a2b067f86` 已完成 2/2 份 N=1 评分。GPT-SoVITS 分句候选自然度 3→4，相似度保持 5；用户确认逗号断句更自然，但停顿略长。已作为可选试用模式接入工作台，未更改默认设置；当前结论限于这句文本和本位评测人的听感。
