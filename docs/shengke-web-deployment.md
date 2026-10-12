# 声刻：网站展示与独立推理

2026-10-11 用户确认采用：现有个人网站负责展示和入口，Mac 继续开发；需要 GPU 实测时按小时租机。当前实施为本地预览，没有新购服务或修改公网 DNS。

## 展示页

源文件为 `apps/shengke-showcase/index.html`，样本为同目录 `audio/gpt-sovits-demo.wav`。复制此目录至个人作品集的 `public/shengke/`，主页入口指向 `/shengke/index.html`。沿用现有静态网站构建；无需把模型、数据库或 Python 环境放入网站。

展示页以本机域名识别工作台入口：只有 localhost、127.0.0.1、[::1] 展示本机工作台链接。公网访问只显示预生成试听与“在线生成暂未开放”。不能将公网体验按钮指向 127.0.0.1，因为那会访问访客自己的电脑。

演示音频现取自 `artifacts/shengke/gpt-rhythm-v6/matrix/accepted-reference/speech.wav`，作业 `f151736b-c85e-491f-8c99-72a0f4552b81`，5.7588 秒，SHA-256 `ee530689c7002caf3725a9bb3b9526f4585707c8105ded89a9a8f96028a07fbb`。这是 GPT 生成后经停顿与 0.9 倍速处理的结果，PCM 与用户接受的节奏候选一致；不代表所有文本质量均已验收。参考语料说明见 `data/tts-eval/README.md`。

## 本地开发

日常双击 `scripts/启动声刻.command`，统一启动四模型、API、worker 与 Gradio UI；也保留 `scripts/shengke_local.py` 与 `scripts/shengke_model.py` 的独立启动方式。完整环境步骤见 `docs/shengke-runtime.md`。服务保持 loopback 监听，不因网站展示而对外开放本地数据库、对象存储或模型接口。

本机工作台： http://127.0.0.1:7860/?view=synthesis 。展示页与工作台分开运行，静态页面不提交任务、不上传参考录音、不自动调用推理服务。

## GPU 实测

沿用 `docs/shengke-gpu-runbook.md`：按小时实例仅运行模型，SSH 隧道将远端 9881–9884 映射至本机 19881–19884。队列清空后，通过 `SHENGKE_MODEL_ENDPOINTS` 切换本机 worker；完成实测后切回本地端点。2026-10-12 已完成 RTX 4090D 实测，144/144 次正式请求成功，实际消费 2.34 元，实例已关机。

数据库、评分与工作台留在 Mac；只传公开权重、代码及必要的公开参考样本。GPU 结果仍必须来自真实 CUDA 运行。

## 后续公网生成

展示页可随已有网站正常部署；真正在线生成需要另外接入可用的后端地址、HTTPS、访问控制、请求限额和作业状态接口，再启用公网体验入口。现有 Gradio 工作台不能作为静态文件复制后直接运行。可以沿用现有域名的子域名，无需另购域名。

上线前核对网站托管实际部署方式和受众，保持现有作品与域名配置；本次本地接入不代表网站已发布；GPU 验收另见实际测试记录。

## 2026-10-11 历史预览验证

个人网站本地 `public/shengke/` 已包含展示页和音频，`app/page.tsx` 的导航增加中英文入口，保留原有作品改动。已用安装好的 Vinext 执行生产构建并通过；标准构建 helper 触发 pnpm 依赖检查时遇到 npm DNS 失败，已终止该次检查，直接构建成功。

展示页与音频 HTTP 均返回 200；浏览器验证主页入口跳转正确、音频 duration=5.02、readyState=4、无媒体错误。本机工作台返回 HTTP 200。未重新生成音频或改写听评分数。

预览地址：`http://127.0.0.1:4176/shengke/index.html`。如预览进程结束，在终端运行：

```bash
/opt/anaconda3/envs/audio-ai/bin/python -m http.server 4176 --bind 127.0.0.1 \
  --directory "$HOME/path/to/personal-portfolio/dist/client"
```

展示页更新后需要重新同步至 `public/shengke/` 并重新构建；静态预览不会自动刷新构建文件。

## 2026-10-12 源文件更新

本项目 `apps/shengke-showcase/` 已更新试听音频、四轮 21 份评分说明、GPU 实测与停止计费状态。个人网站目录和旧 4176 构建预览本轮尚未同步，不应据本项目源文件更新声称旧预览或公网已更新。发布前按上述步骤同步并构建。

更新后的独立本地预览已在 `http://127.0.0.1:4177/` 启动，浏览器确认新试听音频时长 5.758813 秒、readyState=4、无媒体错误。此地址直接服务本项目展示源目录，与旧个人网站 4176 构建预览分开。

## 2026-10-12 个人网站同步完成

已备份并同步展示 HTML 与新试听 WAV 至个人网站 `public/shengke/`，保留主页及其他作品的既有修改。Vinext 生产构建成功，预渲染两条路由。本地 4176 预览已恢复：页面和音频 HTTP 200，均与声刻源文件 SHA-256 一致；浏览器确认 duration=5.758813、readyState=4、无媒体错误。证据位于 `artifacts/shengke/site-sync-20261012/`，含同步前备份、校验结果及页面截图。此前“尚未同步”描述仅对应源文件更新阶段；当前本地作品集已同步，没有发布公网。

## 声刻 1.0.0 本地工程验证版

2026-10-12 展示页状态、工作台说明和个人作品集导航已同步为 v1.0；Vinext 构建通过。按需后台启动、历史与下载见 `shengke-v1-release.md`。试听音频保持用户接受的既有样本，个人网站仍仅更新本地预览；没有发布公网生成。最新发布证据在 `artifacts/shengke/release-v1/`。
