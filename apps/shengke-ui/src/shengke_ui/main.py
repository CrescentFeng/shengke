from __future__ import annotations

import base64
import json
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import gradio as gr

RELEASE = json.loads(
    (Path(__file__).resolve().parents[4] / "configs/shengke/release.json").read_text()
)

API = os.getenv("SHENGKE_API_URL", "http://127.0.0.1:8000").rstrip("/")
MODELS = [("CosyVoice2 · 克隆与情绪", "cosyvoice2"), ("GPT-SoVITS · 克隆", "gpt-sovits-v2")]
EMOTIONS = [("中性", "neutral"), ("开心", "happy"), ("悲伤", "sad"), ("愤怒", "angry")]


def request(path, body=None, *, binary=False):
    import gradio as gr

    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(API + path, data, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise gr.Error(f"请求未完成：{detail}") from exc
    except OSError as exc:
        raise gr.Error("无法连接声刻服务，请检查服务是否启动") from exc
    return raw if binary else json.loads(raw)


def audio_file(path):
    raw = request(path, binary=True)
    suffix = ".flac" if raw.startswith(b"fLaC") else ".wav"
    with tempfile.NamedTemporaryFile(
        prefix="shengke-audio-", suffix=suffix, delete=False
    ) as handle:
        handle.write(raw)
        return handle.name


def load_voices():
    import gradio as gr

    profiles = request("/v1/tts/voices")
    return profiles, gr.update(
        choices=[(p["name"], p["id"]) for p in profiles],
        value=profiles[0]["id"] if profiles else None,
    )


def upload(path, name, transcript, source, license_text):
    import gradio as gr

    if not path:
        raise gr.Error("请先上传参考音频")
    file = Path(path)
    if file.stat().st_size > 10 * 1024 * 1024:
        raise gr.Error("参考音频不能超过 10 MiB")
    profile = request(
        "/v1/tts/voices",
        {
            "name": name,
            "filename": file.name,
            "reference_text": transcript,
            "source": source,
            "license": license_text,
            "audio_base64": base64.b64encode(file.read_bytes()).decode(),
        },
    )
    profiles, _ = load_voices()
    return (
        profiles,
        gr.update(choices=[(p["name"], p["id"]) for p in profiles], value=profile["id"]),
        {
            "时长": profile["asset"]["probe"]["duration_seconds"],
            "提示": profile["warnings"],
            "状态": "已保存；请人工确认是清晰的单人普通话，并核对参考文本",
        },
    )


def create_job(
    profiles,
    voice_id,
    text,
    model,
    emotion,
    punctuation_pause=False,
    gentle_rhythm=False,
    gpt_rhythm=False,
):
    import gradio as gr

    profile = next((p for p in profiles if p["id"] == voice_id), None)
    if not profile:
        raise gr.Error("请上传或选择一个参考音色")
    return request(
        "/v1/audio/jobs",
        {
            "input_uri": profile["asset"]["uri"],
            "operations": ["synthesize"],
            "idempotency_key": str(uuid4()),
            "parameters": {
                "tts": {
                    "model": model,
                    "text": text,
                    "reference_text": profile["reference_text"],
                    "emotion": emotion,
                    "language": "zh",
                    "voice_id": voice_id,
                    "inference_profile": (
                        "gpt_rhythm_v6"
                        if gpt_rhythm and model == "gpt-sovits-v2"
                        else "gpt_pause_v3"
                        if punctuation_pause and model == "gpt-sovits-v2"
                        else "cosy_rhythm_v5"
                        if gentle_rhythm and model == "cosyvoice2" and emotion == "happy"
                        else "baseline"
                    ),
                }
            },
        },
    )["id"]


def poll(job_id):
    return request(f"/v1/audio/jobs/{job_id}")


def completion_status(job_id):
    report = request(f"/v1/audio/jobs/{job_id}/report")
    delivery = report["report"]["summary"].get("tts", {}).get("delivery", {})
    if delivery.get("inference_profile") == "gpt_rhythm_v6":
        points = delivery.get("postprocessing", {}).get("boundaries", [])
        inserted = sum(p["action"] == "insert" for p in points)
        kept = sum(p["action"] == "keep" for p in points)
        skipped = sum(p["action"] == "skip" for p in points)
        return (
            f"生成完成，已舒缓至 0.9 倍速；补入 {inserted} 处停顿，保留 {kept} 处原停顿。"
            + (f"另有 {skipped} 处定位不可靠，未插入；" if skipped else "")
            + "请试听核对内容与断句"
        )
    if delivery.get("fallback_reason") == "early_semantic_termination":
        return "分句生成提前结束，已改为整句重新生成，可能缺少标点停顿；请试听核对内容"
    if delivery.get("inference_profile") == "cosy_rhythm_v5":
        return "生成完成，已应用舒缓节奏（0.85 倍速）；请试听核对内容与情绪"
    return "生成完成，请试听核对内容"


def synthesize(
    profiles,
    voice_id,
    text,
    model,
    emotion,
    punctuation_pause=False,
    gentle_rhythm=False,
    gpt_rhythm=False,
):
    job_id = create_job(
        profiles, voice_id, text, model, emotion, punctuation_pause, gentle_rhythm, gpt_rhythm
    )
    yield job_id, "已排队", None, None
    for _ in range(900):
        job = poll(job_id)
        state = job["state"]
        if state == "succeeded":
            path = audio_file(f"/v1/tts/jobs/{job_id}/audio")
            yield job_id, completion_status(job_id), path, path
            return
        if state in {"failed", "cancelled"}:
            yield job_id, job.get("error_message") or "已取消", None, None
            return
        yield job_id, "正在生成" if state == "running" else "等待生成", None, None
        time.sleep(2)
    yield job_id, "等待超时；作业仍可通过下方刷新按钮查询", None, None


def refresh_result(job_id):
    job = poll(job_id)
    path = audio_file(f"/v1/tts/jobs/{job_id}/audio") if job["state"] == "succeeded" else None
    status = (
        completion_status(job_id)
        if job["state"] == "succeeded"
        else job.get("error_message") or job["state"]
    )
    return status, path, path


def compare(profiles, voice_id, text, mode, emotion):
    import gradio as gr

    if mode == "同音色情绪对比" and emotion == "neutral":
        raise gr.Error("请选择开心、悲伤或愤怒作为对照")
    left = create_job(profiles, voice_id, text, "cosyvoice2", "neutral")
    right = create_job(
        profiles,
        voice_id,
        text,
        "gpt-sovits-v2" if mode == "双引擎对比" else "cosyvoice2",
        "neutral" if mode == "双引擎对比" else emotion,
    )
    paths = [None, None]
    for _ in range(900):
        jobs = [poll(left), poll(right)]
        for i, job in enumerate(jobs):
            if job["state"] == "succeeded" and paths[i] is None:
                paths[i] = audio_file(f"/v1/tts/jobs/{job['id']}/audio")
        yield (
            "\n".join([left, right]),
            " / ".join(j.get("error_message") or j["state"] for j in jobs),
            paths[0],
            paths[1],
        )
        if all(j["state"] in {"succeeded", "failed", "cancelled"} for j in jobs):
            return
        time.sleep(2)


def cancel_jobs(ids):
    for job_id in ids.split():
        request(f"/v1/audio/jobs/{job_id}/cancel", {})
    return "已请求取消；正在执行的模型调用将在返回后丢弃结果"


def create_evaluation(name, ids):
    return request("/v1/tts/evaluations", {"name": name, "job_ids": ids.split()})["id"]


def listening_progress(evaluation_id, listener_id):
    return request(
        f"/v1/tts/evaluations/{evaluation_id}/progress?"
        + urllib.parse.urlencode({"listener_id": listener_id})
    )


def progress_text(progress):
    return f"已完成 **{progress['completed']} / {progress['total']}** 份评分"


def start_listening(evaluation_id, listener_id):
    import gradio as gr

    result = request(
        f"/v1/tts/evaluations/{evaluation_id}?"
        + urllib.parse.urlencode({"listener_id": listener_id})
    )
    progress = listening_progress(evaluation_id, listener_id)
    remaining = [s for s in result["samples"] if s["id"] not in progress["ratings"]]
    first = (remaining or result["samples"])[0]["id"] if result["samples"] else None
    return (
        result["samples"],
        gr.update(
            choices=[(f"样本 {i + 1}", s["id"]) for i, s in enumerate(result["samples"])],
            value=first,
        ),
        progress_text(progress),
    )


def select_sample(evaluation_id, sample_id, samples, listener_id):
    import gradio as gr

    sample = next((s for s in samples if s["id"] == sample_id), None)
    if not sample:
        return None, None, "", gr.update(), gr.update(), 3, "", "请先选择样本", ""
    progress = listening_progress(evaluation_id, listener_id)
    saved = progress["ratings"].get(sample_id, {})
    base = f"/v1/tts/evaluations/{evaluation_id}/samples/{sample_id}/audio"
    reference = audio_file(base + "?reference=true") if sample["similarity_applicable"] else None
    text = sample["text"]
    if sample["target_emotion"]:
        text += "\n目标情绪：" + dict((v, k) for k, v in EMOTIONS)[sample["target_emotion"]]
    return (
        audio_file(base),
        reference,
        text,
        gr.update(visible=sample["similarity_applicable"], value=saved.get("similarity") or 3),
        gr.update(visible=bool(sample["target_emotion"]), value=saved.get("emotion_match") or 3),
        saved.get("naturalness", 3),
        saved.get("comment") or "",
        "当前样本已保存；修改后请再次保存" if saved else "当前样本尚未评分；默认 3 分未提交",
        progress_text(progress),
    )


def score(
    evaluation_id, listener_id, sample_id, samples, naturalness, similarity, emotion, comment
):
    import gradio as gr

    sample = next((s for s in samples if s["id"] == sample_id), None)
    if not sample:
        raise gr.Error("请先选择并试听一个样本")
    request(
        f"/v1/tts/evaluations/{evaluation_id}/scores",
        {
            "listener_id": listener_id,
            "sample_id": sample_id,
            "naturalness": int(naturalness),
            "similarity": int(similarity) if sample["similarity_applicable"] else None,
            "emotion_match": int(emotion) if sample["target_emotion"] else None,
            "comment": comment,
        },
    )
    progress = listening_progress(evaluation_id, listener_id)
    return "评分已保存；同一评测人重复评分会更新原记录", progress_text(progress)


def score_and_next(*args):
    status, progress_label = score(*args)
    evaluation_id, listener_id, current, samples = args[:4]
    progress = listening_progress(evaluation_id, listener_id)
    remaining = [s["id"] for s in samples if s["id"] not in progress["ratings"]]
    if not remaining:
        return "全部样本已保存，可以查看统计结果", progress_label, gr.update(value=current)
    return status, progress_label, gr.update(value=remaining[0])


def export(evaluation_id):
    raw = request(f"/v1/tts/evaluations/{evaluation_id}/export", binary=True)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as handle:
        handle.write(raw)
        return handle.name


STATES = {
    "pending": "待处理",
    "validating": "检查中",
    "queued": "排队中",
    "running": "生成中",
    "succeeded": "已完成",
    "failed": "失败",
    "cancelled": "已取消",
}


def initial_voices():
    profiles, choices = load_voices()
    return profiles, choices, gr.update(open=not profiles)


def use_example(profiles):
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[4] / "data/tts-eval/public-references-3-5-10.json"
        ).read_text()
    )
    example = next(i for i in manifest if i["id"] == "D12_991_10s")
    profile = next(
        (
            p
            for p in profiles
            if p["asset"]["sha256"] == example["sha256"]
            and p["reference_text"] == example["reference_text"]
        ),
        None,
    )
    if not profile:
        raise gr.Error("公开示例尚未导入；请按使用说明准备并导入参考素材，或自行添加音色。")
    return (
        gr.update(value=profile["id"]),
        "清晨的阳光穿过窗帘，落在书桌旁的一杯热茶上。",
        "示例已填入；点击生成语音开始。",
    )


def generate_guarded(
    profiles,
    voice_id,
    text,
    model,
    emotion,
    punctuation_pause=False,
    gentle_rhythm=False,
    gpt_rhythm=False,
):
    yield "", "正在提交", None, None, gr.update(interactive=False)
    current = ""
    try:
        last = ("", "生成未完成", None, None)
        for row in synthesize(
            profiles, voice_id, text, model, emotion, punctuation_pause, gentle_rhythm, gpt_rhythm
        ):
            last = row
            current = row[0]
            yield *row, gr.update(interactive=False)
        # Keep concrete file values in the terminal event. Empty Audio/File updates
        # can clear the browser's pending media load in Gradio 5.
        yield *last, gr.update(interactive=True)
    except Exception as exc:
        yield (
            current,
            f"生成未完成：{exc}。已提交的任务可从历史恢复查看。",
            None,
            None,
            gr.update(interactive=True),
        )


def local_time(value):
    return (
        datetime.fromisoformat(value)
        .astimezone(ZoneInfo("Asia/Shanghai"))
        .strftime("%m-%d %H:%M:%S")
    )


def history_page(page):
    page = max(0, int(page))
    data = request(f"/v1/tts/jobs?limit=20&offset={page * 20}")
    if page and not data["items"]:
        page = max(0, (data["total"] - 1) // 20)
        data = request(f"/v1/tts/jobs?limit=20&offset={page * 20}")
    choices = [
        (
            f"{local_time(j['created_at'])} · "
            f"{STATES.get(j['state'], j['state'])} · "
            f"{j['model']} · {dict((v, k) for k, v in EMOTIONS).get(j['emotion'], j['emotion'])} · "
            f"{j['text'][:45]} · {j['id'][:8]}",
            j["id"],
        )
        for j in data["items"]
    ]
    return (
        page,
        gr.update(choices=choices, value=None),
        f"共 {data['total']} 条 · 第 {page + 1} 页（每页 20 条；北京时间）",
        None,
        None,
        None,
        "选择一条记录查看结果",
    )


def history_result(job_id):
    yield None, None, None, "正在加载"
    if not job_id:
        yield None, None, None, "选择一条记录查看结果"
        return
    try:
        job = poll(job_id)
        envelope = (
            request(f"/v1/audio/jobs/{job_id}/report") if job["state"] == "succeeded" else None
        )
        with tempfile.NamedTemporaryFile(
            prefix="shengke-report-", suffix=".json", mode="w", encoding="utf-8", delete=False
        ) as handle:
            json.dump({"job": job, "report": envelope}, handle, ensure_ascii=False, indent=2)
            exported = handle.name
        path = audio_file(f"/v1/tts/jobs/{job_id}/audio") if envelope else None
        yield (
            path,
            path,
            exported,
            job.get("error_message") or STATES.get(job["state"], job["state"]),
        )
    except Exception as exc:
        yield None, None, None, f"无法读取记录：{exc}；请刷新重试"


def restore_job(job_id, profiles):
    if not job_id:
        raise gr.Error("先选择一条历史记录")
    spec = poll(job_id)["parameters"]["tts"]
    if spec["model"] not in dict((v, k) for k, v in MODELS):
        raise gr.Error("该记录是固定音色评测基线，请从专业评测脚本复现。")
    voice_id = spec.get("voice_id")
    if not any(p["id"] == voice_id for p in profiles):
        raise gr.Error("原参考音色不可用；请刷新音色或重新添加，不会自动替换。")
    profile = spec.get("inference_profile", "baseline")
    if profile not in {"baseline", "gpt_pause_v3", "gpt_rhythm_v6", "cosy_rhythm_v5"}:
        raise gr.Error("此记录使用历史实验配置，请用原始报告复现；不会改为默认配置。")
    is_gpt = spec["model"] == "gpt-sovits-v2"
    return (
        gr.update(value=voice_id),
        spec["text"],
        spec["model"],
        gr.update(choices=EMOTIONS[:1] if is_gpt else EMOTIONS, value=spec["emotion"]),
        gr.update(
            value=profile == "gpt_pause_v3", visible=is_gpt, interactive=profile != "gpt_rhythm_v6"
        ),
        gr.update(
            value=profile == "cosy_rhythm_v5", visible=not is_gpt and spec["emotion"] == "happy"
        ),
        gr.update(value=profile == "gpt_rhythm_v6", visible=is_gpt),
        gr.update(selected="synthesis"),
        "已载入历史参数；点击生成语音将创建新任务。",
        None,
        None,
        "",
        gr.update(open=True),
    )


def build_app():
    import gradio as gr

    with gr.Blocks(title="声刻 · 语音克隆与情感 TTS", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            f"# 声刻 {RELEASE['version']} · {RELEASE['edition']}\n"
            "用一小段声音，讲述新的故事。普通话短句 · 参考人声 3–10 秒"
        )
        profiles = gr.State([])
        with gr.Tabs() as pages:
            with gr.Tab("合成工作台", id="synthesis"):
                with gr.Accordion("添加参考音色", open=False) as add_voice:
                    reference = gr.Audio(
                        type="filepath", sources=["upload"], label="参考人声 WAV / FLAC"
                    )
                    with gr.Row():
                        name = gr.Textbox(label="音色名称")
                        transcript = gr.Textbox(label="参考音频中说出的准确文本")
                    with gr.Row():
                        source = gr.Textbox(label="素材来源链接")
                        license_text = gr.Textbox(label="素材使用许可")
                    upload_button = gr.Button("检查并保存音色")
                    qc = gr.JSON(label="参考音频检查")
                with gr.Row():
                    voices = gr.Dropdown(label="选择音色", choices=[])
                    refresh_voices = gr.Button("刷新音色")
                text = gr.Textbox(
                    label="想让它说什么",
                    lines=3,
                    max_lines=6,
                    value="你好，欢迎来到声刻。让每一句话，都有自己的温度。",
                )
                example_button = gr.Button("填入公开示例")
                emotion = gr.Dropdown(
                    EMOTIONS,
                    value="neutral",
                    label="情绪（CosyVoice 试用）",
                    info="GPT-SoVITS 仅支持中性；情绪强度与自然度仍有局限。",
                )
                with gr.Accordion("高级设置 · 模型与节奏", open=False) as advanced:
                    model = gr.Dropdown(MODELS, value="cosyvoice2", label="合成引擎")
                    punctuation_pause = gr.Checkbox(
                        label="按标点停顿（试用）",
                        value=False,
                        visible=False,
                        info="适合较长的普通话分句；每段至少八个汉字时启用。短分句保持整句朗读。",
                    )
                    gentle_rhythm = gr.Checkbox(
                        label="舒缓节奏（试用）",
                        value=False,
                        visible=False,
                        info="开心模式的较慢版本（0.85 倍速），仅经少量试听；开心程度仍需核对。",
                    )
                    gpt_rhythm = gr.Checkbox(
                        label="舒缓语速与停顿（试用）",
                        value=False,
                        visible=False,
                        info=(
                            "仅中文及标点；0.9 倍速。按实际音频定位停顿，"
                            "定位不可靠时提示跳过；生成稍慢。"
                        ),
                    )
                    gpt_rhythm.input(
                        lambda enabled: gr.update(value=False, interactive=not enabled),
                        gpt_rhythm,
                        punctuation_pause,
                    )
                generate = gr.Button("生成语音", variant="primary")
                status = gr.Textbox(label="状态", interactive=False)
                audio = gr.Audio(label="生成结果", type="filepath")
                download = gr.File(label="下载 WAV")
                with gr.Accordion("任务详情", open=False):
                    job_id = gr.Textbox(label="作业编号", interactive=False)
                    with gr.Row():
                        cancel = gr.Button("取消任务")
                        refresh = gr.Button("刷新结果")
                upload_button.click(
                    upload,
                    [reference, name, transcript, source, license_text],
                    [profiles, voices, qc],
                )
                refresh_voices.click(load_voices, outputs=[profiles, voices])
                model.input(
                    lambda m: (
                        gr.update(
                            choices=EMOTIONS if m == "cosyvoice2" else EMOTIONS[:1], value="neutral"
                        ),
                        gr.update(visible=m == "gpt-sovits-v2", value=False, interactive=True),
                        gr.update(visible=False, value=False),
                        gr.update(visible=m == "gpt-sovits-v2", value=False),
                    ),
                    model,
                    [emotion, punctuation_pause, gentle_rhythm, gpt_rhythm],
                )
                emotion.input(
                    lambda m, e: gr.update(visible=m == "cosyvoice2" and e == "happy", value=False),
                    [model, emotion],
                    gentle_rhythm,
                )
                generate.click(
                    generate_guarded,
                    [
                        profiles,
                        voices,
                        text,
                        model,
                        emotion,
                        punctuation_pause,
                        gentle_rhythm,
                        gpt_rhythm,
                    ],
                    [job_id, status, audio, download, generate],
                    trigger_mode="once",
                )
                example_button.click(use_example, profiles, [voices, text, status])
                cancel.click(cancel_jobs, job_id, status, queue=False)
                refresh.click(refresh_result, job_id, [status, audio, download], queue=False)
            with gr.Tab("历史与下载", id="history"):
                history_index = gr.State(0)
                history_label = gr.Markdown("点击刷新加载历史")
                with gr.Row():
                    prev = gr.Button("上一页")
                    history_refresh = gr.Button("刷新历史")
                    nxt = gr.Button("下一页")
                history_choice = gr.Dropdown(
                    label="生成记录（最新在前）", choices=[], interactive=True
                )
                history_status = gr.Textbox(label="记录状态", interactive=False)
                history_audio = gr.Audio(label="历史音频", type="filepath")
                with gr.Row():
                    history_wav = gr.File(label="下载历史 WAV")
                    history_json = gr.File(label="下载任务报告 JSON")
                history_restore = gr.Button("载入参数，再次生成")
                history_outputs = [
                    history_index,
                    history_choice,
                    history_label,
                    history_audio,
                    history_wav,
                    history_json,
                    history_status,
                ]
                history_refresh.click(history_page, history_index, history_outputs)
                prev.click(lambda p: history_page(p - 1), history_index, history_outputs)
                nxt.click(lambda p: history_page(p + 1), history_index, history_outputs)
                history_choice.change(
                    history_result,
                    history_choice,
                    [history_audio, history_wav, history_json, history_status],
                )
                history_restore.click(
                    restore_job,
                    [history_choice, profiles],
                    [
                        voices,
                        text,
                        model,
                        emotion,
                        punctuation_pause,
                        gentle_rhythm,
                        gpt_rhythm,
                        pages,
                        status,
                        audio,
                        download,
                        job_id,
                        advanced,
                    ],
                )
            with gr.Tab("专业 · 对比与评测", id="professional"):
                with gr.Tabs() as professional_pages:
                    with gr.Tab("对比试听", id="compare"):
                        gr.Markdown("使用工作台当前选中的音色与文本。左侧为 CosyVoice2 中性版本。")
                        mode = gr.Radio(
                            ["双引擎对比", "同音色情绪对比"],
                            value="同音色情绪对比",
                            label="对比方式",
                        )
                        cmp_emotion = gr.Dropdown(EMOTIONS[1:], value="happy", label="对照情绪")
                        cmp_button = gr.Button("生成对照")
                        cmp_status = gr.Textbox(label="状态")
                        with gr.Row():
                            left = gr.Audio(label="版本 A", type="filepath")
                            right = gr.Audio(label="版本 B", type="filepath")
                        cmp_ids = gr.Textbox(label="作业编号，可用于创建听评", lines=2)
                        cmp_cancel = gr.Button("取消对照任务")
                        cmp_button.click(
                            compare,
                            [profiles, voices, text, mode, cmp_emotion],
                            [cmp_ids, cmp_status, left, right],
                        )
                        cmp_cancel.click(cancel_jobs, cmp_ids, cmp_status, queue=False)
                    with gr.Tab("核对参考音频", id="references"):
                        gr.Markdown(
                            "## 六份参考音频核对\n请逐一播放，检查文字是否一致、开头结尾是否截断、"
                            "是否为清晰单人声音。核对后在对话中告知通过的编号，或指出错字和截断位置。"
                            "此页面不会自动把样本标记为通过。"
                        )
                        reference_manifest = json.loads(
                            (
                                Path(__file__).resolve().parents[4]
                                / "data/tts-eval/public-references-3-5-10.json"
                            ).read_text()
                        )
                        review_players = []
                        for item in reference_manifest:
                            gr.Markdown(f"### {item['id']} · {item['duration_s']} 秒")
                            gr.Textbox(
                                label="请核对这段文字",
                                value=item["reference_text"],
                                interactive=False,
                            )
                            review_players.append(gr.Audio(label="参考音频", interactive=False))
                        reload_review = gr.Button("加载 / 刷新六份参考音频")

                        def load_reference_audio():
                            available = request("/v1/tts/voices")
                            by_hash = {p["asset"]["sha256"]: p for p in available}
                            paths = []
                            for item in reference_manifest:
                                profile = by_hash.get(item["sha256"])
                                if not profile:
                                    raise gr.Error(f"参考音频 {item['id']} 尚未导入")
                                paths.append(audio_file(f"/v1/tts/voices/{profile['id']}/audio"))
                            return paths

                        reload_review.click(load_reference_audio, outputs=review_players)
                    with gr.Tab("试听评分 / 评测中心", id="evaluation"):
                        gr.Markdown(
                            "自然度 1=很不自然，5=非常自然。"
                            "单人评分只作为预评测，不能代表多人 MOS。"
                        )
                        with gr.Accordion("模型运行状态", open=False):
                            model_status = gr.JSON(label="模型状态")
                            gr.Button("检查模型").click(
                                lambda: request("/v1/tts/models"), outputs=model_status
                            )
                        with gr.Accordion("创建听评", open=False):
                            eval_name = gr.Textbox(label="评测名称", value="声刻第一轮听评")
                            eval_jobs = gr.Textbox(label="已完成作业编号（每行一个）", lines=4)
                            eval_create = gr.Button("创建")
                        evaluation_id = gr.Textbox(label="评测编号")
                        listener_id = gr.Textbox(
                            label="匿名评测人编号（英文、数字、下划线）", value="listener_01"
                        )
                        begin = gr.Button("加载盲听样本")
                        progress_label = gr.Markdown("加载后显示评分进度")
                        samples = gr.State([])
                        sample_id = gr.Dropdown(label="样本", choices=[])
                        prompt = gr.Textbox(label="目标文本与情绪", interactive=False)
                        with gr.Row():
                            blind_audio = gr.Audio(label="待评语音")
                            original_audio = gr.Audio(label="参考人声（用于音色相似度）")
                        naturalness = gr.Slider(1, 5, value=3, step=1, label="自然度")
                        similarity = gr.Slider(1, 5, value=3, step=1, label="音色相似度")
                        emotion_match = gr.Slider(1, 5, value=3, step=1, label="情绪匹配度")
                        comment = gr.Textbox(label="听感备注")
                        with gr.Row():
                            save = gr.Button("保存评分")
                            save_next = gr.Button("保存并继续下一个", variant="primary")
                        score_status = gr.Textbox(label="评分状态")
                        with gr.Accordion("评分结束后查看结果", open=False):
                            result = gr.JSON(label="统计结果")
                            gr.Button("统计").click(
                                lambda e: request(f"/v1/tts/evaluations/{e}/report"),
                                evaluation_id,
                                result,
                            )
                            exported = gr.File(label="原始评分")
                            gr.Button("导出 CSV").click(export, evaluation_id, exported)
                        eval_create.click(create_evaluation, [eval_name, eval_jobs], evaluation_id)
                        begin.click(
                            start_listening,
                            [evaluation_id, listener_id],
                            [samples, sample_id, progress_label],
                        )
                        sample_id.change(
                            select_sample,
                            [evaluation_id, sample_id, samples, listener_id],
                            [
                                blind_audio,
                                original_audio,
                                prompt,
                                similarity,
                                emotion_match,
                                naturalness,
                                comment,
                                score_status,
                                progress_label,
                            ],
                        )
                        save.click(
                            score,
                            [
                                evaluation_id,
                                listener_id,
                                sample_id,
                                samples,
                                naturalness,
                                similarity,
                                emotion_match,
                                comment,
                            ],
                            [score_status, progress_label],
                        )
                        save_next.click(
                            score_and_next,
                            [
                                evaluation_id,
                                listener_id,
                                sample_id,
                                samples,
                                naturalness,
                                similarity,
                                emotion_match,
                                comment,
                            ],
                            [score_status, progress_label, sample_id],
                        )

        def open_page(req: gr.Request):
            view = req.query_params.get("view", "synthesis")
            professional = view in {"compare", "references", "evaluation"}
            if view not in {
                "synthesis",
                "history",
                "professional",
                "compare",
                "references",
                "evaluation",
            }:
                view = "synthesis"
            return (
                gr.update(selected="professional" if professional else view),
                req.query_params.get("evaluation", ""),
                gr.update(selected=view if professional else "compare"),
            )

        demo.load(open_page, outputs=[pages, evaluation_id, professional_pages])
        demo.load(initial_voices, outputs=[profiles, voices, add_voice])
        demo.load(lambda: history_page(0), outputs=history_outputs)
    return demo


def main():
    build_app().queue(default_concurrency_limit=2).launch(
        server_name=os.getenv("SHENGKE_UI_HOST", "127.0.0.1"),
        server_port=int(os.getenv("SHENGKE_UI_PORT", "7860")),
        share=False,
    )


if __name__ == "__main__":
    main()
