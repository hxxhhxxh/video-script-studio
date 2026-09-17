# -*- coding: utf-8 -*-
"""流水线：链接 → 音轨 → 转录 → 节拍 → 剧本/人物/分镜

每一步都是独立函数，可以单独替换：
  download_media()  换下载器（默认 yt-dlp，支持绝大多数站点）
  transcribe()      换语音识别（默认 faster-whisper 本地跑）
  llm_json()        换大模型（默认本地 Ollama，也可指向任意 OpenAI 兼容接口）
"""

import os

# ⚠ 必须在 import huggingface_hub（或其下游 faster_whisper）之前设置：
#   1) 国内直连 huggingface.co 会失败 —— 走 hf-mirror 镜像
#   2) huggingface_hub 1.x 默认启用 Xet 存储后端，镜像不代理它（必然 401）—— 必须关掉
#   3) 顺手静音 Windows 无开发者模式导致的符号链接警告
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import csv  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import re  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402

import prompts  # noqa: E402

BASE_DIR = pathlib.Path(__file__).parent
RUNS_DIR = BASE_DIR / "runs"
MODELS_DIR = BASE_DIR / "models"

# ── 可调参数（都能用环境变量覆盖）──────────────────────────────
OLLAMA_HOST = os.environ.get("VSS_OLLAMA", "http://127.0.0.1:11434").rstrip("/")
LLM_MODEL = os.environ.get("VSS_LLM_MODEL", "deepseek-r1:8b")
LLM_NUM_CTX = int(os.environ.get("VSS_NUM_CTX", "16384"))
LLM_NUM_PREDICT = int(os.environ.get("VSS_NUM_PREDICT", "3072"))
WHISPER_SIZE = os.environ.get("VSS_WHISPER", "small")
CHUNK_CHARS = int(os.environ.get("VSS_CHUNK_CHARS", "1200"))

# 指向云端时不走 Ollama：设了 VSS_LLM_BASE_URL 就切到 OpenAI 兼容协议
CLOUD_BASE = os.environ.get("VSS_LLM_BASE_URL", "").rstrip("/")
CLOUD_KEY = os.environ.get("VSS_LLM_API_KEY", "")
CLOUD_MODEL = os.environ.get("VSS_LLM_CLOUD_MODEL", "")


def _post_json(url: str, payload: dict, timeout: int = 1800, headers: dict | None = None) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json(url: str, timeout: int = 30) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ─────────────────────────────────────────────────────────────
# 健康检查
# ─────────────────────────────────────────────────────────────

def check_health() -> dict:
    out = {"llm_mode": "cloud" if CLOUD_BASE else "ollama", "ok": True, "items": []}

    try:
        tags = _get_json(f"{OLLAMA_HOST}/api/tags", timeout=8)
        names = [m["name"] for m in tags.get("models", [])]
        if CLOUD_BASE:
            out["items"].append({"name": "云端模型", "ok": True, "detail": f"{CLOUD_BASE} / {CLOUD_MODEL or '默认'}"})
        else:
            hit = any(n.startswith(LLM_MODEL.split(":")[0]) for n in names)
            out["items"].append({
                "name": f"本地模型 {LLM_MODEL}", "ok": hit,
                "detail": "已就绪" if hit else f"未找到，已装: {', '.join(names) or '无'}",
            })
            out["ok"] = out["ok"] and hit
    except Exception as e:  # noqa: BLE001
        out["ok"] = False if not CLOUD_BASE else out["ok"]
        out["items"].append({"name": "本地模型", "ok": False, "detail": f"{OLLAMA_HOST} 连不上: {e}"})

    try:
        import yt_dlp
        out["items"].append({"name": "下载器 yt-dlp", "ok": True, "detail": yt_dlp.version.__version__})
    except Exception as e:  # noqa: BLE001
        out["ok"] = False
        out["items"].append({"name": "下载器 yt-dlp", "ok": False, "detail": str(e)})

    try:
        import faster_whisper
        import av  # noqa: F401
        out["items"].append({"name": "语音识别 faster-whisper", "ok": True,
                             "detail": f"{faster_whisper.__version__} / 模型 {WHISPER_SIZE}"})
    except Exception as e:  # noqa: BLE001
        out["ok"] = False
        out["items"].append({"name": "语音识别 faster-whisper", "ok": False, "detail": str(e)})

    return out


# ─────────────────────────────────────────────────────────────
# 1. 下载
# ─────────────────────────────────────────────────────────────

def download_media(url: str, workdir: pathlib.Path, log) -> dict:
    import yt_dlp

    workdir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(workdir / "source.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "retries": 3,
        "socket_timeout": 30,
    }
    log("正在解析链接…")
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    files = sorted(workdir.glob("source.*"))
    if not files:
        raise RuntimeError("下载完成但没找到音轨文件")
    media = files[0]
    meta = {
        "path": str(media),
        "title": info.get("title") or "未命名",
        "duration": info.get("duration"),
        "uploader": info.get("uploader") or info.get("channel") or "",
        "platform": info.get("extractor_key") or info.get("extractor") or "",
        "webpage_url": info.get("webpage_url") or url,
        "description": (info.get("description") or "")[:800],
    }
    log(f"下载完成：{media.name}（{media.stat().st_size/1048576:.1f} MB）")
    return meta


# ─────────────────────────────────────────────────────────────
# 2. 转录
# ─────────────────────────────────────────────────────────────

def transcribe(media_path, model_size: str, log) -> dict:
    # 镜像与 Xet 开关在模块顶部已设好（必须在 huggingface_hub 导入前）
    from faster_whisper import WhisperModel

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    log(f"加载语音模型 {model_size}（首次会自动下载，约 500MB，之后走缓存）…")
    t0 = time.time()
    model = WhisperModel(model_size, device="cpu", compute_type="int8",
                         download_root=str(MODELS_DIR))
    log(f"模型就绪（{time.time()-t0:.1f}s），开始识别…")

    segments, info = model.transcribe(
        media_path, language=None, vad_filter=True,
        beam_size=1, condition_on_previous_text=False,
    )

    total = float(info.duration or 0)
    out, last = [], 0.0
    for seg in segments:
        out.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": seg.text.strip()})
        last = seg.end
        if total and seg.end - getattr(transcribe, "_last_log", 0) > 30:
            transcribe._last_log = seg.end
            log(f"识别进度 {seg.end/total*100:.0f}%（{seg.end:.0f}s / {total:.0f}s）")

    joiner = "" if str(info.language).startswith("zh") else " "
    text = joiner.join(s["text"] for s in out if s["text"]).strip()
    return {"text": text, "segments": out, "language": info.language,
            "duration": round(total or last, 1), "model": model_size}


# ─────────────────────────────────────────────────────────────
# 3. 大模型调用
# ─────────────────────────────────────────────────────────────

def _strip_fence(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    if not s.startswith("{"):
        i, j = s.find("{"), s.rfind("}")
        if i >= 0 and j > i:
            s = s[i:j + 1]
    return s.strip()


def _llm_once(system: str, user: str, json_mode: bool, max_tokens: int | None = None) -> str:
    limit = max_tokens or LLM_NUM_PREDICT
    if CLOUD_BASE:
        payload = {
            "model": CLOUD_MODEL or LLM_MODEL,
            "messages": ([{"role": "system", "content": system}] if system else []) +
                        [{"role": "user", "content": user}],
            "temperature": 0.6, "max_tokens": limit,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        r = _post_json(f"{CLOUD_BASE}/chat/completions", payload,
                       headers={"Authorization": f"Bearer {CLOUD_KEY}"})
        return r["choices"][0]["message"]["content"] or ""

    payload = {
        "model": LLM_MODEL,
        "messages": ([{"role": "system", "content": system}] if system else []) +
                    [{"role": "user", "content": user}],
        "stream": False,
        "options": {"num_ctx": LLM_NUM_CTX, "num_predict": limit,
                    "temperature": 0.6, "top_p": 0.95},
        "keep_alive": "15m",
    }
    if json_mode:
        payload["format"] = "json"
    r = _post_json(f"{OLLAMA_HOST}/api/chat", payload)
    msg = r.get("message") or {}
    return msg.get("content") or ""


def llm_json(user: str, system: str = "", label: str = "模型生成", log=print,
             max_tokens: int | None = None) -> dict:
    log(f"{label}…（本地模型出字约 13 字/秒，它习惯先思考再写，请耐心；下面会报心跳）")
    box: dict = {}

    def _work():
        try:
            box["raw"] = _llm_once(system, user, json_mode=True, max_tokens=max_tokens)
        except Exception as e:  # noqa: BLE001
            box["err"] = e

    th = threading.Thread(target=_work, daemon=True)
    t0 = time.time()
    th.start()
    while th.is_alive():
        th.join(timeout=20)
        if th.is_alive():
            log(f"[心跳] {label} 生成中… 已用 {time.time()-t0:.0f} 秒")
    if "err" in box:
        raise box["err"]
    raw = box.get("raw", "")
    log(f"{label} 首次输出完成（{time.time()-t0:.0f} 秒，{len(raw)} 字），解析中…")

    for attempt in (1, 2):
        try:
            return json.loads(_strip_fence(raw))
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                raise RuntimeError(f"模型输出不是合法 JSON（{e}）。原始输出前 300 字：{raw[:300]}")
            log(f"JSON 解析失败，让模型重出一次（{e}）")
            raw = _llm_once("", prompts.REPAIR_PROMPT.replace("{error}", str(e))
                            .replace("{schema_hint}", prompts.FINAL_SCHEMA[:600]), json_mode=True,
                            max_tokens=max_tokens)
    return {}


# ─────────────────────────────────────────────────────────────
# 4. 分块与总装
# ─────────────────────────────────────────────────────────────

def split_chunks(segments: list, max_chars: int = CHUNK_CHARS) -> list:
    chunks, cur, size = [], [], 0
    for seg in segments:
        if not seg["text"]:
            continue
        if size + len(seg["text"]) > max_chars and cur:
            chunks.append(cur)
            cur, size = [cur[-1]], len(cur[-1]["text"])   # 重叠一句，避免断章
        cur.append(seg)
        size += len(seg["text"])
    if cur:
        chunks.append(cur)
    return chunks


def analyze(url: str, opts: dict, log) -> dict:
    """完整流水线。log 是回调，用来把进度推给前端。"""
    target = int(opts.get("target_seconds") or 60)
    style_hint = (opts.get("style") or "").strip()
    aspect = opts.get("aspect") or "9:16 竖屏"
    extra = (opts.get("notes") or "").strip()
    whisper_size = opts.get("whisper") or WHISPER_SIZE

    job_dir = RUNS_DIR / f"{int(time.time())}-{re.sub(r'[^a-zA-Z0-9]', '', url)[-12:]}"
    job_dir.mkdir(parents=True, exist_ok=True)

    log("① 解析链接并下载音轨")
    media = download_media(url, job_dir, log)
    log(f"标题：{media['title']}｜平台：{media['platform']}｜时长：{media['duration']}s")

    log("② 语音转文字")
    asr = transcribe(media["path"], whisper_size, log)
    if not asr["text"].strip():
        raise RuntimeError("没识别到任何语音内容（可能是纯音乐/无人声视频）")
    log(f"识别完成：{len(asr['text'])} 字，语言 {asr['language']}")

    log("③ 逐段提炼节拍与人物线索")
    chunks = split_chunks(asr["segments"])
    beats_all = []
    for i, ch in enumerate(chunks, 1):
        text = "".join(s["text"] for s in ch)
        log(f"提炼第 {i}/{len(chunks)} 块（{len(text)} 字）")
        data = llm_json(prompts.beats_prompt(text, i, len(chunks)),
                        system="你是严谨的短视频编剧分析助手，只做提炼，不做创作。",
                        label=f"提炼 {i}/{len(chunks)}", log=log)
        beats_all.extend(data.get("beats") or [])

    log(f"④ 写剧本（共 {len(beats_all)} 个节拍）")
    merged = json.dumps({"beats": beats_all}, ensure_ascii=False)
    sc = llm_json(
        prompts.script_prompt(merged, target, style_hint, aspect, extra),
        system="你是资深短视频导演，输出严格的 JSON。", label="剧本", log=log,
        max_tokens=int(os.environ.get("VSS_NUM_PREDICT_FINAL", "4096")))

    log("⑤ 整理人物画像")
    ch = llm_json(
        prompts.characters_prompt(merged, style_hint),
        system="你是影视造型指导，输出严格的 JSON。", label="人物画像", log=log,
        max_tokens=int(os.environ.get("VSS_NUM_PREDICT_CHARS", "3072")))

    log("⑥ 拆分镜与生成提示词")
    sh = llm_json(
        prompts.shots_prompt(merged,
                             json.dumps(sc.get("scenes") or [], ensure_ascii=False),
                             json.dumps(ch.get("characters") or [], ensure_ascii=False),
                             target, aspect),
        system="你是分镜师与 AI 视频提示词工程师，输出严格的 JSON。", label="分镜与提示词", log=log,
        max_tokens=int(os.environ.get("VSS_NUM_PREDICT_SHOTS", "4096")))

    result = {
        "source": media,
        "transcript": asr,
        "target_seconds": target,
        "options": {"style": style_hint, "aspect": aspect, "notes": extra, "whisper": whisper_size},
        "llm": {"mode": "cloud" if CLOUD_BASE else "ollama",
                "model": (CLOUD_MODEL or LLM_MODEL) if CLOUD_BASE else LLM_MODEL},
        "script": {"title": sc.get("title", ""), "logline": sc.get("logline", ""),
                   "style": sc.get("style", ""), "scenes": sc.get("scenes") or []},
        "characters": ch.get("characters") or [],
        "shots": sh.get("shots") or [],
    }

    total = sum(int(s.get("duration") or 0) for s in result["shots"])
    result["storyboard_total_seconds"] = total
    (job_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    (job_dir / "result.md").write_text(to_markdown(result), "utf-8")
    (job_dir / "storyboard.csv").write_text(to_csv(result), "utf-8-sig")
    result["work_dir"] = str(job_dir)
    log(f"完成！分镜共 {len(result['shots'])} 个，合计 {total} 秒")
    return result


# ─────────────────────────────────────────────────────────────
# 5. 导出
# ─────────────────────────────────────────────────────────────

def to_markdown(r: dict) -> str:
    L = []
    src = r["source"]
    L.append(f"# {r['script'].get('title') or src['title']}")
    L.append("")
    L.append(f"- 来源：{src.get('webpage_url')}")
    L.append(f"- 平台 / 作者：{src.get('platform')} / {src.get('uploader')}")
    L.append(f"- 原片时长：{src.get('duration')}s｜分镜合计：{r.get('storyboard_total_seconds')}s")
    L.append(f"- 生成模型：{r['llm']['mode']} / {r['llm']['model']}")
    L.append("")
    L.append(f"**一句话故事线**：{r['script'].get('logline', '')}")
    L.append("")
    L.append(f"**影像风格**：{r['script'].get('style', '')}")
    L.append("")
    L.append("## 人物画像")
    for c in r.get("characters") or []:
        L.append(f"### {c.get('name')}（{c.get('role')}）")
        L.append(f"- 年龄：{c.get('age')}｜性别：{c.get('gender')}｜体型：{c.get('body')}")
        L.append(f"- 面部发型：{c.get('face')}")
        L.append(f"- 穿戴：{c.get('outfit')}")
        L.append(f"- 配饰：{c.get('accessories')}")
        L.append(f"- 气质：{c.get('vibe')}")
        L.append(f"- 锚定提示词（中文）：{c.get('anchor_cn')}")
        L.append(f"- 锚定提示词（英文）：{c.get('anchor_en')}")
        L.append("")
    L.append("## 分镜表")
    L.append("")
    L.append("| # | 时长 | 景别 | 运镜 | 画面 | 提示词（中文） | 负向 | 字幕 | 音效 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for s in r.get("shots") or []:
        cell = lambda k: str(s.get(k, "") or "").replace("|", "／").replace("\n", " ")
        L.append(f"| {s.get('no')} | {s.get('duration')}s | {cell('shot_type')} | {cell('camera')} | "
                 f"{cell('visual')} | {cell('prompt_cn')} | {cell('negative')} | {cell('subtitle')} | {cell('sfx')} |")
    L.append("")
    L.append("## 分场剧本")
    for sc in r["script"].get("scenes") or []:
        L.append(f"### 第 {sc.get('no')} 场 · {sc.get('heading')}（{sc.get('duration')}s）")
        L.append(f"{sc.get('summary', '')}")
        L.append("")
        L.append(f"> {sc.get('voiceover', '')}")
        L.append("")
    L.append("## 原视频转录（供核对）")
    L.append("")
    L.append(r["transcript"]["text"])
    return "\n".join(L)


CSV_COLS = ["no", "scene_no", "duration", "shot_type", "camera", "visual", "prompt_cn",
            "prompt_en", "negative", "voiceover", "subtitle", "sfx", "bgm"]


def to_csv(r: dict) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLS, extrasaction="ignore")
    w.writeheader()
    for s in r.get("shots") or []:
        w.writerow({k: s.get(k, "") for k in CSV_COLS})
    return buf.getvalue()
