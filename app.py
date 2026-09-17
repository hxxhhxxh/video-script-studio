# -*- coding: utf-8 -*-
"""视频链接 → 改写剧本 + 人物画像 + 分镜提示词  本地工作台

启动：python app.py        然后浏览器打开 http://127.0.0.1:8790
"""

import json
import os
import pathlib
import sys
import threading
import time
import uuid

# 有些环境会开 PYTHONSAFEPATH，导致脚本所在目录不在 sys.path 里，显式补上
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import pipeline
import prompts

BASE_DIR = pathlib.Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
PORT = int(os.environ.get("PORT", "8790"))

app = FastAPI(title="视频改写剧本工作台")

# 对外暴露时的口令闸：设了 VSS_ACCESS_TOKEN 才生效。
# 不带口令的浏览器访问会被拦；带对了会种一个 cookie，后续页面正常用。
ACCESS_TOKEN = os.environ.get("VSS_ACCESS_TOKEN", "").strip()


@app.middleware("http")
async def access_gate(request: Request, call_next):
    if not ACCESS_TOKEN:
        return await call_next(request)
    if request.method == "OPTIONS":          # 预检请求交给 CORS 中间件处理
        return await call_next(request)
    tok = (request.query_params.get("token")
           or request.headers.get("x-auth-token")
           or request.cookies.get("vss_token"))
    if tok == ACCESS_TOKEN:
        resp = await call_next(request)
        if request.query_params.get("token"):
            resp.set_cookie("vss_token", ACCESS_TOKEN, httponly=True,
                            max_age=30 * 86400, samesite="lax")
        return resp
    if request.url.path.startswith("/api/"):
        return JSONResponse({"error": "需要访问口令"}, status_code=401)
    return HTMLResponse(
        "<meta charset='utf-8'><body style=\"font:16px/1.8 sans-serif;padding:40px\">"
        "<h2>需要访问口令</h2><p>请在网址后面加上 <code>?token=你的口令</code> 再回车。"
        "<p>口令写在本机 <code>video-script-studio/.access-token.txt</code>。</p></body>",
        status_code=401)


# 跨域：让 GitHub Pages 上的前端能直接调这个后端。
# 必须加在 access_gate 之后，这样 CORS 是最外层，预检请求不会被口令闸拦住。
# 可用 VSS_CORS_ORIGINS 覆盖（逗号分隔）。
CORS_ORIGINS = [o.strip() for o in os.environ.get(
    "VSS_CORS_ORIGINS",
    "https://hxxhhxxh.github.io,http://127.0.0.1:8790,http://localhost:8790"
).split(",") if o.strip()]

from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


class AnalyzeReq(BaseModel):
    url: str
    target_seconds: int = 60
    style: str = ""
    aspect: str = "9:16 竖屏"
    notes: str = ""
    whisper: str = ""


def _job_log(job_id: str, msg: str):
    with LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        job["log"].append({"t": time.strftime("%H:%M:%S"), "msg": msg})
        job["log"] = job["log"][-200:]


def _run_job(job_id: str, req: AnalyzeReq):
    with LOCK:
        JOBS[job_id]["status"] = "running"
    try:
        result = pipeline.analyze(
            req.url,
            {"target_seconds": req.target_seconds, "style": req.style,
             "aspect": req.aspect, "notes": req.notes, "whisper": req.whisper},
            lambda m: _job_log(job_id, m),
        )
        with LOCK:
            JOBS[job_id].update({"status": "done", "result": result, "finished": time.time()})
    except Exception as e:  # noqa: BLE001
        _job_log(job_id, f"失败：{e}")
        with LOCK:
            JOBS[job_id].update({"status": "error", "error": str(e), "finished": time.time()})


@app.get("/api/health")
def health():
    return pipeline.check_health()


@app.get("/api/prompts")
def get_prompts():
    return prompts.recipes_preview()


@app.post("/api/analyze")
def analyze(req: AnalyzeReq):
    if not req.url.strip():
        raise HTTPException(400, "请填写视频链接")
    job_id = uuid.uuid4().hex[:12]
    with LOCK:
        JOBS[job_id] = {"id": job_id, "status": "queued", "url": req.url,
                        "created": time.time(), "log": [], "result": None}
    threading.Thread(target=_run_job, args=(job_id, req), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    out = {k: v for k, v in job.items() if k != "result"}
    out["has_result"] = bool(job.get("result"))
    return out


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str):
    job = JOBS.get(job_id)
    if not job or not job.get("result"):
        raise HTTPException(404, "结果还没生成")
    return job["result"]


@app.get("/api/jobs")
def list_jobs():
    items = sorted(JOBS.values(), key=lambda j: j["created"], reverse=True)[:30]
    return [{"id": j["id"], "status": j["status"], "url": j["url"],
             "created": j["created"], "error": j.get("error")} for j in items]


@app.get("/api/jobs/{job_id}/export/{fmt}")
def export(job_id: str, fmt: str):
    job = JOBS.get(job_id)
    if not job or not job.get("result"):
        raise HTTPException(404, "结果还没生成")
    r = job["result"]
    title = (r["script"].get("title") or "storyboard").replace("/", "_")[:40]
    work = pathlib.Path(r.get("work_dir", BASE_DIR))
    if fmt == "json":
        path = work / "result.json"
        if path.exists():
            return FileResponse(path, filename=f"{title}.json", media_type="application/json")
    if fmt == "md":
        return PlainTextResponse(pipeline.to_markdown(r), media_type="text/markdown; charset=utf-8",
                                 headers={"Content-Disposition": f'attachment; filename="{job_id}.md"'})
    if fmt == "csv":
        return PlainTextResponse("\ufeff" + pipeline.to_csv(r), media_type="text/csv; charset=utf-8",
                                 headers={"Content-Disposition": f'attachment; filename="{job_id}.csv"'})
    raise HTTPException(400, "只支持 json / md / csv")


@app.get("/api/jobs/{job_id}/transcript")
def transcript(job_id: str):
    job = JOBS.get(job_id)
    if not job or not job.get("result"):
        raise HTTPException(404, "结果还没生成")
    return JSONResponse(job["result"]["transcript"])


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    print(f"视频改写剧本工作台 → http://{host}:{PORT}")
    uvicorn.run(app, host=host, port=PORT, log_level="info")
