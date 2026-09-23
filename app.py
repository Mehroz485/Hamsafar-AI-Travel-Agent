import json
import traceback
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from backend import (
    run_travel_agent,
    resume_travel_agent,
    stream_travel_agent,
    stream_resume_travel_agent,
)

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Hamsafar",
    description="LangGraph Multi-Agent Travel Planner with Supervisor, Guardrails, Human-in-the-Loop and FastAPI Frontend",
    version="2.1.0",
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class TravelRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ApprovalRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    approved: bool
    feedback: str = ""


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})




def _sse(generator):
    """Wrap an event generator as a Server-Sent Events response."""

    def events():
        try:
            for event in generator:
                yield f"data: {json.dumps(event, default=str)}\n\n"
        except Exception as exc:
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/travel/stream")
async def travel_stream(request_data: TravelRequest):
    message = request_data.message.strip()
    if not message:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Message cannot be empty."},
        )
    return _sse(stream_travel_agent(message, request_data.thread_id))


@app.post("/api/travel/approve/stream")
async def approve_stream(request_data: ApprovalRequest):
    if not request_data.approved and not request_data.feedback.strip():
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Please provide revision feedback when rejecting the draft.",
            },
        )
    return _sse(
        stream_resume_travel_agent(
            request_data.thread_id,
            request_data.approved,
            request_data.feedback,
        )
    )




@app.post("/api/travel")
async def travel_planner(request_data: TravelRequest):
    try:
        user_message = request_data.message.strip()
        if not user_message:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Message cannot be empty."},
            )
        result = run_travel_agent(
            user_input=user_message, thread_id=request_data.thread_id
        )
        return JSONResponse(content={"success": True, **result})
    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/api/travel/approve")
async def approve_travel_plan(request_data: ApprovalRequest):
    try:
        if not request_data.approved and not request_data.feedback.strip():
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Please provide revision feedback when rejecting the draft.",
                },
            )
        result = resume_travel_agent(
            thread_id=request_data.thread_id,
            approved=request_data.approved,
            feedback=request_data.feedback,
        )
        return JSONResponse(content={"success": True, **result})
    except Exception as e:
        print("APPROVAL ERROR:", e)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "Hamsafar API is running",
        "features": [
            "supervisor_agent",
            "input_guardrail",
            "budget_agent",
            "human_in_the_loop",
            "live_agent_streaming",
        ],
    }


@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={})


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)