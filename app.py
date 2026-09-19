from pathlib import Path
import traceback
import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from backend import run_travel_agent

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="TripMate AI",
    description="LangGraph Multi-Agent Travel Planner with FastAPI Frontend",
    version="1.0.0"
)


# NOTE: this expects a folder named "static" next to app.py, containing
# style.css and script.js (referenced in index.html as /static/style.css
# and /static/script.js). Unchanged — just confirming the expected path:
#   AI-Travel-Agent/
#     app.py
#     static/
#       style.css
#       script.js
app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static"
)


# NOTE: this is what threw jinja2.exceptions.TemplateNotFound — Jinja2Templates
# only looks inside BASE_DIR / "templates". index.html must live at:
#   AI-Travel-Agent/
#     app.py
#     templates/
#       index.html
# Unchanged — the fix was moving the file, not this line.
templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)



class TravelRequest(BaseModel):
    message: str
    thread_id: str | None = None



@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Looks up templates/index.html via the Jinja2Templates config above.
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.post("/api/travel")
async def travel_planner(request_data: TravelRequest):
    try:
        user_message = request_data.message.strip()

        if not user_message:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Message cannot be empty."
                }
            )

        result = run_travel_agent(
            user_input=user_message,
            thread_id=request_data.thread_id
        )

        # NOTE: keys here (success, thread_id, answer) match exactly what
        # script.js's sendMessage()/showResult() read from the response —
        # no change needed for the frontend to work with this.
        return JSONResponse(
            content={
                "success": True,
                "thread_id": result["thread_id"],
                "answer": result["answer"],
                "flight_results": result["flight_results"],
                "hotel_results": result["hotel_results"],
                "itinerary": result["itinerary"],
                "llm_calls": result["llm_calls"],
            }
        )

    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )



@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "AI Travel Planner API is running"
    }


@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={})



if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )