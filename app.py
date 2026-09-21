# Path helps us build file/folder paths that work on any operating system
from pathlib import Path
# traceback lets us print the full error details when something crashes
import traceback
# uvicorn is the server that actually runs our FastAPI app
import uvicorn

# FastAPI = the web framework, Request = the incoming browser request
from fastapi import FastAPI, Request
# HTMLResponse = send a web page back, JSONResponse = send data (JSON) back
from fastapi.responses import HTMLResponse, JSONResponse
# StaticFiles = lets the browser download our css/js files
from fastapi.staticfiles import StaticFiles
# Jinja2Templates = lets us send HTML pages from the "templates" folder
from fastapi.templating import Jinja2Templates
# BaseModel = describes what data the frontend must send us (and checks it)
# Field = lets us add rules to a value (for example: must not be empty)
from pydantic import BaseModel, Field

# Our travel agent functions (the LangGraph brain) live in backend.py
# run_travel_agent = starts a new plan, resume_travel_agent = continues after the human review
from backend import run_travel_agent, resume_travel_agent

# The folder where this app.py file lives. All other folders are found from here
BASE_DIR = Path(__file__).resolve().parent

# Create the web app. The title/description/version show up in the auto docs (/docs)
app = FastAPI(
    title="Hamsafar",
    description="LangGraph Multi-Agent Travel Planner with Supervisor, Guardrails, Human-in-the-Loop and FastAPI Frontend",
    version="2.0.0"
)


# Make the "static" folder available to the browser at the address /static
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


# Tell FastAPI where our HTML pages are (the "templates" folder)
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


# This describes the data the frontend sends to /api/travel
class TravelRequest(BaseModel):
    # The user's travel question (required)
    message: str
    # The id of an old conversation. Optional: it is None for a new chat
    thread_id: str | None = None


# This describes the data the frontend sends to /api/travel/approve (the human review)
class ApprovalRequest(BaseModel):
    # Which paused conversation to continue (required, cannot be empty)
    thread_id: str = Field(min_length=1)
    # True = the user approved the draft, False = the user wants changes
    approved: bool
    # The user's revision notes (optional when approving)
    feedback: str = ""


# When someone opens the home page ("/"), send them the HTML page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Looks up templates/index.html via the Jinja2Templates config above.
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# The frontend calls this address when the user clicks "Generate plan"
@app.post("/api/travel")
async def travel_planner(request_data: TravelRequest):
    try:
        # Take the user's message and remove extra spaces at the start/end
        user_message = request_data.message.strip()

        # If the message is empty, stop early and tell the user (400 = bad request)
        if not user_message:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Message cannot be empty."
                }
            )

        # Run the whole travel agent (guardrail -> supervisor -> agents -> final answer)
        result = run_travel_agent(
            user_input=user_message,
            thread_id=request_data.thread_id
        )

        # NOTE: keys here (success, thread_id, answer) match exactly what
        # script.js's sendMessage()/showResult() read from the response —
        # no change needed for the frontend to work with this.
        # CHANGED: instead of listing each key by hand, **result sends ALL the
        # keys from the agent. This now includes the new ones the frontend uses:
        # selected_agents (agent chips), guardrail_allowed (blocked requests)
        # and budget_results, plus the old ones (thread_id, answer,
        # flight_results, hotel_results, itinerary, llm_calls).
        return JSONResponse(
            content={
                "success": True,
                **result
            }
        )

    # If anything crashes above, we land here so the server doesn't just die
    except Exception as e:
        # Print the error in the terminal so we can debug it
        print("ERROR:", e)
        traceback.print_exc()

        # Send a clean error message to the frontend (500 = server error)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


# The frontend calls this address when the user clicks "Approve" or "Request changes"
@app.post("/api/travel/approve")
async def approve_travel_plan(request_data: ApprovalRequest):
    try:
        # If the user rejects the draft, we need to know WHAT to change
        if not request_data.approved and not request_data.feedback.strip():
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Please provide revision feedback when rejecting the draft."
                }
            )

        # Wake up the paused graph and give it the user's decision
        result = resume_travel_agent(
            thread_id=request_data.thread_id,
            approved=request_data.approved,
            feedback=request_data.feedback
        )

        # Send the final result back (same shape as /api/travel)
        return JSONResponse(
            content={
                "success": True,
                **result
            }
        )

    # If anything crashes above, we land here so the server doesn't just die
    except Exception as e:
        # Print the error in the terminal so we can debug it
        print("APPROVAL ERROR:", e)
        traceback.print_exc()

        # Send a clean error message to the frontend (500 = server error)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


# A simple address to check that the server is alive
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "Hamsafar API is running",
        # The features this app has (CHANGED: new features listed)
        "features": [
            "supervisor_agent",
            "input_guardrail",
            "budget_agent",
            "human_in_the_loop"
        ]
    }


# Browsers ask for a favicon (tab icon) automatically. We answer with an empty
# reply so the terminal doesn't fill with "404 not found" errors
@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={})


# This part runs only when we start the file directly with: python app.py
if __name__ == "__main__":
    # Start the server at http://127.0.0.1:8000
    # reload=True restarts the server automatically when we change the code
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )