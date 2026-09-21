# os lets us read settings (like API keys) from the computer.
import os
# certifi gives a trusted list of security certificates for safe internet (HTTPS) calls.
import certifi
# dotenv reads the secret values we wrote in the .env file.
from dotenv import load_dotenv

# Load everything from the .env file (GROQ_API_KEY, DATABASE_URL, ...).
# We do this FIRST so the code below can use those values.
load_dotenv()

# Tell Python to use certifi's certificate list for secure connections.
# This avoids 'SSL certificate' errors on some computers.
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# asyncio runs 'async' functions (our weather MCP functions are async).
import asyncio
# json turns text like '{"a": 1}' into a Python dictionary.
import json
# operator.add is used so the messages list GROWS (new items are added) instead of being replaced.
import operator
# uuid makes a random unique id (we use it for thread_id).
import uuid
# ThreadPoolExecutor lets us run code in a separate thread (used in _run_async).
from concurrent.futures import ThreadPoolExecutor
# Type helpers. They only DESCRIBE what kind of data we use. They don't change how the code runs.
from typing import Any, TypedDict, Annotated

# psycopg is the library that talks to the PostgreSQL database.
import psycopg
# dict_row makes database rows come back as dictionaries.
from psycopg.rows import dict_row

# LangGraph pieces:
# StateGraph = the flowchart builder, START = where the flow begins, END = where the flow finishes.
from langgraph.graph import StateGraph, START, END
# PostgresSaver saves the progress of every conversation in PostgreSQL (the graph's memory).
from langgraph.checkpoint.postgres import PostgresSaver
# interrupt = PAUSES the graph and waits for a human.
# Command = RESUMES the paused graph with the human's answer.
from langgraph.types import Command, interrupt
# Message types used to talk to the AI:
# HumanMessage = what the user says, AIMessage = what the AI says,
# SystemMessage = instructions for the AI, AnyMessage = any of these.
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
# ChatGroq lets us use AI models hosted on Groq.
from langchain_groq import ChatGroq

# Our own weather MCP helpers:
# weather_mcp_search = current weather, forecast_mcp_search = forecast,
# extract_destination = finds the city name inside the user's text.
from mcp_client import forecast_mcp_search, extract_destination, weather_mcp_search
# Our tool that searches the web (used for hotels).
from tools.tavily_tool import tavily_search
# Our tool that searches flights.
from tools.flight_tool import search_flights


# =========================
# Config
# =========================
# Returns the database link (URL).
# It reads DATABASE_URL from .env and adds 'sslmode=require' when the database is not on our own computer.
def get_database_url():
    # Read the database link from the .env file.
    database_url = os.getenv("DATABASE_URL")

    # If the link is missing, stop the program with a clear message.
    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your PostgreSQL URL to .env"
        )

    # Only append sslmode=require for remote databases
    # Check if the database is on our own computer (localhost).
    is_local = "localhost" in database_url or "127.0.0.1" in database_url

    # If the database is remote AND has no ssl setting yet, add it.
    # Remote databases need a secure connection.
    if not is_local and "sslmode=" not in database_url:
        # If the link already has a '?', join with '&'. Otherwise start with '?'.
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    # Give back the final, ready-to-use link.
    return database_url


# Read the Groq API key (like a password for the AI model) from .env.
# If it is missing, stop right away with a clear message.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")


# =========================
# LLM
# =========================
# Create the AI model object. Every agent below uses this same 'llm'.
# model = which AI model to use, api_key = our Groq password.
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
)


# =========================
# State
# =========================
# STATE = a shared notebook that every agent can read and write.
# Each agent writes its own result into it, and the next agent reads it.
# total=False means not every key must always be filled.
class TravelState(TypedDict, total=False):
    # Chat history. operator.add means new messages are ADDED to the list, not replacing the old ones.
    messages: Annotated[list[AnyMessage], operator.add]
    # The user's original question.
    user_query: str

    # Supervisor + guardrail state
    # True = request is travel-related and allowed, False = blocked.
    guardrail_allowed: bool
    # Short reason from the guardrail (why it was allowed or blocked).
    guardrail_reason: str
    # List of agents the supervisor picked for this request.
    selected_agents: list[str]
    # Trip details the supervisor pulled out of the question (destination, budget, ...).
    trip_constraints: dict[str, Any]
    # Why the supervisor picked those agents.
    supervisor_reasoning: str

    # Specialist results
    # Output of each specialist agent (each one is saved in its own key):
    flight_results: str
    hotel_results: str
    weather_results: str
    budget_results: str
    # The day-by-day plan written by the itinerary agent.
    itinerary: str
    # The final answer shown to the user.
    final_response: str

    # Human-in-the-loop (HITL) state
    # The message asking the human to review the draft itinerary.
    approval_request: str
    # True = the human approved the draft, False = the human wants changes.
    approved: bool
    # The human's revision notes (empty if approved without notes).
    human_feedback: str

    # Counter: how many times we called the AI model.
    llm_calls: int


# =========================
# Shared helpers
# =========================
# Order in which specialists run. itinerary_agent is always last.
# The supervisor only chooses WHICH agents run.
# This list decides the ORDER they run in.
AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]
# A set of valid agent names.
# Used to ignore any wrong / made-up name the AI might return.
KNOWN_AGENTS = set(AGENT_ORDER)


# Small helper: sends a system prompt + a user prompt to the AI and returns only the reply text.
# The supervisor uses it for the guardrail check and for planning.
def _llm_text(system_prompt: str, user_prompt: str) -> str:
    # invoke = send the messages to the AI and wait for its answer.
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    # response.content is the AI's text answer.
    return str(response.content)


# Helper: AI models sometimes add extra words around the JSON.
# This finds the first '{' and the last '}' and reads only that part as JSON.
def _json_from_llm(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object returned by the model."""
    # Position of the first '{'.
    start = text.find("{")
    # Position of the LAST '}'.
    end = text.rfind("}")

    # If no proper JSON was found, raise an error (the caller catches it).
    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON object.")

    # Cut out only the JSON part and turn it into a Python dictionary.
    return json.loads(text[start : end + 1])


# Helper: returns a blank trip-details dictionary.
# Used as a starting point, and when a request is blocked.
def _empty_constraints() -> dict[str, Any]:
    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
    }


# Helper: our weather MCP functions are async, but graph nodes are normal (sync) functions.
# This runs an async function and gives back its result.
# Case 1: no event loop is running -> just use asyncio.run.
# Case 2: a loop is already running (e.g. inside async FastAPI) -> run it in a separate thread to avoid errors.
def _run_async(coro):
    """
    Run a coroutine from sync code. Works both in plain scripts and when the
    graph is invoked from inside a running event loop (e.g. an async FastAPI
    endpoint), where asyncio.run() would otherwise raise.
    """
    try:
        # Ask: is an event loop already running?
        # If yes this line works. If no, it raises RuntimeError (handled below).
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop is running -> it is safe to just use asyncio.run.
        return asyncio.run(coro)

    # A loop IS running -> run the coroutine in a new thread and wait for the result.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


# =========================
# Supervisor Agent + Input Guardrail
# =========================
# SUPERVISOR = the boss node. It runs FIRST and does 2 jobs:
# 1) Guardrail: is this a travel question? If not, block it.
# 2) Supervisor: decide which specialist agents are needed and pull out the trip details.
def supervisor_agent(state: TravelState):
    # Get the user's question from the shared state.
    query = state["user_query"]
    # Get the current AI-call counter (0 if not set yet).
    llm_calls = state.get("llm_calls", 0)

    # The instructions we send to the AI for the guardrail check.
    # Note: {{ }} (double braces) are used so Python prints a single { } inside an f-string.
    guardrail_prompt = f"""
Determine whether the following request belongs to travel planning or travel
information. Valid requests can include destinations, flights, hotels, weather,
budgets, visas, transportation, sightseeing, food, packing, or itineraries.

Block clearly unrelated requests and requests asking for harmful or illegal
instructions. Do not block a valid travel request merely because some details
are missing.

Return strict JSON only:
{{
  "allowed": true,
  "reason": ""
}}

User request:
{query}
"""

    # STEP 1: GUARDRAIL - ask the AI if the request is allowed.
    # Fail open on parser/model errors so a temporary JSON-format issue does not
    # break normal travel planning.
    try:
        # Send the guardrail prompt to the AI.
        guardrail_raw = _llm_text(
            "You are the input guardrail for a travel-planning application. "
            "Return strict JSON only.",
            guardrail_prompt,
        )
        # Turn the AI's reply text into a dictionary.
        guardrail_result = _json_from_llm(guardrail_raw)
        # Read 'allowed' (True/False). If it is missing, default to True.
        allowed = bool(guardrail_result.get("allowed", True))
        # Read the reason text and remove extra spaces.
        guardrail_reason = str(guardrail_result.get("reason", "")).strip()
        # Count this AI call.
        llm_calls += 1
    except Exception as exc:
        # If anything went wrong (AI error or bad JSON), print the error but DON'T stop.
        # 'Fail open' = we allow the request, so a small glitch does not break the whole app.
        print(f"Guardrail fallback used: {exc}")
        allowed = True
        guardrail_reason = "Guardrail validation fallback allowed the request."

    # If the guardrail said NO, stop here and return a rejection message.
    # No other agent will run.
    if not allowed:
        # Use the AI's reason. If it is empty, use our default polite message.
        reason = guardrail_reason or (
            "Hamsafar AI can only help with travel-planning requests. "
            "Please ask about a destination, flight, hotel, weather, "
            "or itinerary."
        )
        # Save the blocked result in the state: allowed=False, no agents,
        # and the message that will be shown to the user.
        # The graph then goes to 'guardrail_blocked' and ends.
        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [AIMessage(content=f"Guardrail blocked request: {reason}")],
            "llm_calls": llm_calls,
        }

    # STEP 2: SUPERVISOR - instructions for the AI to choose the agents and pull out trip details.
    # The AI must reply in JSON only.
    supervisor_prompt = f"""
You are the supervisor of a multi-agent travel-planning system.
Choose only the specialist agents needed for the request.

Available agents:
- flight_agent: flights, airports, airlines, routes, airfare, or booking advice
- hotel_agent: hotels, accommodation, neighborhoods, or places to stay
- weather_agent: weather, climate, season, forecast, or packing advice
- budget_agent: cost, affordability, price limits, or budget feasibility
- itinerary_agent: creates the integrated travel plan and must always be included

Return strict JSON only using this schema:
{{
  "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
  "trip_constraints": {{
    "destination": "",
    "origin": "",
    "duration": "",
    "budget": "",
    "travel_style": "",
    "special_preferences": []
  }},
  "reasoning": ""
}}

User request:
{query}
"""

    try:
        # Send the supervisor prompt to the AI.
        supervisor_raw = _llm_text(
            "You route work to travel specialist agents. Return strict JSON only.",
            supervisor_prompt,
        )
        # Turn the AI's reply into a dictionary.
        parsed = _json_from_llm(supervisor_raw)

        # The list of agents the AI asked for.
        requested_agents = parsed.get("selected_agents", [])
        # Keep only valid agent names and put them in the fixed order (AGENT_ORDER).
        # This ignores made-up names and duplicates.
        selected_agents = [
            name
            for name in AGENT_ORDER
            if name in requested_agents and name in KNOWN_AGENTS
        ]

        # Safety rule: the itinerary agent must ALWAYS be included, even if the AI forgot it.
        # The itinerary agent integrates whichever specialist results were selected.
        if "itinerary_agent" not in selected_agents:
            selected_agents.append("itinerary_agent")

        # Start with a blank trip-details dictionary.
        constraints = _empty_constraints()
        # The trip details found by the AI (destination, budget, ...).
        parsed_constraints = parsed.get("trip_constraints", {})
        # Only use them if they really are a dictionary.
        if isinstance(parsed_constraints, dict):
            # Fill our blank dictionary with the AI's values.
            constraints.update(parsed_constraints)

        # The AI's explanation for its choice.
        reasoning = str(parsed.get("reasoning", "")).strip()
        # Count this AI call.
        llm_calls += 1
    except Exception as exc:
        # If the supervisor step failed, print the error and use a safe backup plan.
        print(f"Supervisor fallback used: {exc}")
        # Fallback: run the full workflow.
        # Backup plan: run ALL agents so the user still gets a full answer.
        selected_agents = AGENT_ORDER.copy()
        # No trip details available, so use the blank dictionary.
        constraints = _empty_constraints()
        reasoning = (
            "Supervisor parsing failed, so the full travel workflow "
            "was selected as a safe fallback."
        )

    # Save everything the supervisor decided into the state:
    # allowed=True, chosen agents, trip details, reasoning and the AI-call counter.
    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": llm_calls,
    }


# =========================
# Guardrail blocked node
# =========================
# GUARDRAIL BLOCKED node: runs ONLY when the guardrail said NO.
# It puts the rejection message into the final answer, then the graph ends.
def guardrail_blocked_agent(state: TravelState):
    # Pick the message to show. 'or' picks the first one that is not empty:
    # first final_response, then guardrail_reason, otherwise a default message.
    reason = (
        state.get("final_response")
        or state.get("guardrail_reason")
        or "This request was blocked by the travel input guardrail."
    )
    # Save the message as the final answer and add it to the chat history.
    return {
        "final_response": reason,
        "messages": [AIMessage(content=reason)],
    }


# =========================
# Flight Agent (tool)
# =========================
# FLIGHT AGENT: gets flight information using our search_flights tool.
def flight_agent(state: TravelState):
    # Take the user's question.
    query = state["user_query"]

    try:
        # Call the flight tool with the user's question.
        flight_data = search_flights(query)
    except Exception as exc:
        # If the tool fails (no internet, API limit, ...) don't crash.
        # Save a fallback message so the AI writes general advice instead.
        print(f"FLIGHT AGENT ERROR: {type(exc).__name__}: {exc}", flush=True)
        flight_data = (
            "Live flight search is temporarily unavailable. Provide general "
            "flight guidance and clearly label it as non-live advice."
        )

    # Save the flight info in the state, add a chat message,
    # and increase the AI-call counter by 1.
    return {
        "flight_results": flight_data,
        "messages": [AIMessage(content="Flight results fetched.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Hotel Agent (tool)
# =========================
# HOTEL AGENT: searches the web for hotels using the tavily_search tool.
def hotel_agent(state: TravelState):
    # Make a search sentence like: 'Best hotels for <user question>'.
    query = f"Best hotels for {state['user_query']}"

    try:
        # Run the web search.
        hotel_results = tavily_search(query)
    except Exception as exc:
        # If the search fails, use a fallback message instead of crashing.
        print(f"HOTEL AGENT ERROR: {type(exc).__name__}: {exc}", flush=True)
        hotel_results = (
            "Live hotel search is temporarily unavailable. Provide general "
            "accommodation and neighborhood guidance based on the destination "
            "and clearly label it as non-live advice."
        )

    # Save the hotel info in the state, add a chat message,
    # and increase the AI-call counter by 1.
    return {
        "hotel_results": hotel_results,
        "messages": [AIMessage(content="Hotel information fetched.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Weather Agent (custom weather MCP)
# =========================
# WEATHER AGENT: gets current weather and forecast from our custom weather MCP.
def weather_agent(state: TravelState):
    # Find the city name in the user's question.
    city = extract_destination(state["user_query"])

    try:
        # Get the current weather (async function, so we use _run_async).
        weather_data = _run_async(weather_mcp_search(city))
        # Get the weather forecast.
        forecast_data = _run_async(forecast_mcp_search(city))

        # Join both results into one text block.
        weather_results = f"""
Current Weather:
{weather_data}

Forecast:
{forecast_data}
"""
    except Exception as exc:
        # If the weather MCP fails, save a fallback message instead of crashing.
        print(f"WEATHER AGENT MCP ERROR: {type(exc).__name__}: {exc}", flush=True)
        weather_results = (
            f"Live weather information for {city} is temporarily unavailable. "
            "Give general seasonal guidance and advise the traveler to verify "
            "the forecast before departure."
        )

    # Save the weather info in the state and add a chat message.
    # (llm_calls is not increased here because no AI call happened.)
    return {
        "weather_results": weather_results,
        "messages": [AIMessage(content="Weather information fetched.")],
    }


# =========================
# Budget Agent
# =========================
# BUDGET AGENT: checks if the trip is affordable,
# using the flight, hotel and weather info collected before it.
def budget_agent(state: TravelState):
    # Instructions for the AI: read all results so far and judge the budget.
    prompt = f"""
Analyze whether this trip is realistic for the user's budget.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results:
{state.get('flight_results', '')}

Hotel Results:
{state.get('hotel_results', '')}

Weather Results:
{state.get('weather_results', '')}

Return:
1. Estimated cost categories
2. Budget risk areas
3. Money-saving suggestions
4. Overall feasibility

If exact live prices are unavailable, clearly label estimates as approximate.
"""

    try:
        # Send the prompt to the AI.
        response = llm.invoke(
            [
                SystemMessage(content="You are a practical travel budget analyst."),
                HumanMessage(content=prompt),
            ]
        )
        # Take the AI's text answer.
        budget_results = str(response.content)
    except Exception as exc:
        # If the AI call fails, use a fallback message instead of crashing.
        print(f"BUDGET AGENT ERROR: {type(exc).__name__}: {exc}", flush=True)
        budget_results = (
            "Budget analysis is temporarily unavailable. Give only rough, "
            "clearly-labelled approximate cost guidance."
        )

    # Save the budget analysis in the state, add a chat message,
    # and increase the AI-call counter by 1.
    return {
        "budget_results": budget_results,
        "messages": [AIMessage(content="Budget assessment generated.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Itinerary Agent
# =========================
# ITINERARY AGENT: ALWAYS runs. It combines the flight, hotel, weather
# and budget results into one day-by-day plan.
def itinerary_agent(state: TravelState):
    # Instructions for the AI: build a full itinerary from all the results.
    prompt = f"""
Create a complete travel itinerary.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results:
{state.get('flight_results', '')}

Hotel Results:
{state.get('hotel_results', '')}

Weather Results:
{state.get('weather_results', '')}

Budget Results:
{state.get('budget_results', '')}

Make the itinerary practical, budget-aware, and easy to follow.
Use only the information provided above; if a section is empty, it was not
requested, so skip it rather than inventing details.
Create a clear draft that is ready for human review.
"""

    # Send the prompt to the AI and get the itinerary.
    response = llm.invoke(
        [
            SystemMessage(content="You are an expert travel planner."),
            HumanMessage(content=prompt),
        ]
    )

    # The message we show to the human, asking them to review this draft.
    approval_request = (
        "Please review the generated draft itinerary. Approve it to create the "
        "final polished plan, or provide feedback for revision."
    )

    # Save the itinerary in the state, add a chat message,
    # and increase the AI-call counter by 1.
    # Also save the approval message. The next node (human_approval) will pause and show it.
    return {
        "itinerary": response.content,
        "approval_request": approval_request,
        "messages": [AIMessage(content="Draft itinerary created for human review.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Human-in-the-Loop approval
# =========================
# HUMAN APPROVAL node: PAUSES the graph so a real person can review the draft itinerary.
# The person can approve it, or send feedback asking for changes.
# After the person answers, the graph continues to the final agent.
def human_approval_agent(state: TravelState):
    # Do not wrap interrupt() in try/except. LangGraph uses it to pause execution.
    # interrupt() stops the graph here and sends this data out to the frontend.
    # The graph stays paused (saved in PostgreSQL) until resume_travel_agent() is called.
    review = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get("itinerary", ""),
            "approval_request": state.get("approval_request", ""),
            "selected_agents": state.get("selected_agents", []),
            "supervisor_reasoning": state.get("supervisor_reasoning", ""),
            "expected_response": {
                "approved": True,
                "feedback": "Optional revision feedback",
            },
        }
    )

    # After the human answers, 'review' holds their answer.
    # Read 'approved' (True/False). If it is missing, treat it as False.
    approved = bool(review.get("approved", False))
    # Read the human's feedback text and remove extra spaces.
    human_feedback = str(review.get("feedback", "")).strip()

    # Save the human's decision in the state so the final agent can use it.
    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [AIMessage(content="Human approval step completed.")],
    }


# =========================
# Final Response Agent
# =========================
# FINAL AGENT: takes everything (flights, hotels, weather, budget, itinerary)
# and asks the AI to write one neat final answer for the user.
def final_agent(state: TravelState):
    # Tell the AI what the human decided:
    # approved -> keep the draft and just polish it.
    # not approved -> apply the human's feedback carefully.
    if state.get("approved", False):
        review_instruction = (
            "The user approved the draft. Preserve its decisions while polishing it."
        )
    else:
        review_instruction = f"""
The user requested a revision. Apply this feedback carefully:
{state.get('human_feedback', '') or 'Improve the draft before finalizing it.'}
"""

    # Instructions for the AI on how to write the final answer
    # (including the list of sections to use).
    final_prompt = f"""
Generate the final travel response for the user.

Human Review:
{review_instruction}

User Request:
{state['user_query']}

Supervisor Constraints:
{state.get('trip_constraints', {})}

Flights:
{state.get('flight_results', '')}

Hotels:
{state.get('hotel_results', '')}

Weather Results:
{state.get('weather_results', '')}

Budget Analysis:
{state.get('budget_results', '')}

Itinerary:
{state.get('itinerary', '')}

Format the final answer beautifully using these sections:

1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Weather Information
5. Day-by-Day Itinerary
6. Estimated Budget
7. Final Recommendations

Important:
- Be clear and practical.
- Mention that live flight API may not provide ticket prices if pricing is unavailable.
- Keep the response useful for real travel planning.
- Omit any section whose data was not gathered and is not relevant to the request.
- Incorporate the human feedback when revision was requested.
"""

    # Send the prompt to the AI and get the final answer.
    response = llm.invoke(
        [
            SystemMessage(content="You are a professional AI travel booking assistant."),
            HumanMessage(content=final_prompt),
        ]
    )

    # Save the final answer. The AI reply is also added to 'messages',
    # so it becomes the last chat message.
    return {
        "final_response": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Dynamic Supervisor Routing
# =========================
# ROUTING = deciding which node runs next.
# ROUTE_MAP lists every node the flow is allowed to go to.
# LangGraph needs it to know all possible destinations.
ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}


# Helper: returns the agents chosen by the supervisor, in the correct fixed order.
def _selected_agents(state: TravelState) -> list[str]:
    # The list picked by the supervisor.
    selected = state.get("selected_agents", [])
    # Go through AGENT_ORDER and keep only the selected ones.
    # This keeps the order fixed.
    return [agent for agent in AGENT_ORDER if agent in selected]


# Runs right after the supervisor. Returns the NAME of the next node.
# Blocked -> 'guardrail_blocked'. Otherwise -> the first selected agent.
def route_from_supervisor(state: TravelState) -> str:
    # If the guardrail blocked the request, go to the blocked node.
    if not state.get("guardrail_allowed", True):
        return "guardrail_blocked"

    # Get the chosen agents in order.
    selected = _selected_agents(state)
    # Go to the first chosen agent.
    # If the list is somehow empty, go straight to the itinerary agent.
    return selected[0] if selected else "itinerary_agent"


# Makes a routing function for ONE agent. It answers: 'after THIS agent, who is next?'
# It looks at the agents that come later in AGENT_ORDER
# and picks the first one that was selected.
# If none is left, it goes to the itinerary agent.
def route_after_agent(current_agent: str):
    # This inner function is what LangGraph actually calls after the agent finishes.
    def route(state: TravelState) -> str:
        # Which agents were chosen?
        selected = _selected_agents(state)
        # Position of the current agent in the fixed order.
        current_index = AGENT_ORDER.index(current_agent)

        # Look only at the agents that come AFTER the current one.
        for next_agent in AGENT_ORDER[current_index + 1 :]:
            # The first selected one we find is the next agent to run.
            if next_agent in selected:
                return next_agent

        # No selected agents left -> go to the itinerary agent.
        return "itinerary_agent"

    # Give back the inner function.
    return route


# =========================
# Build Graph
# =========================
# Build the flowchart.
# StateGraph(TravelState) = a graph that uses our shared notebook (the state).
graph = StateGraph(TravelState)

# Add nodes (the boxes in the flowchart).
# Format: add_node('name', function_to_run)
graph.add_node("supervisor", supervisor_agent)
graph.add_node("guardrail_blocked", guardrail_blocked_agent)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("budget_agent", budget_agent)
graph.add_node("itinerary_agent", itinerary_agent)
# The node where the graph pauses and waits for the human to approve or give feedback.
graph.add_node("human_approval", human_approval_agent)
graph.add_node("final_agent", final_agent)

# Add edges (the arrows). The flow always starts at the supervisor.
graph.add_edge(START, "supervisor")
# After the supervisor, the next node is decided at runtime by route_from_supervisor.
graph.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)

# After each specialist agent, route_after_agent decides which agent runs next.
graph.add_conditional_edges("flight_agent", route_after_agent("flight_agent"), ROUTE_MAP)
graph.add_conditional_edges("hotel_agent", route_after_agent("hotel_agent"), ROUTE_MAP)
graph.add_conditional_edges("weather_agent", route_after_agent("weather_agent"), ROUTE_MAP)

graph.add_conditional_edges("budget_agent", route_after_agent("budget_agent"), ROUTE_MAP)

# Fixed arrows: itinerary -> final answer -> END.
# A blocked request goes straight to END.
# CHANGED: the itinerary now goes to human_approval first (the graph pauses there),
# and only after the human answers does it continue to the final agent.
graph.add_edge("itinerary_agent", "human_approval")
graph.add_edge("human_approval", "final_agent")
graph.add_edge("final_agent", END)
graph.add_edge("guardrail_blocked", END)


# =========================
# PostgreSQL Checkpointer
# =========================
# Get the database link.
DATABASE_URL = get_database_url()

# Connect to PostgreSQL.
# autocommit=True = save changes right away,
# row_factory=dict_row = rows come back as dictionaries.
_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row,
)

# The checkpointer saves the state after every step,
# so a conversation (thread_id) can be remembered and resumed later.
checkpointer = PostgresSaver(_conn)
# Create the needed tables in the database if they don't exist yet.
checkpointer.setup()

# Finish building the graph and attach the database saver.
# travel_graph is the ready-to-use app.
travel_graph = graph.compile(checkpointer=checkpointer)


# =========================
# FastAPI-facing helpers
# =========================
# Helper: checks if the graph PAUSED for human approval.
# When the graph pauses, the result has an '__interrupt__' key. We read the data inside it.
def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    # Get the list of pauses (empty if the graph did not pause).
    interrupts = result.get("__interrupt__", [])
    # No pause -> nothing to return.
    if not interrupts:
        return None

    # Take the first pause.
    first_interrupt = interrupts[0]
    # The data we sent inside interrupt() is stored in '.value'.
    payload = getattr(first_interrupt, "value", first_interrupt)
    # Always give back a dictionary.
    return payload if isinstance(payload, dict) else {"value": payload}


# Helper: turns the graph result into the clean dictionary we send to the frontend.
# Used by both run_travel_agent (first run) and resume_travel_agent (after human review).
def _serialize_result(
    result: dict[str, Any],
    thread_id: str,
) -> dict[str, Any]:
    # The chat history at the end of the run.
    messages = result.get("messages", [])
    # Text of the last message (empty if there are none).
    last_message = messages[-1].content if messages else ""
    # Prefer final_response. If it is empty, use the last message.
    answer = result.get("final_response") or last_message
    # Did the graph pause for human approval? (None = no)
    interrupt_payload = _interrupt_payload(result)

    # If the graph is paused, the answer to show is the DRAFT itinerary (waiting for approval).
    if interrupt_payload:
        answer = interrupt_payload.get("draft_itinerary") or result.get(
            "itinerary", ""
        )

    # Send back the answer plus every intermediate result,
    # so the frontend can show them.
    return {
        "thread_id": thread_id,
        "answer": answer,
        # True = the frontend must show the Approve / Request changes buttons.
        "requires_approval": interrupt_payload is not None,
        "approval_request": (
            interrupt_payload.get("approval_request", "")
            if interrupt_payload
            else result.get("approval_request", "")
        ),
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "budget_results": result.get("budget_results", ""),
        "itinerary": (
            interrupt_payload.get("draft_itinerary", "")
            if interrupt_payload
            else result.get("itinerary", "")
        ),
        "selected_agents": result.get("selected_agents", []),
        "trip_constraints": result.get("trip_constraints", {}),
        "supervisor_reasoning": result.get("supervisor_reasoning", ""),
        "guardrail_allowed": result.get("guardrail_allowed", True),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "approved": result.get("approved"),
        "human_feedback": result.get("human_feedback", ""),
        "llm_calls": result.get("llm_calls", 0),
    }


# =========================
# Function for FastAPI
# =========================
# The function FastAPI calls.
# Input: the user's question (and an optional thread_id).
# Output: a dictionary with the answer and all the details.
def run_travel_agent(user_input: str, thread_id: str | None = None):
    # New conversation? Make a random unique thread id.
    # Same thread_id = same saved conversation.
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    # Tell LangGraph which conversation (thread) to save to / load from.
    config = {"configurable": {"thread_id": thread_id}}

    # Run the whole graph.
    # We give it the starting state: every key with an empty starting value.
    result = travel_graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "guardrail_allowed": True,
            "guardrail_reason": "",
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": "",
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "budget_results": "",
            "itinerary": "",
            "final_response": "",
            "approval_request": "",
            "approved": False,
            "human_feedback": "",
            "llm_calls": 0,
        },
        config=config,
    )

    # Turn the raw result into a clean dictionary for the frontend.
    return _serialize_result(result, thread_id)


# The function FastAPI calls AFTER the human reviewed the draft.
# Input: the thread_id of the paused conversation, approved (True/False) and optional feedback.
# Output: the same kind of dictionary as run_travel_agent (now with the final answer).
def resume_travel_agent(
    thread_id: str,
    approved: bool,
    feedback: str = "",
):
    """Resume the paused LangGraph thread after human review."""
    # We must know WHICH paused conversation to continue.
    if not thread_id:
        raise ValueError("thread_id is required to resume a travel plan.")

    # Tell LangGraph which conversation (thread) to load from the database.
    config = {"configurable": {"thread_id": thread_id}}
    # Command(resume=...) wakes up the paused graph and hands it the human's answer.
    # The answer goes straight into interrupt() inside human_approval_agent.
    result = travel_graph.invoke(
        Command(
            resume={
                "approved": approved,
                "feedback": feedback.strip(),
            }
        ),
        config=config,
    )

    # Turn the raw result into a clean dictionary for the frontend.
    return _serialize_result(result, thread_id)