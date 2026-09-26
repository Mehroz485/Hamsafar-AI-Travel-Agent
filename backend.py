
import os

import certifi

from dotenv import load_dotenv



load_dotenv()



os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


import asyncio

import json

import operator

import uuid

from concurrent.futures import ThreadPoolExecutor

from typing import Any, TypedDict, Annotated


import psycopg

from psycopg.rows import dict_row



from langgraph.graph import StateGraph, START, END

from langgraph.checkpoint.postgres import PostgresSaver


from langgraph.types import Command, interrupt



from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq




from mcp_client import forecast_mcp_search, extract_destination, weather_mcp_search

from tools.tavily_tool import tavily_search

from tools.flight_tool import search_flights





def get_database_url():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your PostgreSQL URL to .env"
        )
    is_local = "localhost" in database_url or "127.0.0.1" in database_url
    if not is_local and "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"
    return database_url

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")




llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
)


class TravelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    guardrail_allowed: bool
    guardrail_reason: str
    selected_agents: list[str]
    trip_constraints: dict[str, Any]
    supervisor_reasoning: str
    flight_results: str
    hotel_results: str
    weather_results: str
    budget_results: str
    itinerary: str
    final_response: str
    approval_request: str
    approved: bool
    human_feedback: str
    llm_calls: int

AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]
KNOWN_AGENTS = set(AGENT_ORDER)


def _llm_text(system_prompt: str, user_prompt: str) -> str:
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return str(response.content)


def _json_from_llm(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON object.")
    return json.loads(text[start : end + 1])


def _empty_constraints() -> dict[str, Any]:
    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
    }

def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()





def supervisor_agent(state: TravelState):
    query = state["user_query"]
    llm_calls = state.get("llm_calls", 0)

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

    try:
        guardrail_raw = _llm_text(
            "You are the input guardrail for a travel-planning application. Return strict JSON only.",
            guardrail_prompt,
        )
        guardrail_result = _json_from_llm(guardrail_raw)
        allowed = bool(guardrail_result.get("allowed", True))
        guardrail_reason = str(guardrail_result.get("reason", "")).strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Guardrail fallback used: {exc}")
        allowed = True
        guardrail_reason = "Guardrail validation fallback allowed the request."

    if not allowed:
        reason = guardrail_reason or (
            "Hamsafar AI can only help with travel-planning requests. "
            "Please ask about a destination, flight, hotel, weather, "
            "or itinerary."
        )
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
        supervisor_raw = _llm_text(
            "You route work to travel specialist agents. Return strict JSON only.",
            supervisor_prompt,
        )
        parsed = _json_from_llm(supervisor_raw)

        requested_agents = parsed.get("selected_agents", [])
        selected_agents = [
            name
            for name in AGENT_ORDER
            if name in requested_agents and name in KNOWN_AGENTS
        ]

        if "itinerary_agent" not in selected_agents:
            selected_agents.append("itinerary_agent")

        constraints = _empty_constraints()
        parsed_constraints = parsed.get("trip_constraints", {})
        if isinstance(parsed_constraints, dict):
            constraints.update(parsed_constraints)

        reasoning = str(parsed.get("reasoning", "")).strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Supervisor fallback used: {exc}")
        selected_agents = AGENT_ORDER.copy()
        constraints = _empty_constraints()
        reasoning = (
            "Supervisor parsing failed, so the full travel workflow "
            "was selected as a safe fallback."
        )

    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": llm_calls,
    }




def guardrail_blocked_agent(state: TravelState):
    reason = (
        state.get("final_response")
        or state.get("guardrail_reason")
        or "This request was blocked by the travel input guardrail."
    )
    return {
        "final_response": reason,
        "messages": [AIMessage(content=reason)],
    }




def flight_agent(state: TravelState):
    query = state["user_query"]
    try:
        flight_data = search_flights(query)
    except Exception as exc:
        print(f"FLIGHT AGENT ERROR: {type(exc).__name__}: {exc}", flush=True)
        
        flight_data = (
            "Live flight pricing is currently unavailable. Provide typical flight estimates "
            "based on historical data."
        )

    return {
        "flight_results": flight_data,
        "messages": [AIMessage(content="Flight results fetched.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }





def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    try:
        hotel_results = tavily_search(query)
    except Exception as exc:
        print(f"HOTEL AGENT ERROR: {type(exc).__name__}: {exc}", flush=True)
        
        hotel_results = (
            "Live hotel search is currently unavailable. Provide typical hotel and neighborhood "
            "guidance based on the destination."
        )

    return {
        "hotel_results": hotel_results,
        "messages": [AIMessage(content="Hotel information fetched.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }





def weather_agent(state: TravelState):
    city = (state.get("trip_constraints") or {}).get("destination", "").strip()
    try:
        if not city:
            city = extract_destination(state["user_query"])

        weather_data = _run_async(weather_mcp_search(city))
        forecast_data = _run_async(forecast_mcp_search(city))

        weather_results = f"""
Current Weather:
{weather_data}

Forecast:
{forecast_data}
"""
    except Exception as exc:
        print(f"WEATHER AGENT MCP ERROR: {type(exc).__name__}: {exc}", flush=True)
        
        weather_results = (
            f"Live weather information for {city or 'the destination'} is currently unavailable. "
            "Provide general seasonal weather guidance."
        )

    return {
        "weather_results": weather_results,
        "messages": [AIMessage(content="Weather information fetched.")],
    }




def budget_agent(state: TravelState):
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
        response = llm.invoke(
            [
                SystemMessage(content="You are a practical travel budget analyst."),
                HumanMessage(content=prompt),
            ]
        )
        budget_results = str(response.content)
    except Exception as exc:
        print(f"BUDGET AGENT ERROR: {type(exc).__name__}: {exc}", flush=True)
        budget_results = "Budget analysis is temporarily unavailable. Give rough cost estimates."

    return {
        "budget_results": budget_results,
        "messages": [AIMessage(content="Budget assessment generated.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }





def itinerary_agent(state: TravelState):
    
    selected = state.get("selected_agents", [])
    
    
    is_full_trip = len(selected) > 2  
    
    if is_full_trip:
        task_instruction = "Create a complete, day-by-day travel itinerary incorporating all the details below."
    else:
        task_instruction = "Compile the requested travel information logically. Do NOT create a day-by-day itinerary unless explicitly requested."

    
    data_blocks = []
    if "flight_agent" in selected and state.get('flight_results'): 
        data_blocks.append(f"Flight Results:\n{state['flight_results']}")
    if "hotel_agent" in selected and state.get('hotel_results'): 
        data_blocks.append(f"Hotel Results:\n{state['hotel_results']}")
    if "weather_agent" in selected and state.get('weather_results'): 
        data_blocks.append(f"Weather Results:\n{state['weather_results']}")
    if "budget_agent" in selected and state.get('budget_results'): 
        data_blocks.append(f"Budget Results:\n{state['budget_results']}")
    
    combined_data = "\n\n".join(data_blocks)


    prompt = f"""
{task_instruction}

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

{combined_data}

CRITICAL RULES FOR THE DRAFT - READ CAREFULLY:
1. SCOPE: Use only the information provided above. Do not invent hotel, weather, or budget details if those sections are missing.
2. NO SYSTEM LEAKS: NEVER mention internal tools (e.g., AviationStack, MCP, OpenWeather). NEVER output raw JSON, error codes, HTTP statuses, or technical phrases like "Invalid API key" or "data-retrieval error."
3. GRACEFUL FALLBACKS: If the provided data contains an error message, ignore the error text. Simply state: "Live data is currently unavailable. Please check local sources before traveling." Do NOT explain *why* the system failed.
4. TONE: Write as a helpful travel assistant talking to an everyday traveler. 
5. Do NOT create a section called "Prepared for Human Review".
Create a clear, beautifully formatted draft.
"""

    response = llm.invoke(
        [
            SystemMessage(content="You are an expert travel planner."),
            HumanMessage(content=prompt),
        ]
    )

    approval_request = (
        "Please review the generated draft itinerary. Approve it to create the "
        "final polished plan, or provide feedback for revision."
    )

    return {
        "itinerary": response.content,
        "approval_request": approval_request,
        "messages": [AIMessage(content="Draft itinerary created for human review.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }





def human_approval_agent(state: TravelState):
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

    approved = bool(review.get("approved", False))
    human_feedback = str(review.get("feedback", "")).strip()

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [AIMessage(content="Human approval step completed.")],
    }





def _trim(text, limit=500):
    """Keep prompts under the TPM limit by capping each raw data block."""
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated for length]"


def final_agent(state: TravelState):
    if state.get("approved", False):
        review_instruction = (
            "The user approved the draft. Preserve its decisions while polishing it."
        )
    else:
        review_instruction = f"""
The user requested a revision. Apply this feedback carefully:
{_trim(state.get('human_feedback', ''), 300) or 'Improve the draft before finalizing it.'}
"""

    selected = state.get("selected_agents", [])
    sections = ["- Summary of Request"]

    if "flight_agent" in selected: sections.append("- Flight Information")
    if "hotel_agent" in selected: sections.append("- Hotel Suggestions")
    if "weather_agent" in selected: sections.append("- Weather Information")
    if "budget_agent" in selected: sections.append("- Estimated Budget")
    if len(selected) > 2: sections.append("- Day-by-Day Itinerary")
    sections.append("- Final Recommendations")

    section_format = "\n".join(sections)

    # CHANGED: every raw data block is trimmed before it enters the prompt —
    # this is what was blowing past the 8000 TPM limit (weather/flight JSON is verbose)
    data_blocks = []
    if "flight_agent" in selected: data_blocks.append(f"Flights:\n{_trim(state.get('flight_results', ''))}")
    if "hotel_agent" in selected: data_blocks.append(f"Hotels:\n{_trim(state.get('hotel_results', ''))}")
    if "weather_agent" in selected: data_blocks.append(f"Weather:\n{_trim(state.get('weather_results', ''))}")
    if "budget_agent" in selected: data_blocks.append(f"Budget:\n{_trim(state.get('budget_results', ''))}")
    combined_data = "\n\n".join(data_blocks)

    final_prompt = f"""
Generate the final travel response for the user.

Human Review:
{review_instruction}

User Request:
{_trim(state['user_query'], 300)}

Supervisor Constraints:
{state.get('trip_constraints', {})}

Raw Data:
{combined_data}

Itinerary Draft (Use this to structure your response):
{_trim(state.get('itinerary', ''), 1500)}

Format the final answer beautifully using Markdown. 
ONLY include the following sections if they apply to the user's request:
{section_format}

CRITICAL RULES - READ CAREFULLY:
1. SCOPE STRICTNESS: Do NOT output sections for Hotels, Weather, or Budgets if they are not in the allowed list above. If the user only asked for flights, give them ONLY flights.
2. NO API LEAKS: NEVER mention internal tool names (e.g., AviationStack, MCP, Amadeus, LangGraph, Tavily). NEVER mention "the agent", "the tools", or "endpoints".
3. GRACEFUL FALLBACKS: If exact live prices or data are missing, simply state "Live pricing is currently unavailable, check Google Flights/Booking.com for current rates." Do NOT explain that an API endpoint failed.
4. TONE: Be professional, clear, practical, and highly formatted.
"""

    response = llm.invoke(
        [
            SystemMessage(content="You are a professional AI travel booking assistant."),
            HumanMessage(content=final_prompt),
        ]
    )

    return {
        "final_response": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }



ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}


def _selected_agents(state: TravelState) -> list[str]:
    selected = state.get("selected_agents", [])
    return [agent for agent in AGENT_ORDER if agent in selected]


def route_from_supervisor(state: TravelState) -> str:
    if not state.get("guardrail_allowed", True):
        return "guardrail_blocked"

    selected = _selected_agents(state)
    return selected[0] if selected else "itinerary_agent"


def route_after_agent(current_agent: str):
    def route(state: TravelState) -> str:
        selected = _selected_agents(state)
        current_index = AGENT_ORDER.index(current_agent)

        for next_agent in AGENT_ORDER[current_index + 1 :]:
            if next_agent in selected:
                return next_agent

        return "itinerary_agent"
    return route





graph = StateGraph(TravelState)

graph.add_node("supervisor", supervisor_agent)
graph.add_node("guardrail_blocked", guardrail_blocked_agent)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("budget_agent", budget_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("human_approval", human_approval_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)

graph.add_conditional_edges("flight_agent", route_after_agent("flight_agent"), ROUTE_MAP)
graph.add_conditional_edges("hotel_agent", route_after_agent("hotel_agent"), ROUTE_MAP)
graph.add_conditional_edges("weather_agent", route_after_agent("weather_agent"), ROUTE_MAP)
graph.add_conditional_edges("budget_agent", route_after_agent("budget_agent"), ROUTE_MAP)

graph.add_edge("itinerary_agent", "human_approval")
graph.add_edge("human_approval", "final_agent")
graph.add_edge("final_agent", END)
graph.add_edge("guardrail_blocked", END)





DATABASE_URL = get_database_url()

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row,
)

checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)





def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return None

    first_interrupt = interrupts[0]
    payload = getattr(first_interrupt, "value", first_interrupt)
    return payload if isinstance(payload, dict) else {"value": payload}


def _serialize_result(
    result: dict[str, Any],
    thread_id: str,
) -> dict[str, Any]:
    messages = result.get("messages", [])
    last_message = messages[-1].content if messages else ""
    answer = result.get("final_response") or last_message
    interrupt_payload = _interrupt_payload(result)

    if interrupt_payload:
        answer = interrupt_payload.get("draft_itinerary") or result.get(
            "itinerary", ""
        )

    return {
        "thread_id": thread_id,
        "answer": answer,
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





def run_travel_agent(user_input: str, thread_id: str | None = None):
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {"configurable": {"thread_id": thread_id}}

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

    return _serialize_result(result, thread_id)


def resume_travel_agent(
    thread_id: str,
    approved: bool,
    feedback: str = "",
):
    if not thread_id:
        raise ValueError("thread_id is required to resume a travel plan.")

    config = {"configurable": {"thread_id": thread_id}}
    result = travel_graph.invoke(
        Command(
            resume={
                "approved": approved,
                "feedback": feedback.strip(),
            }
        ),
        config=config,
    )

    return _serialize_result(result, thread_id)





def _stream_graph(graph_input, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    running = None  

    for mode, chunk in travel_graph.stream(graph_input, config, stream_mode=["tasks"]):
        if not isinstance(chunk, dict):
            continue
        name = chunk.get("name")

        if "triggers" in chunk:  
            running = name  
            yield {"type": "agent_start", "agent": name}
        elif name == "human_approval" and chunk.get("interrupts"):
            if running and running != "human_approval":
                yield {"type": "agent_done", "agent": running, "error": None}
            running = None
            yield {"type": "waiting_approval", "agent": name}
        else:
            running = None  
            yield {"type": "agent_done", "agent": name, "error": chunk.get("error")}

    if running:
        yield {"type": "agent_done", "agent": running, "error": None}

    snapshot = travel_graph.get_state(config)
    result = dict(snapshot.values)
    pending = [i for task in snapshot.tasks for i in task.interrupts]
    if pending:
        result["__interrupt__"] = pending

    yield {"type": "final", "data": {"success": True, **_serialize_result(result, thread_id)}}


def stream_travel_agent(user_input: str, thread_id: str | None = None):
    thread_id = thread_id or f"user_{uuid.uuid4().hex}"
    graph_input = {
        "user_query": user_input,
        "messages": [HumanMessage(content=user_input)],
        "guardrail_allowed": True,
        "guardrail_reason": "",
        "selected_agents": [],
        "trip_constraints": {},
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
    }
    yield from _stream_graph(graph_input, thread_id)


def stream_resume_travel_agent(thread_id: str, approved: bool, feedback: str = ""):
    resume = Command(resume={"approved": approved, "feedback": feedback})
    yield from _stream_graph(resume, thread_id)