# ✈️ Hamsafar — Multi-Agent AI Travel Planner

Hamsafar turns a natural-language trip request into a practical travel plan: live flight info, hotel suggestions, weather, a budget check, and a day-by-day itinerary. A human reviews the draft before the final plan is produced.

Built with **LangGraph**, **LangChain**, **FastAPI**, **Groq**, and **PostgreSQL**, with a **custom MCP server** for weather.

---

## Features

- 🛡️ **Input guardrail** — blocks non-travel or harmful requests before any agent runs
- 🧭 **Supervisor agent** — extracts trip constraints (destination, origin, duration, budget, style, preferences) and dynamically picks which specialist agents to run
- ✈️ **Flight agent** — live flight data via AviationStack (tool)
- 🏨 **Hotel agent** — hotel and neighborhood research via Tavily (tool)
- 🌦️ **Weather agent** — current weather + forecast through a **custom MCP server** (OpenWeatherMap)
- 💰 **Budget agent** — checks trip feasibility against flights, hotels, and weather results
- 🗓️ **Itinerary agent** — builds an integrated, budget-aware day-by-day draft
- 🙋 **Human-in-the-loop approval** — graph pauses on the draft; approve it or send revision feedback
- 💾 **Persistent state** — every conversation is checkpointed in PostgreSQL and resumable by `thread_id`
- 🧱 **Graceful degradation** — every tool/agent has a fallback, so a failing API never crashes the run
- 🐳 **Dockerized** — one-command container build

---

## Architecture

```mermaid
flowchart TD
    A([User request]) --> S[Supervisor<br/>guardrail + planning]
    S -- blocked --> X[Guardrail blocked] --> Z([END])
    S -- allowed --> F[Flight agent<br/>AviationStack tool]
    F --> H[Hotel agent<br/>Tavily tool]
    H --> W[Weather agent<br/>custom MCP]
    W --> B[Budget agent]
    B --> I[Itinerary agent]
    I --> HA{{Human approval<br/>interrupt}}
    HA -- approve / feedback --> FA[Final agent]
    FA --> Z
```

The supervisor selects **which** specialists run; a fixed order (`flight → hotel → weather → budget → itinerary`) decides **when**. The itinerary agent always runs.

### Tools vs. MCP

| Capability | Integration | Source |
|---|---|---|
| Flights | Direct tool (`search_flights`) | AviationStack |
| Hotels | Direct tool (`tavily_search`) | Tavily |
| Weather | **Custom MCP server** (stdio) | OpenWeatherMap |

The custom MCP server (`custom_mcp.py`) is built with `FastMCP` and exposes two tools:

- `get_current_weather(city)` — temperature, feels-like, humidity, condition, wind speed
- `get_forecast(city)` — next 5 forecast entries

`mcp_client.py` launches it as a local stdio subprocess through `langchain-mcp-adapters` (`MultiServerMCPClient`), so no separate server process is needed. The destination city is extracted from the user query with a small LLM call.

---

## Tech Stack

| Layer | Tech |
|---|---|
| Orchestration | LangGraph, LangChain |
| LLMs | Groq — `openai/gpt-oss-120b` (agents), `llama-3.3-70b-versatile` (destination extraction) |
| MCP | `mcp` (FastMCP), `langchain-mcp-adapters` |
| Search / data | Tavily, AviationStack, OpenWeatherMap |
| Backend | FastAPI, Uvicorn |
| Frontend | Jinja2 + HTML/CSS/JavaScript |
| Persistence | PostgreSQL (`langgraph-checkpoint-postgres`, `psycopg`) |
| Deployment | Docker (`python:3.11-slim`) |

---

## Project Structure

```
.
├── app.py             # FastAPI app: routes, request models
├── backend.py         # LangGraph workflow: supervisor, agents, HITL, checkpointer
├── mcp_client.py      # MCP client for the weather server + destination extractor
├── custom_mcp.py      # Custom weather MCP server (FastMCP)
├── tools/             # Flight (AviationStack) and web search (Tavily) tools
├── static/            # CSS / JS
├── templates/         # index.html
├── requirements.txt
├── Dockerfile
└── LICENSE
```

---

## Prerequisites

- Python 3.11 recommended (3.10+ works)
- A running PostgreSQL instance
- API keys: **Groq**, **Tavily**, **AviationStack**, **OpenWeatherMap**

## Environment Variables

Create a `.env` file in the project root:

```env
DATABASE_URL=postgresql://user:password@localhost:5432/travel_db
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
AVIATIONSTACK_API_KEY=your_aviationstack_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
DEFAULT_ORIGIN_IATA=DAC
```

> For remote databases, `sslmode=require` is appended to `DATABASE_URL` automatically.

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Health check + enabled features |
| `POST` | `/api/travel` | Start a plan (runs until the approval step) |
| `POST` | `/api/travel/approve` | Approve the draft or submit revision feedback |

### Start a plan

```bash
curl -X POST http://127.0.0.1:8000/api/travel \
  -H "Content-Type: application/json" \
  -d '{"message":"Plan a 3-day trip to Tokyo with a budget of $1200"}'
```

Response (trimmed):

```json
{
  "success": true,
  "thread_id": "user_ab12...",
  "requires_approval": true,
  "answer": "<draft itinerary>",
  "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
  "trip_constraints": { "destination": "Tokyo", "duration": "3 days", "budget": "$1200" },
  "guardrail_allowed": true,
  "llm_calls": 5
}
```

Pass `thread_id` back in later requests to continue the same conversation.

### Approve or revise

```bash
# Approve
curl -X POST http://127.0.0.1:8000/api/travel/approve \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"user_ab12...","approved":true}'

# Request changes (feedback is required when approved=false)
curl -X POST http://127.0.0.1:8000/api/travel/approve \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"user_ab12...","approved":false,"feedback":"Add more food spots and cut the budget by 10%"}'
```

---

## How It Works

1. **Guardrail** — the supervisor first checks that the request is travel-related. Blocked requests end immediately with an explanation.
2. **Supervisor** — extracts trip constraints and selects the needed specialist agents.
3. **Specialists** — flight, hotel, weather, and budget agents run in a fixed order and write results to shared state.
4. **Itinerary** — combines all gathered results into a draft plan.
5. **Human approval** — the graph pauses (`interrupt`) and the state is saved to PostgreSQL. The user approves or sends feedback.
6. **Final response** — the final agent polishes the approved draft, or applies the feedback, into a structured answer: trip summary, flights, hotels, weather, itinerary, budget, and recommendations.

### Reliability

- Guardrail **fails open**: a parsing/model error won't block valid travel requests.
- Supervisor failure **falls back to running every agent**.
- Flight, hotel, weather, and budget failures return a labeled *non-live advice* fallback instead of raising.
- Sync graph nodes call async MCP functions through a safe event-loop bridge, so it works inside FastAPI's running loop.

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a pull request

## License

MIT — see [LICENSE](LICENSE).