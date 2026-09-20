#MCP Tools
# Flight Agent
# def flight_agent(state: TravelState):
#     print("\nINSIDE FLIGHT AGENT\n")

#     query = state["user_query"]

#     try:

#         airports = asyncio.run(
#             aviation_mcp_call(
#                 "list_airports"
#             )
#         )

#         airlines = asyncio.run(
#             aviation_mcp_call(
#                 "list_airlines"
#             )
#         )


#         print("\nAIRPORTS:", airports)
#         print("\nAIRLINES:", airlines)

#         prompt = FLIGHT_AGENT_PROMPT.format(
#             query=query,
#             airport_data=str(airports)[:3000],
#             airline_data=str(airlines)[:3000]
#         )

#         response = llm.invoke([
#             SystemMessage(
#                 content="You are an expert travel flight planner."
#             ),
#             HumanMessage(content=prompt)
#         ])

#         flight_data = response.content

#     except Exception as e:

#         flight_data = f"Flight information unavailable: {str(e)}"

#     return {
#         "flight_results": flight_data,
#         "messages": [
#             AIMessage(
#                 content="Flight recommendations generated"
#             )
#         ],
#         "llm_calls": state.get("llm_calls", 0) + 1
#     }





# # =========================
# # Hotel Agent
# # =========================

# def hotel_agent(state: TravelState):
#     query = f"Best hotels for {state['user_query']}"
#     # hotel_results = tavily_search(query)
#     hotel_results = asyncio.run(tavily_mcp_search(query))

#     return {
#         "hotel_results": hotel_results,
#         "messages": [
#             AIMessage(content="Hotel information fetched.")
#         ],
#         "llm_calls": state.get("llm_calls", 0) + 1
#     }
