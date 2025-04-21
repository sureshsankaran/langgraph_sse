from langgraph.graph import StateGraph, END
#from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, List, TypedDict, Optional
import asyncio
import json

# Updated State Schema to track history
class WorkflowState(TypedDict):
    query: str  # User query
    subtasks: List[str]  # List of subtasks for tool calls
    tool_outputs: Dict[str, str]  # Outputs from each tool
    aggregated_context: str  # Current iteration's aggregated context
    context_history: List[Dict[str, str]]  # NEW: History of contexts and metadata per iteration
    agent_response: Optional[str]  # Current iteration's agent response
    response_history: List[Dict[str, str]]  # NEW: History of agent responses per iteration
    needs_loop: bool  # Flag for looping
    iteration: int  # NEW: Track iteration count

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0)

def clean_response(response_content: str) -> str:
    return response_content.strip('`').split('\n', 1)[-1].rsplit('\n', 1)[0]

# Simulated MCP server tools (unchanged)
async def log_analyzer(query: str) -> str:
    return json.dumps({"tool": "log_analyzer", "output": f"Error logs for {query}: timeout at 2025-04-20T10:00:00"})

async def system_monitor(query: str) -> str:
    return json.dumps({"tool": "system_monitor", "output": f"CPU usage: 90%, Memory: 85% for {query}"})

async def network_diagnostic(query: str) -> str:
    return json.dumps({"tool": "network_diagnostic", "output": f"Network latency: 200ms for {query}"})

TOOLS = {
    "log_analyzer": log_analyzer,
    "system_monitor": system_monitor,
    "network_diagnostic": network_diagnostic
    #"pyats_mcp": pyats_mcp
}

# Input Node
def input_node(state: WorkflowState) -> WorkflowState:
    print(f"Received query: {state['query']}")
    return {
        "subtasks": [],
        "tool_outputs": {},
        "aggregated_context": "",
        "context_history": [],  # Initialize history
        "agent_response": None,
        "response_history": [],  # Initialize history
        "needs_loop": False,
        "iteration": 0  # Start at iteration 0
    }

# Orchestrator Node
async def orchestrator_node(state: WorkflowState) -> WorkflowState:
    # Increment iteration count
    iteration = state["iteration"] + 1
    # Include prior context in prompt for continuity
    prior_context = "\n".join([ctx["context"] for ctx in state["context_history"]])
    prompt = ChatPromptTemplate.from_template(
        "Given the query: '{query}', prior context: {prior_context}, and available tools: {tools}, "
        "identify troubleshooting subtasks for iteration {iteration}. Return a JSON list of tool names."
    )
    chain = prompt | llm
    tools_list = list(TOOLS.keys())
    response = await chain.ainvoke({
        "query": state["query"],
        "prior_context": prior_context or "None",
        "tools": tools_list,
        "iteration": iteration
    })
    response_stripped = clean_response(response.content)
    subtasks = json.loads(response_stripped) if isinstance(response_stripped, str) else response_stripped
    print(f"Iteration {iteration} - Orchestrator assigned subtasks: {subtasks}")
    
    return {"subtasks": subtasks, "iteration": iteration}

# Tool Node (unchanged)
async def tool_node(state: WorkflowState) -> WorkflowState:
    subtasks = state["subtasks"]
    tool_outputs = {}
    tasks = [TOOLS[tool](state["query"]) for tool in subtasks if tool in TOOLS]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for tool, result in zip(subtasks, results):
        if isinstance(result, Exception):
            tool_outputs[tool] = f"Error: {str(result)}"
        else:
            tool_outputs[tool] = result
    
    print(f"Iteration {state['iteration']} - Tool outputs: {tool_outputs}")
    return {"tool_outputs": tool_outputs}

# Aggregation Node
async def aggregation_node(state: WorkflowState) -> WorkflowState:
    tool_outputs = state["tool_outputs"]
    prompt = ChatPromptTemplate.from_template(
        "Given the tool outputs: {outputs} for iteration {iteration}, synthesize into a concise context summary."
    )
    chain = prompt | llm
    response = await chain.ainvoke({
        "outputs": json.dumps(tool_outputs),
        "iteration": state["iteration"]
    })
    
    aggregated_context = response.content
    # Append to context history
    context_entry = {
        "iteration": state["iteration"],
        "context": aggregated_context,
        "tool_outputs": tool_outputs
    }
    context_history = state["context_history"] + [context_entry]
    
    print(f"Iteration {state['iteration']} - Aggregated context: {aggregated_context}")
    return {"aggregated_context": aggregated_context, "context_history": context_history}

# Agent Node
async def agent_node(state: WorkflowState) -> WorkflowState:
    # Include full context history in prompt
    context_history = "\n".join([f"Iteration {ctx['iteration']}: {ctx['context']}" for ctx in state["context_history"]])
    prior_responses = "\n".join([f"Iteration {resp['iteration']}: {resp['response']}" for resp in state["response_history"]])
    prompt = ChatPromptTemplate.from_template(
        "Given the query: '{query}', current context: {context}, context history: {context_history}, "
        "and prior responses: {prior_responses}, reason through the troubleshooting problem for iteration {iteration}. "
        "Return a JSON object with 'response' (answer or next steps) and 'needs_loop' (boolean)."
    )
    chain = prompt | llm
    response = await chain.ainvoke({
        "query": state["query"],
        "context": state["aggregated_context"],
        "context_history": context_history or "None",
        "prior_responses": prior_responses or "None",
        "iteration": state["iteration"]
    })
    
    stripped_response = clean_response(response.content)
    result = json.loads(stripped_response) if isinstance(stripped_response, str) else stripped_response
    # Append to response history
    response_entry = {
        "iteration": state["iteration"],
        "response": result["response"]
    }
    response_history = state["response_history"] + [response_entry]
    
    print(f"Iteration {state['iteration']} - Agent response: {result['response']}, Needs loop: {result['needs_loop']}")
    return {
        "agent_response": result["response"],
        "response_history": response_history,
        "needs_loop": result["needs_loop"]
    }

# Output Node
def output_node(state: WorkflowState) -> WorkflowState:
    # Synthesize final response using context and response history
    context_history = state["context_history"]
    response_history = state["response_history"]
    final_summary = (
        f"Final Troubleshooting Summary for: {state['query']}\n"
        f"{'='*50}\n"
    )
    for ctx in context_history:
        final_summary += (
            f"Iteration {ctx['iteration']} Context:\n"
            f"{ctx['context']}\n"
            f"Tools Used: {list(ctx['tool_outputs'].keys())}\n\n"
        )
    for resp in response_history:
        final_summary += f"Iteration {resp['iteration']} Response: {resp['response']}\n\n"
    final_summary += f"Final Conclusion: {state['agent_response']}"
    
    print(final_summary)
    return {"agent_response": final_summary}

# Conditional Edge (unchanged)
def route_after_agent(state: WorkflowState) -> str:
    return "orchestrator" if state["needs_loop"] else "output"

# Build the Graph (unchanged)
def build_graph() -> StateGraph:
    graph = StateGraph(WorkflowState)
    graph.add_node("input", input_node)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("tools", tool_node)
    graph.add_node("aggregation", aggregation_node)
    graph.add_node("agent", agent_node)
    graph.add_node("output", output_node)
    
    graph.set_entry_point("input")
    graph.add_edge("input", "orchestrator")
    graph.add_edge("orchestrator", "tools")
    graph.add_edge("tools", "aggregation")
    graph.add_edge("aggregation", "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"orchestrator": "orchestrator", "output": "output"}
    )
    graph.add_edge("output", END)
    
    return graph

graph = build_graph()
compiled_graph = graph.compile()

# Run the Graph
async def run_graph(query: str):
    initial_state = WorkflowState(
        query=query,
        subtasks=[],
        tool_outputs={},
        aggregated_context="",
        context_history=[],
        agent_response=None,
        response_history=[],
        needs_loop=False,
        iteration=0
    )  
    async for state in compiled_graph.astream(initial_state):
        try:
            print(f"Current state: {state}")
        except Exception as e:
            print(f"Error while printing state: {e}")

    
    return state

# Example usage
if __name__ == "__main__":
    import asyncio
    query = "Why is my server crashing?"
    final_state = asyncio.run(run_graph(query))
    print(f"Final output: {final_state['agent_response']}")