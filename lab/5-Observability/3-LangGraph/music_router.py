"""LangGraph sample that streams music tool calls with GenAI telemetry.

Install the dependencies:

    pip install langchain langgraph langchain-openai langchain-azure-ai python-dotenv
    pip install azure-identity  # required when API_HOST=azure

Environment variables:

* API_HOST="github" (default) or "azure"
* APPLICATION_INSIGHTS_CONNECTION_STRING (optional)
* GitHub Models: GITHUB_TOKEN, optional GITHUB_MODEL / GITHUB_OPENAI_BASE_URL
* Azure OpenAI: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_VERSION, AZURE_OPENAI_CHAT_DEPLOYMENT

The `AzureAIOpenTelemetryTracer` records span.azure.ai.inference.client spans for
model invocations, plus `invoke_agent` and `execute_tool` spans as the graph
runs through the tool node.
"""

from __future__ import annotations

import os

import azure.identity
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from langchain_azure_ai.callbacks.tracers import AzureAIOpenTelemetryTracer

load_dotenv(override=True)


def _build_model():
    host = os.getenv("API_HOST", "github").lower()

    if host == "azure":
        token_provider = azure.identity.get_bearer_token_provider(
            azure.identity.DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default",
        )
        return AzureChatOpenAI(
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            openai_api_version=os.environ.get("AZURE_OPENAI_VERSION"),
            azure_ad_token_provider=token_provider,
        )

    if host == "github":
        return ChatOpenAI(
            model=os.getenv("GITHUB_MODEL", "gpt-4o"),
            base_url=os.getenv("GITHUB_OPENAI_BASE_URL", "https://models.inference.ai.azure.com"),
            api_key=os.environ.get("GITHUB_TOKEN"),
        )

    raise ValueError("API_HOST must be 'github' or 'azure'")


TRACER = AzureAIOpenTelemetryTracer(
    connection_string=os.environ.get("APPLICATION_INSIGHTS_CONNECTION_STRING"),
    enable_content_recording=os.getenv("OTEL_RECORD_CONTENT", "true").lower() == "true",
    name="Music Player Agent",
)


@tool
def play_song_on_spotify(song: str) -> str:
    """Simulated Spotify playback."""

    return f"Successfully played {song} on Spotify!"


@tool
def play_song_on_apple(song: str) -> str:
    """Simulated Apple Music playback."""

    return f"Successfully played {song} on Apple Music!"


MODEL = _build_model().bind_tools([play_song_on_spotify, play_song_on_apple], parallel_tool_calls=False)
TOOL_NODE = ToolNode([play_song_on_spotify, play_song_on_apple])


def should_continue(state: MessagesState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    return "end" if not last_message.tool_calls else "continue"


def call_model(state: MessagesState) -> dict:
    messages = state["messages"]
    response = MODEL.invoke(messages)
    return {"messages": [response]}


WORKFLOW = StateGraph(MessagesState)
WORKFLOW.add_node("agent", call_model)
WORKFLOW.add_node("action", TOOL_NODE)
WORKFLOW.add_edge(START, "agent")
WORKFLOW.add_conditional_edges("agent", should_continue, {"continue": "action", "end": END})
WORKFLOW.add_edge("action", "agent")

MEMORY = MemorySaver()
APP = WORKFLOW.compile(checkpointer=MEMORY)


def main() -> None:
    config = {"configurable": {"thread_id": "1"}, "callbacks": [TRACER]}
    input_message = HumanMessage(content="Can you play Taylor Swift's most popular song?")

    for event in APP.stream({"messages": [input_message]}, config, stream_mode="values"):
        event["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()
