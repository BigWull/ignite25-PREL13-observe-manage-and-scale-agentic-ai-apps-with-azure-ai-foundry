"""LangChain v1 weekend planner instrumented with Azure AI OpenTelemetry tracer.

Install the dependencies:

    pip install langchain langchain-openai langchain-azure-ai rich python-dotenv
    pip install azure-identity  # only required when API_HOST=azure

Environment variables:

* API_HOST="github" (default) or "azure"
* APPLICATION_INSIGHTS_CONNECTION_STRING (optional)
* GitHub Models:
    - GITHUB_TOKEN
    - GITHUB_MODEL (optional, defaults to "gpt-4o")
* Azure OpenAI:
    - AZURE_OPENAI_ENDPOINT
    - AZURE_OPENAI_VERSION
    - AZURE_OPENAI_CHAT_DEPLOYMENT

The Azure tracer emits spans that comply with the GenAI semantic conventions,
including `invoke_agent` and nested tool execution spans.
"""

from __future__ import annotations

import logging
import os
import random
from datetime import datetime

import azure.identity
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from rich import print
from rich.logging import RichHandler

from langchain_azure_ai.callbacks.tracers import AzureAIOpenTelemetryTracer

load_dotenv(override=True)

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
LOGGER = logging.getLogger("weekend_planner")


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
    name="Weekend Planner Agent",
)


@tool
def get_weather(city: str, date: str) -> dict:
    """Returns weather data for a given city and date."""

    LOGGER.info("Getting weather for %s on %s", city, date)
    if random.random() < 0.05:
        return {"temperature": 72, "description": "Sunny"}
    return {"temperature": 60, "description": "Rainy"}


@tool
def get_activities(city: str, date: str) -> list:
    """Returns a list of activities for a given city and date."""

    LOGGER.info("Getting activities for %s on %s", city, date)
    return [
        {"name": "Hiking", "location": city},
        {"name": "Beach", "location": city},
        {"name": "Museum", "location": city},
    ]


@tool
def get_current_date() -> str:
    """Gets the current date from the system and returns as YYYY-MM-DD."""

    LOGGER.info("Getting current date")
    return datetime.now().strftime("%Y-%m-%d")


MODEL = _build_model()
AGENT = create_agent(
    model=MODEL,
    system_prompt=(
        "You help users plan their weekends and choose the best activities for the given weather. "
        "If an activity would be unpleasant in the weather, avoid suggesting it. "
        "Always include the relevant weekend date in your response."
    ),
    tools=[get_weather, get_activities, get_current_date],
)


def main() -> None:
    response = AGENT.invoke(
        {"messages": [{"role": "user", "content": "Hi, what can I do this weekend in San Francisco?"}]},
        config={"callbacks": [TRACER]},
    )
    latest_message = response["messages"][-1]
    print(latest_message.content)


if __name__ == "__main__":
    LOGGER.setLevel(logging.INFO)
    main()
