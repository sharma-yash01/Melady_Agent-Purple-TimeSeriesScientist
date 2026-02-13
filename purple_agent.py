"""
Time Series Purple Agent - TimeSeriesScientist (LangChain) Implementation

This is a TimeSeriesScientist-based purple agent for the AgentX Time Series benchmark.
It receives prompts from the green agent and responds with JSON-formatted answers
or forecasts, using LangChain's ChatOpenAI (TSci's LLM backbone) for inference.

This agent is designed to be:
1. LangChain-integrated: Uses langchain_openai.ChatOpenAI (same as TSci's agent layer)
2. Tool-free: Tests raw model performance on timeseries tasks
3. Extensible: Easy to add tools, change prompts, or swap models
4. Best-effort: Optimized prompt structure for timeseries performance

Key difference from base: Uses langchain_openai.ChatOpenAI instead of raw
openai.OpenAI client, enabling LangChain's message abstraction, async invocation,
and provider compatibility (the same stack TimeSeriesScientist uses).
"""

import argparse
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import uvicorn
from dotenv import load_dotenv

load_dotenv()

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message

from prompt_engine import build_system_prompt, extract_task_info, format_response
from prompt_processor import post_process_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("purple_agent")

# ===========================
# LLM Configuration
# ===========================
DEFAULT_MODEL = os.getenv("PURPLE_AGENT_MODEL", "openai/gpt-4o")
DEFAULT_TEMPERATURE = float(os.getenv("PURPLE_AGENT_TEMPERATURE", "0.0"))
DEFAULT_MAX_TOKENS = int(os.getenv("PURPLE_AGENT_MAX_TOKENS", "4000"))


# ===========================
# Model Name Utilities
# ===========================

def normalize_model_for_openrouter(model: str) -> str:
    """Ensure model has provider prefix for OpenRouter."""
    if "/" not in model:
        if model.startswith("gpt-") or model.startswith("o1-"):
            return f"openai/{model}"
        elif model.startswith("claude-"):
            return f"anthropic/{model}"
        elif model.startswith("gemini-"):
            return f"google/{model}"
        raise ValueError(f"Cannot auto-prefix '{model}' for OpenRouter")
    return model


def strip_provider_prefix(model: str) -> str:
    """Strip provider prefix for direct API calls."""
    return model.split("/", 1)[1] if "/" in model else model


def _create_langchain_llm(model: str, temperature: float, max_tokens: int):
    """
    Create a LangChain ChatOpenAI instance based on environment configuration.

    This is the same LLM layer used by TimeSeriesScientist's agents
    (analysis_agent, forecast_agent, etc.).

    Supports OpenAI and OpenRouter (OpenAI-compatible endpoint).
    """
    from langchain_openai import ChatOpenAI

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if openrouter_key:
        normalized_model = normalize_model_for_openrouter(model)
        if normalized_model != model:
            logger.warning(
                f"Bare model name '{model}' detected. "
                f"Auto-prefixing to '{normalized_model}' for OpenRouter."
            )

        logger.info(f"Creating LangChain ChatOpenAI for OpenRouter: model={normalized_model}")

        return ChatOpenAI(
            model=normalized_model,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=openrouter_key,
            openai_api_base="https://openrouter.ai/api/v1",
        )

    elif openai_key:
        normalized_model = strip_provider_prefix(model)
        if normalized_model != model:
            logger.warning(
                f"Provider-prefixed model '{model}' detected. "
                f"Stripping to '{normalized_model}' for OpenAI API."
            )

        logger.info(f"Creating LangChain ChatOpenAI for OpenAI: model={normalized_model}")

        return ChatOpenAI(
            model=normalized_model,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=openai_key,
        )

    else:
        raise ValueError(
            "No API key found. Set either OPENAI_API_KEY or OPENROUTER_API_KEY environment variable."
        )


def prepare_agent_card(url: str):
    """Create the agent card for the TSci purple agent."""
    skill = AgentSkill(
        id="timeseries_forecasting",
        name="Time Series Forecasting",
        description="Performs time series forecasting and analysis using TimeSeriesScientist (LangChain) framework",
        tags=["benchmark", "timeseries", "forecasting", "langchain", "tsci"],
        examples=[],
    )
    return AgentCard(
        name="Melady_Agent-TS-TSci-Purple",
        description="TimeSeriesScientist-based purple agent for Time Series benchmark",
        url=url,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
    )


class PurpleAgentExecutor(AgentExecutor):
    """
    Executor for the TimeSeriesScientist purple agent.

    Handles A2A protocol messages from the green agent and responds with
    JSON-formatted answers or forecasts. Uses LangChain's ChatOpenAI
    (TSci's LLM backbone) for inference.
    """

    def __init__(self, model: str = None, temperature: float = None, max_tokens: int = None):
        self.ctx_id_to_messages: Dict[str, List[Dict[str, Any]]] = {}
        self.model_name = model or DEFAULT_MODEL
        self.temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
        self.max_tokens = max_tokens or DEFAULT_MAX_TOKENS

        # Create LangChain ChatOpenAI (replaces get_llm_client() from base)
        self.llm = _create_langchain_llm(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        logger.info(
            f"Initialized PurpleAgentExecutor (TSci/LangChain): model={self.model_name}, "
            f"temperature={self.temperature}, max_tokens={self.max_tokens}"
        )

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """
        Execute a task from the green agent.

        Routes prompts through LangChain's ChatOpenAI.ainvoke() which handles
        message formatting and async API execution (the same LLM layer
        TimeSeriesScientist uses internally).
        """
        from langchain_core.messages import SystemMessage, HumanMessage

        raw_user_input = context.get_user_input()
        logger.info(f"Received task (context_id={context.context_id[:8]}...): {raw_user_input[:200]}...")

        task_info = extract_task_info(raw_user_input)
        task_type = task_info.get("task_type", "unknown")

        processed_prompt = post_process_prompt(raw_user_input, task_info)
        system_prompt = build_system_prompt(task_type, task_info)

        # Call LangChain ChatOpenAI
        try:
            logger.debug(f"Calling LangChain ChatOpenAI: {self.model_name}")

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=processed_prompt),
            ]

            # ainvoke is async and returns an AIMessage with .content as string
            response = await self.llm.ainvoke(messages)
            response_text = response.content

            if not response_text:
                raise ValueError("LangChain ChatOpenAI returned empty response")

            logger.info(f"LLM response length: {len(response_text)} chars")
            logger.debug(f"LLM response preview: {response_text[:200]}...")

            formatted_response = format_response(response_text, task_type, task_info)

        except Exception as e:
            logger.error(f"LLM error: {e}", exc_info=True)
            if task_type in ["T1", "T3"]:
                formatted_response = json.dumps({"answers": {}, "error": str(e)})
            else:
                formatted_response = json.dumps({"forecasts": [], "error": str(e)})

        await event_queue.enqueue_event(
            new_agent_text_message(formatted_response, context_id=context.context_id)
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        """Handle task cancellation."""
        logger.warning(f"Cancel requested for context {context.context_id}")


def main():
    """Main entrypoint for the TSci purple agent server."""
    parser = argparse.ArgumentParser(description="Run the Time Series Purple Agent (TSci/LangChain).")
    parser.add_argument("--host", type=str, default=os.getenv("AGENT_HOST", "0.0.0.0"), help="Host to bind")
    parser.add_argument("--port", type=int, default=int(os.getenv("AGENT_PORT", "9023")), help="Port to bind")
    parser.add_argument("--card-url", type=str, default=os.getenv("CARD_URL"), help="External URL for agent card")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="LLM model identifier")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max tokens in response")
    args = parser.parse_args()

    logger.info(f"Starting TSci Time Series Purple Agent on {args.host}:{args.port}")
    logger.info(f"Model: {args.model}, Temperature: {args.temperature}, Max Tokens: {args.max_tokens}")

    card_url = args.card_url or f"http://{args.host}:{args.port}/"
    card = prepare_agent_card(card_url)

    executor = PurpleAgentExecutor(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )

    logger.info(f"Agent card available at: {card_url}.well-known/agent.json")

    uvicorn.run(
        app.build(),
        host=args.host,
        port=args.port,
        timeout_keep_alive=300,
    )


if __name__ == "__main__":
    main()
