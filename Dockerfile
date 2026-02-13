FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AGENT_HOST=0.0.0.0 \
    AGENT_PORT=9023 \
    PURPLE_AGENT_MODEL=openai/gpt-4o \
    PURPLE_AGENT_TEMPERATURE=0.0 \
    PURPLE_AGENT_MAX_TOKENS=4000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies (lightweight - LangChain stack only)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy agent code
COPY purple_agent.py prompt_engine.py prompt_processor.py agent_card.toml ./

COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

LABEL org.opencontainers.image.source="https://github.com/sharma-yash01/Melady_Agent-TS-TSci-Purple" \
      org.opencontainers.image.description="TimeSeriesScientist Time Series Purple Agent (A2A-compliant)" \
      org.opencontainers.image.licenses="MIT"

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["--host", "0.0.0.0", "--port", "9023"]
