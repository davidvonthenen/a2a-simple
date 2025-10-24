# Airbnb remote agent

This package exposes the "Airbnb Agent" service used by the demo host application. It responds to A2A messages by generating fictional accommodation suggestions with an OpenAI chat model.

## Requirements

- Python dependencies from the repository root (`pip install -r requirements.txt` or `uv pip install -r requirements.txt`).
- `OPENAI_API_KEY` set in your shell or in a `.env` file (the module loads environment variables automatically).
- Optional overrides:
  - `OPENAI_AIRBNB_MODEL` – specific model for this agent. Defaults to `OPENAI_MODEL` or `gpt-5-nano`.
  - `APP_URL` – public URL to advertise inside the agent card. Defaults to the local host/port.

Example `.env` values:

```bash
OPENAI_API_KEY="sk-your-api-key"
OPENAI_AIRBNB_MODEL="gpt-5-nano"
APP_URL="http://localhost:10002"
```

## Running the service

Launch the agent on port `10002` (default):

```bash
uv run python -m src.airbnb_agent
# or
make airbnb_agent
```

The process starts a Starlette app via Uvicorn and serves the agent card at `http://localhost:10002/.well-known/a2a.json`.

## Disclaimer

Important: The sample code provided is for demonstration purposes and illustrates the mechanics of the Agent-to-Agent (A2A) protocol. When building production applications, it is critical to treat any agent operating outside of your direct control as a potentially untrusted entity.

All data received from an external agent—including but not limited to its AgentCard, messages, artifacts, and task statuses—should be handled as untrusted input. For example, a malicious agent could provide an AgentCard containing crafted data in its fields (e.g., description, name, skills.description). If this data is used without sanitization to construct prompts for a Large Language Model (LLM), it could expose your application to prompt injection attacks.  Failure to properly validate and sanitize this data before use can introduce security vulnerabilities into your application.

Developers are responsible for implementing appropriate security measures, such as input validation and secure handling of credentials to protect their systems and users.
