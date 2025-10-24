# Weather remote agent

The Weather agent is an A2A-compatible HTTP service that fabricates realistic weather forecasts using an OpenAI chat model. It is consumed by the host routing agent included in this repository.

## Requirements

- Install the shared project dependencies (`pip install -r requirements.txt` or `uv pip install -r requirements.txt`).
- Provide an `OPENAI_API_KEY` via environment variable or a `.env` file in the repository root.
- Optional variables:
  - `OPENAI_WEATHER_MODEL` – override the OpenAI chat model for this service.
  - `APP_URL` – advertise an externally reachable URL in the agent card (defaults to the local host/port).

Example environment configuration:

```bash
OPENAI_API_KEY="sk-your-api-key"
OPENAI_WEATHER_MODEL="gpt-5-nano"
APP_URL="http://localhost:10001"
```

## Running the service

Start the HTTP server on port `10001` (default):

```bash
uv run python -m src.weather_agent
# or
make weather_agent
```

The service publishes its agent card at `http://localhost:10001/.well-known/a2a.json` and streams task updates back to callers.
