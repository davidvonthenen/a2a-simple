# Host agent UI

This module launches the Gradio-based host agent that coordinates the Airbnb and Weather remote agents.

## Requirements

- Shared project dependencies installed (`pip install -r requirements.txt` or `uv pip install -r requirements.txt`).
- Environment variables for OpenAI and remote agent discovery:

  ```bash
  OPENAI_API_KEY="sk-your-api-key"
  # Optional overrides
  OPENAI_ROUTER_MODEL="gpt-5-nano"
  AIR_AGENT_URL="http://localhost:10002"
  WEA_AGENT_URL="http://localhost:10001"
  ```

  Additional optional variables:
  - `OPENAI_MODEL` – fallback when a service-specific model override is not provided.
  - `APP_URL` – publish an externally reachable host URL inside the UI (defaults to the local address).

## Running the UI

Start the Gradio application on port `11000`:

```bash
uv run python -m src.host_agent
# or
make host_agent
```

Visit <http://127.0.0.1:11000/> to chat with the orchestration agent. The host automatically fetches the agent cards from the configured remote agent URLs when it starts.
