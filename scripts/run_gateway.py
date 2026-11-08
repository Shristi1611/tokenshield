"""Start the TokenShield gateway: an OpenAI-compatible endpoint in front of Bifrost.

  python scripts/run_gateway.py

Point an OpenAI-style client at http://127.0.0.1:8766/v1 (the gateway adds its own virtual key,
so the client sends none). Optionally send an X-Tokenshield-Session header to track a
conversation's token total. Keys stay in this process (read from .env).

Optional .env settings:
  TOKENSHIELD_GATEWAY_PORT    default 8766
  TOKENSHIELD_SESSION_BUDGET  tokens per session; the next request after a session exceeds it
                              is refused. Unset means no limit.
"""

from __future__ import annotations

from pathlib import Path

from tokenshield.bifrost_client import load_env_file
from tokenshield.gateway import DEFAULT_PORT, GatewayService, make_gateway_server


def main() -> int:
    env = load_env_file(Path(".env"))
    try:
        service = GatewayService.from_env(env)
        port = int(env.get("TOKENSHIELD_GATEWAY_PORT", str(DEFAULT_PORT)))
    except ValueError as error:
        print(f"Setup failed: {error}")
        return 1
    try:
        server = make_gateway_server(service, port=port)
    except OSError as error:
        print(f"Could not start the server: {error}")
        return 1
    budget = service.ledger.budget
    print(f"tokenshield gateway on http://127.0.0.1:{port}/v1  (Ctrl+C to stop)")
    print(f"session budget: {'none' if budget is None else f'{budget:g} tokens'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())