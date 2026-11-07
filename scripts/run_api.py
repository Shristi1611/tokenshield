"""Start the local API for the frontend. Keys stay in this process (read from .env).

  python scripts/run_api.py

Optional .env settings: TOKENSHIELD_API_PORT (default 8765) and TOKENSHIELD_ALLOWED_ORIGINS
(comma-separated; default is the usual Vite and Create React App dev servers).
"""

from __future__ import annotations

from pathlib import Path

from tokenshield.api import DEFAULT_ORIGINS, ApiService, make_server
from tokenshield.bifrost_client import load_env_file


def main() -> int:
    env = load_env_file(Path(".env"))
    try:
        service = ApiService.from_env(env)
    except ValueError as error:
        print(f"Setup failed: {error}")
        return 1
    try:
        port = int(env.get("TOKENSHIELD_API_PORT", "8765"))
    except ValueError:
        print("TOKENSHIELD_API_PORT must be a number")
        return 1
    raw_origins = env.get("TOKENSHIELD_ALLOWED_ORIGINS", "")
    origins = {item.strip() for item in raw_origins.split(",") if item.strip()}
    origins = origins or set(DEFAULT_ORIGINS)
    try:
        server = make_server(service, port=port, allowed_origins=origins)
    except OSError as error:
        print(f"Could not start the server: {error}")
        return 1

    print(f"tokenshield API on http://127.0.0.1:{port}  (Ctrl+C to stop)")
    print(f"allowed origins: {', '.join(sorted(origins))}")
    if not service.freshness_enabled:
        print("note: TAVILY_API_KEY is missing, so /api/freshness is disabled")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())