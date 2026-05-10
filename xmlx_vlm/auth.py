"""
HTTP authentication dependency.

A single FastAPI ``Depends``-compatible function that validates the
``Authorization: Bearer <token>`` header against ``XMLX_VLM_API_KEY``.
When the env var is not set, every request is allowed through (open mode).
"""
import os

from fastapi import HTTPException, Request


def verify_api_key(request: Request) -> None:
    """FastAPI dependency: verify Bearer token API key.

    Reading from env var rather than a module-level constant because
    ``uvicorn.run("module:app")`` re-imports the module on reload, which
    would reset any global set at startup.  Env vars survive re-imports.
    """
    key = os.environ.get("XMLX_VLM_API_KEY")
    if not key:
        return
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer ") or auth.startswith("bearer "):
        token = auth[7:]
    else:
        token = auth
    if token != key:
        raise HTTPException(status_code=401, detail="Invalid API key")
