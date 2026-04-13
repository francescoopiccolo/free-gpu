from __future__ import annotations

from fastapi.responses import JSONResponse
from starlette.applications import Starlette
from starlette.requests import Request

from .mcp_server import create_mcp


def create_http_app() -> Starlette:
    mcp = create_mcp(host="0.0.0.0")

    @mcp.custom_route("/", methods=["GET"], include_in_schema=False)
    async def root(_: Request) -> JSONResponse:
        return JSONResponse(
            {
                "name": "free-gpu",
                "service": "mcp-http",
                "transport": "streamable-http",
                "mcp_path": "/mcp",
                "notes": [
                    "Use /mcp as the MCP endpoint.",
                    "This hosted endpoint is optional; the package also works locally via `free-gpu-mcp`.",
                ],
            }
        )

    @mcp.custom_route("/health", methods=["GET"], include_in_schema=False)
    async def health(_: Request) -> JSONResponse:
        return JSONResponse({"ok": True})

    return mcp.streamable_http_app()


app = create_http_app()
