from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from yolorag.tools.base import ToolCallRequest, ToolCallResult


logger = logging.getLogger(__name__)
_warned_invalid_env_values: set[tuple[str, str]] = set()

DEFAULT_GITHUB_MCP_URL = "https://api.githubcopilot.com/mcp/"
DEFAULT_GITHUB_MCP_ALLOWED_REPOSITORIES = ("ultralytics/ultralytics",)
DEFAULT_GITHUB_MCP_TOOLSETS = "repos,issues,pull_requests,actions"


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    transport: str = "stdio"
    command: str | None = None
    url: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    sse_read_timeout_seconds: float = 300.0
    terminate_on_close: bool = True
    allowed_repositories: tuple[str, ...] = ()


class MCPToolProvider:
    def __init__(self, servers: list[MCPServerConfig]) -> None:
        self.servers = servers
        self._tools: list[MCPServerTool] | None = None

    @classmethod
    def from_env(cls, env_name: str = "YOLORAG_MCP_SERVERS") -> MCPToolProvider | None:
        raw = os.getenv(env_name)
        servers = _custom_server_configs_from_env(raw, env_name=env_name)
        if servers:
            return cls(servers)

        servers = _default_server_configs_from_env()
        if not servers:
            return None
        return cls(servers)

    async def tools(self) -> list["MCPServerTool"]:
        if self._tools is not None:
            return self._tools

        discovered: list[MCPServerTool] = []
        for server in self.servers:
            try:
                discovered.extend(await _discover_server_tools(server))
            except ModuleNotFoundError:
                logger.warning(
                    "MCP server %s is configured, but the optional 'mcp' package is not installed.",
                    server.name,
                )
            except Exception:
                logger.warning(
                    "Failed to discover MCP tools for server %s.",
                    server.name,
                    exc_info=True,
                )
        self._tools = discovered
        return discovered


class MCPServerTool:
    def __init__(
        self,
        *,
        server: MCPServerConfig,
        exposed_name: str,
        original_name: str,
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        self.server = server
        self.name = exposed_name
        self.original_name = original_name
        self.description = description
        self.parameters = parameters

    async def call(self, request: ToolCallRequest) -> ToolCallResult:
        try:
            arguments = _repo_scoped_arguments(
                self.server,
                self.original_name,
                request.arguments,
            )
            result = await _call_server_tool(
                self.server,
                self.original_name,
                arguments,
            )
        except Exception as exc:
            logger.warning("MCP tool %s failed.", self.name, exc_info=True)
            return ToolCallResult(
                name=self.name,
                output={"error": f"{type(exc).__name__}: {exc}"},
                error=f"{type(exc).__name__}: {exc}",
            )
        return ToolCallResult(name=self.name, output=result, cost_hint="mcp")


async def _discover_server_tools(server: MCPServerConfig) -> list[MCPServerTool]:
    from mcp.client.session import ClientSession

    async with _server_connection(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()

    tools = []
    for tool in result.tools:
        original_name = tool.name
        if not _should_expose_tool(server, original_name, tool.inputSchema):
            continue
        exposed_name = _tool_name(server.name, original_name)
        description = tool.description or f"MCP tool {original_name} from {server.name}."
        if server.allowed_repositories:
            description = (
                f"{description} This MCP connection is restricted to: "
                f"{', '.join(server.allowed_repositories)}."
            )
        tools.append(
            MCPServerTool(
                server=server,
                exposed_name=exposed_name,
                original_name=original_name,
                description=description,
                parameters=tool.inputSchema or {"type": "object", "properties": {}},
            )
        )
    return tools


async def _call_server_tool(
    server: MCPServerConfig,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    from mcp.client.session import ClientSession

    async with _server_connection(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)

    return {
        "content": [_content_block(block) for block in result.content],
        "structured_content": getattr(result, "structuredContent", None),
        "is_error": getattr(result, "isError", False),
    }


@asynccontextmanager
async def _server_connection(server: MCPServerConfig) -> AsyncIterator[tuple[Any, Any]]:
    if server.transport == "stdio":
        from mcp.client.stdio import StdioServerParameters, stdio_client

        params = _stdio_params(StdioServerParameters, server)
        async with stdio_client(params) as (read, write):
            yield read, write
        return

    if server.transport == "streamable_http":
        import httpx
        from mcp.client.streamable_http import streamable_http_client
        from mcp.shared._httpx_utils import create_mcp_http_client

        if server.url is None:
            raise ValueError(f"MCP HTTP server {server.name!r} is missing a URL.")

        timeout = httpx.Timeout(
            server.timeout_seconds,
            read=server.sse_read_timeout_seconds,
        )
        async with create_mcp_http_client(
            headers=server.headers,
            timeout=timeout,
        ) as client:
            async with streamable_http_client(
                server.url,
                http_client=client,
                terminate_on_close=server.terminate_on_close,
            ) as (read, write, _get_session_id):
                yield read, write
        return

    raise ValueError(f"Unsupported MCP transport {server.transport!r}.")


def _stdio_params(stdio_params_cls: type, server: MCPServerConfig) -> Any:
    if server.command is None:
        raise ValueError(f"MCP stdio server {server.name!r} is missing a command.")
    return stdio_params_cls(
        command=server.command,
        args=server.args,
        env={**os.environ, **server.env},
    )


def _content_block(block: Any) -> Any:
    text = getattr(block, "text", None)
    if text is not None:
        return {"type": getattr(block, "type", "text"), "text": text}
    if hasattr(block, "model_dump"):
        return block.model_dump()
    return str(block)


def _parse_server_configs(data: Any) -> list[MCPServerConfig]:
    if isinstance(data, dict):
        items = [
            {"name": name, **value}
            for name, value in data.items()
            if isinstance(value, dict)
        ]
    elif isinstance(data, list):
        items = data
    else:
        return []

    configs = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        transport = _normalize_transport(item)
        command = _optional_str(item.get("command"))
        url = _optional_str(item.get("url"))

        if transport == "stdio" and not command:
            continue
        if transport == "streamable_http" and not url:
            continue
        if transport not in {"stdio", "streamable_http"}:
            logger.warning("Ignoring MCP server %s with unsupported transport %s.", name, transport)
            continue

        args = [str(arg) for arg in item.get("args", [])]
        env = _string_dict(item.get("env", {}), expand_env=True)
        headers = _string_dict(item.get("headers", {}), expand_env=True)
        configs.append(
            MCPServerConfig(
                name=name,
                transport=transport,
                command=command,
                url=url,
                args=args,
                env=env,
                headers=headers,
                timeout_seconds=_float_config(
                    item,
                    keys=("timeout_seconds", "timeout"),
                    default=30.0,
                ),
                sse_read_timeout_seconds=_float_config(
                    item,
                    keys=("sse_read_timeout_seconds", "sse_read_timeout"),
                    default=300.0,
                ),
                terminate_on_close=_bool_config(
                    item.get("terminate_on_close"),
                    default=True,
                ),
                allowed_repositories=_repository_list(
                    item.get("allowed_repositories")
                    or item.get("allowedRepositories")
                    or item.get("repositories")
                ),
            )
        )
    return configs


def _custom_server_configs_from_env(raw: str | None, *, env_name: str) -> list[MCPServerConfig]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        warning_key = (env_name, raw)
        if warning_key not in _warned_invalid_env_values:
            logger.warning(
                "%s must be valid JSON; ignoring custom MCP server config: %s",
                env_name,
                exc,
            )
            _warned_invalid_env_values.add(warning_key)
        logger.debug("%s invalid value: %r", env_name, raw)
        return []

    servers = _parse_server_configs(data)
    if not servers:
        logger.warning("%s did not define any MCP servers.", env_name)
    return servers


def _default_server_configs_from_env() -> list[MCPServerConfig]:
    github_token = os.getenv("GITHUB_MCP_TOKEN") or os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    if not github_token:
        return []
    return [
        MCPServerConfig(
            name="github",
            transport="streamable_http",
            url=DEFAULT_GITHUB_MCP_URL,
            headers={
                "Authorization": f"Bearer {github_token}",
                "X-MCP-Toolsets": DEFAULT_GITHUB_MCP_TOOLSETS,
                "X-MCP-Readonly": "true",
            },
            allowed_repositories=DEFAULT_GITHUB_MCP_ALLOWED_REPOSITORIES,
        )
    ]


def _normalize_transport(item: dict[str, Any]) -> str:
    raw = str(item.get("transport") or item.get("type") or "").strip().lower()
    if not raw:
        raw = "streamable_http" if item.get("url") else "stdio"
    raw = raw.replace("-", "_")
    if raw in {"http", "streamablehttp"}:
        return "streamable_http"
    return raw


def _optional_str(value: Any) -> str | None:
    text = str(value or "").strip()
    return _expand_env_refs(text) if text else None


def _string_dict(value: Any, *, expand_env: bool = False) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    result = {}
    for key, item in value.items():
        text = str(item)
        result[str(key)] = _expand_env_refs(text) if expand_env else text
    return result


def _repository_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw_items = value.split(",")
    elif isinstance(value, list):
        raw_items = value
    else:
        return ()

    repositories = []
    for item in raw_items:
        if isinstance(item, dict):
            owner = str(item.get("owner") or "").strip()
            repo = str(item.get("repo") or item.get("name") or "").strip()
            raw = f"{owner}/{repo}" if owner and repo else ""
        else:
            raw = str(item or "").strip()
        normalized = _normalize_repo(raw)
        if normalized and normalized not in repositories:
            repositories.append(normalized)
    return tuple(repositories)


def _float_config(
    item: dict[str, Any],
    *,
    keys: tuple[str, ...],
    default: float,
) -> float:
    for key in keys:
        if key not in item:
            continue
        try:
            return float(item[key])
        except (TypeError, ValueError):
            return default
    return default


def _bool_config(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _expand_env_refs(value: str) -> str:
    return re.sub(
        r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}",
        lambda match: os.getenv(match.group(1), match.group(0)),
        value,
    )


def _repo_scoped_arguments(
    server: MCPServerConfig,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    if not server.allowed_repositories or not _is_github_server(server):
        return arguments

    scoped_arguments = dict(arguments)
    allowed = set(server.allowed_repositories)
    _validate_search_query_scope(scoped_arguments, allowed)

    owner = str(scoped_arguments.get("owner") or "").strip()
    repo = str(scoped_arguments.get("repo") or "").strip()
    if owner or repo:
        repository = _normalize_repo(f"{owner}/{repo}")
        if repository not in allowed:
            raise ValueError(
                f"GitHub MCP access is restricted to {', '.join(server.allowed_repositories)}."
            )
        return scoped_arguments

    if _is_scopable_search_tool(tool_name) and len(server.allowed_repositories) == 1:
        allowed_owner, allowed_repo = server.allowed_repositories[0].split("/", 1)
        scoped_arguments["owner"] = allowed_owner
        scoped_arguments["repo"] = allowed_repo
        return scoped_arguments

    raise ValueError(
        f"GitHub MCP access is restricted to {', '.join(server.allowed_repositories)}; "
        f"tool {tool_name!r} must target an allowed owner and repo."
    )


def _should_expose_tool(
    server: MCPServerConfig,
    tool_name: str,
    input_schema: dict[str, Any] | None,
) -> bool:
    if not server.allowed_repositories or not _is_github_server(server):
        return True
    if tool_name == "search_repositories":
        return False

    properties = (input_schema or {}).get("properties") or {}
    if "owner" in properties and "repo" in properties:
        return True
    return _is_scopable_search_tool(tool_name)


def _is_github_server(server: MCPServerConfig) -> bool:
    if server.name.lower() == "github":
        return True
    return bool(server.url and "githubcopilot.com/mcp" in server.url)


def _is_scopable_search_tool(tool_name: str) -> bool:
    return tool_name in {
        "search_code",
        "search_issues",
        "search_pull_requests",
    }


def _validate_search_query_scope(arguments: dict[str, Any], allowed: set[str]) -> None:
    query = arguments.get("query")
    if not isinstance(query, str):
        return

    for match in re.finditer(r"(?:^|\s)repo:([^\s]+)", query, flags=re.IGNORECASE):
        repository = _normalize_repo(match.group(1).strip("\"'"))
        if repository not in allowed:
            raise ValueError(
                f"GitHub MCP search is restricted to {', '.join(sorted(allowed))}."
            )

    if re.search(r"(?:^|\s)(?:org|user):[^\s]+", query, flags=re.IGNORECASE):
        raise ValueError(
            "GitHub MCP search cannot use org: or user: qualifiers when repository "
            "allowlisting is enabled."
        )


def _normalize_repo(value: str) -> str:
    text = value.strip().strip("/")
    if text.count("/") != 1:
        return ""
    owner, repo = (part.strip().lower() for part in text.split("/", 1))
    if not owner or not repo:
        return ""
    return f"{owner}/{repo}"


def _tool_name(server_name: str, tool_name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", f"mcp_{server_name}_{tool_name}")
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized[:64]
