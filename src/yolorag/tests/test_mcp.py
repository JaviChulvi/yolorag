from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from yolorag.tools.mcp import (
    MCPServerConfig,
    MCPToolProvider,
    _parse_server_configs,
    _repo_scoped_arguments,
    _should_expose_tool,
)


class MCPServerConfigTests(unittest.TestCase):
    def test_default_github_mcp_config_comes_from_token_env(self) -> None:
        with patch.dict(os.environ, {"GITHUB_MCP_TOKEN": "test-token"}, clear=True):
            provider = MCPToolProvider.from_env()

        self.assertIsNotNone(provider)
        assert provider is not None
        self.assertEqual(len(provider.servers), 1)
        config = provider.servers[0]
        self.assertEqual(config.name, "github")
        self.assertEqual(config.transport, "streamable_http")
        self.assertEqual(config.url, "https://api.githubcopilot.com/mcp/")
        self.assertEqual(config.headers["Authorization"], "Bearer test-token")
        self.assertEqual(config.headers["X-MCP-Toolsets"], "repos,issues,pull_requests,actions")
        self.assertEqual(config.headers["X-MCP-Readonly"], "true")
        self.assertEqual(config.allowed_repositories, ("ultralytics/ultralytics",))

    def test_invalid_custom_mcp_json_falls_back_to_default_github_config(self) -> None:
        from yolorag.tools import mcp

        mcp._warned_invalid_env_values.clear()
        try:
            with patch.dict(
                os.environ,
                {
                    "GITHUB_MCP_TOKEN": "test-token",
                    "YOLORAG_MCP_SERVERS": "[{name:\"github\"}]",
                },
                clear=True,
            ):
                provider = MCPToolProvider.from_env()
        finally:
            mcp._warned_invalid_env_values.clear()

        self.assertIsNotNone(provider)
        assert provider is not None
        self.assertEqual(len(provider.servers), 1)
        self.assertEqual(provider.servers[0].name, "github")
        self.assertEqual(provider.servers[0].headers["Authorization"], "Bearer test-token")

    def test_valid_custom_mcp_json_overrides_default_github_config(self) -> None:
        with patch.dict(
            os.environ,
            {
                "GITHUB_MCP_TOKEN": "test-token",
                "YOLORAG_MCP_SERVERS": '{"local":{"command":"uvx","args":["example"]}}',
            },
            clear=True,
        ):
            provider = MCPToolProvider.from_env()

        self.assertIsNotNone(provider)
        assert provider is not None
        self.assertEqual(len(provider.servers), 1)
        self.assertEqual(provider.servers[0].name, "local")
        self.assertEqual(provider.servers[0].transport, "stdio")
        self.assertEqual(provider.servers[0].command, "uvx")

    def test_parses_streamable_http_server_config(self) -> None:
        with patch.dict(os.environ, {"GITHUB_MCP_TOKEN": "test-token"}):
            configs = _parse_server_configs(
                [
                    {
                        "name": "github",
                        "type": "http",
                        "url": "https://api.githubcopilot.com/mcp/",
                        "headers": {
                            "Authorization": "Bearer ${GITHUB_MCP_TOKEN}",
                            "X-MCP-Toolsets": "repos,issues,pull_requests,actions",
                            "X-MCP-Readonly": "true",
                        },
                        "allowed_repositories": ["Ultralytics/Ultralytics"],
                        "timeout": 7,
                        "sse_read_timeout": 11,
                        "terminate_on_close": "false",
                    }
                ]
            )

        self.assertEqual(len(configs), 1)
        config = configs[0]
        self.assertEqual(config.name, "github")
        self.assertEqual(config.transport, "streamable_http")
        self.assertEqual(config.url, "https://api.githubcopilot.com/mcp/")
        self.assertIsNone(config.command)
        self.assertEqual(config.headers["Authorization"], "Bearer test-token")
        self.assertEqual(config.headers["X-MCP-Readonly"], "true")
        self.assertEqual(config.timeout_seconds, 7)
        self.assertEqual(config.sse_read_timeout_seconds, 11)
        self.assertFalse(config.terminate_on_close)
        self.assertEqual(config.allowed_repositories, ("ultralytics/ultralytics",))

    def test_parses_legacy_stdio_server_config(self) -> None:
        configs = _parse_server_configs(
            {
                "local": {
                    "command": "uvx",
                    "args": ["example-mcp-server"],
                    "env": {"EXAMPLE_TOKEN": "${EXAMPLE_TOKEN}"},
                }
            }
        )

        self.assertEqual(len(configs), 1)
        config = configs[0]
        self.assertEqual(config.name, "local")
        self.assertEqual(config.transport, "stdio")
        self.assertEqual(config.command, "uvx")
        self.assertEqual(config.args, ["example-mcp-server"])

    def test_repository_allowlist_accepts_allowed_repo(self) -> None:
        server = MCPServerConfig(
            name="github",
            allowed_repositories=("ultralytics/ultralytics",),
        )

        arguments = _repo_scoped_arguments(
            server,
            "get_file_contents",
            {"owner": "ultralytics", "repo": "ultralytics", "path": "README.md"},
        )

        self.assertEqual(arguments["owner"], "ultralytics")
        self.assertEqual(arguments["repo"], "ultralytics")

    def test_repository_allowlist_rejects_disallowed_repo(self) -> None:
        server = MCPServerConfig(
            name="github",
            allowed_repositories=("ultralytics/ultralytics",),
        )

        with self.assertRaises(ValueError):
            _repo_scoped_arguments(
                server,
                "get_file_contents",
                {"owner": "pytorch", "repo": "pytorch", "path": "README.md"},
            )

    def test_repository_allowlist_scopes_search_tools(self) -> None:
        server = MCPServerConfig(
            name="github",
            allowed_repositories=("ultralytics/ultralytics",),
        )

        arguments = _repo_scoped_arguments(
            server,
            "search_issues",
            {"query": "export bug"},
        )

        self.assertEqual(arguments["owner"], "ultralytics")
        self.assertEqual(arguments["repo"], "ultralytics")

    def test_repository_allowlist_rejects_query_repo_qualifier_escape(self) -> None:
        server = MCPServerConfig(
            name="github",
            allowed_repositories=("ultralytics/ultralytics",),
        )

        with self.assertRaises(ValueError):
            _repo_scoped_arguments(
                server,
                "search_issues",
                {"query": "repo:pytorch/pytorch export bug"},
            )

    def test_repository_allowlist_hides_global_repository_search(self) -> None:
        server = MCPServerConfig(
            name="github",
            allowed_repositories=("ultralytics/ultralytics",),
        )

        self.assertFalse(
            _should_expose_tool(
                server,
                "search_repositories",
                {"type": "object", "properties": {"query": {"type": "string"}}},
            )
        )


if __name__ == "__main__":
    unittest.main()
