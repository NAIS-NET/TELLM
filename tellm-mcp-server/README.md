# TELLM MCP Server
This MCP Server is developed based on  [davidlin2k/onos-mcp-server](https://github.com/davidlin2k/onos-mcp-server), extending its capabilities to support TELLM APIs for enhanced traffic engineering (TE) control.

## Overview
The TELLM MCP Server is an implementation of the Model Context Protocol (MCP) designed to provide seamless access to TELLM APIs for LLM agents. It facilitates efficient traffic engineering (TE) control, enabling intelligent network management and optimization.


## Features
The server provides access to TELLM API endpoints, including:

- Asynchronous Control of TE-Head in TELLM
- Network Topology Information
- Monitoring of Traffic Engineering (TE) Metrics
- ONOS MCP Application Integration (e.g., Flow Rules, policys)

## Requirements

- Python 3.7+
- [uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) for dependency management
- Running ONOS controller
- httpx library
- mcp library

## Configuration

Configure the server using environment variables:

- `TE_HEAD_API_BASE`: Base URL for TE-Head (default: http://localhost:5001)
- `ONOS_API_BASE`: Base URL for ONOS API (default: http://localhost:8181/onos/v1)
- `ONOS_USERNAME`: Username for ONOS API authentication (default: onos)
- `ONOS_PASSWORD`: Password for ONOS API authentication (default: rocks)

## Usage with Claude Desktop

```json
{
  "mcpServers": {
    "onos": {
      "command": "uv",
      "args": [
        "--directory",
        "parent_of_servers_repo/src/tellm-mcp-server",
        "run",
        "server.py"
      ],
      "env": {
        "TE_HEAD_API_BASE": "http://localhost:5001",
        "ONOS_API_BASE": "http://localhost:8181/onos/v1",
        "ONOS_USERNAME": "onos",
        "ONOS_PASSWORD": "rocks"
      }
    }
  }
}
```