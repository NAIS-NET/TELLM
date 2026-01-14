from typing import Any, Dict, Optional
from mcp.server.fastmcp import FastMCP
from tellm_mcp_server.api_client import make_onos_request


async def get_all_policys(detail: bool = False) -> str:
    """Gets all policys in the system.

    Args:
        detail: Flag to return full details of policys in list

    Returns array containing all the policys in the system.
    """
    try:
        params = {"detail": str(detail).lower()}
        policys = await make_onos_request("get", "/policys", params=params)
        return str(policys)
    except Exception as e:
        return f"Error retrieving policys: {str(e)}"


async def get_policy(appId: str, key: str) -> str:
    """Gets policy by application ID and key.

    Args:
        appId: Application identifier
        key: policy key

    Returns details of the specified policy.
    """
    try:
        policy = await make_onos_request("get", f"/policys/{appId}/{key}")
        return str(policy)
    except Exception as e:
        return f"Error retrieving policy {key} for app {appId}: {str(e)}"


async def get_policys_by_application(appId: str, detail: bool = False) -> str:
    """Gets policys by application.

    Args:
        appId: Application identifier
        detail: Flag to return full details of policys in list

    Returns the policys specified by the application ID.
    """
    try:
        params = {"detail": str(detail).lower()}
        policys = await make_onos_request(
            "get", f"/policys/application/{appId}", params=params
        )
        return str(policys)
    except Exception as e:
        return f"Error retrieving policys for app {appId}: {str(e)}"


async def get_policy_installables(appId: str, key: str) -> str:
    """Gets policy installables by application ID and key.

    Args:
        appId: Application identifier
        key: policy key
    """
    try:
        installables = await make_onos_request(
            "get", f"/policys/installables/{appId}/{key}"
        )
        return str(installables)
    except Exception as e:
        return f"Error retrieving installables for policy {key}, app {appId}: {str(e)}"


async def get_policy_related_flows(appId: str, key: str) -> str:
    """Gets all related flow entries created by a particular policy.

    Args:
        appId: Application identifier
        key: policy key

    Returns all flow entries of the specified policy.
    """
    try:
        flows = await make_onos_request("get", f"/policys/relatedflows/{appId}/{key}")
        return str(flows)
    except Exception as e:
        return f"Error retrieving flows for policy {key}, app {appId}: {str(e)}"


async def get_policys_summary() -> str:
    """Gets summary of all policys.

    Returns a summary of the policys in the system.
    """
    try:
        summary = await make_onos_request("get", "/policys/minisummary")
        return str(summary)
    except Exception as e:
        return f"Error retrieving policys summary: {str(e)}"


async def submit_policy(policy_data: Dict[str, Any]) -> str:
    """Submits a new policy.

    Args:
        policy_data: policy configuration including type, priority, constraints, etc.

    Creates and submits policy from the supplied JSON data.
    """
    try:
        result = await make_onos_request("post", "/policys", json=policy_data)
        return f"policy submitted successfully: {result}"
    except Exception as e:
        return f"Error submitting policy: {str(e)}"


async def withdraw_policy(appId: str, key: str) -> str:
    """Withdraws an policy.

    Args:
        appId: Application identifier
        key: policy key

    Withdraws the specified policy from the system.
    """
    try:
        await make_onos_request("delete", f"/policys/{appId}/{key}")
        return f"policy {key} for app {appId} withdrawn successfully"
    except Exception as e:
        return f"Error withdrawing policy {key} for app {appId}: {str(e)}"


def register_tools(mcp_server: FastMCP):
    """Register all policy management tools with the MCP server."""
    mcp_server.tool()(get_all_policys)
    mcp_server.tool()(get_policy)
    mcp_server.tool()(get_policys_by_application)
    mcp_server.tool()(get_policy_installables)
    mcp_server.tool()(get_policy_related_flows)
    mcp_server.tool()(get_policys_summary)
    mcp_server.tool()(submit_policy)
    mcp_server.tool()(withdraw_policy)
