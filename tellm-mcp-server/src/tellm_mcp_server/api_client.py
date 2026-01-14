from typing import Any, Dict, List, Optional
import os
import httpx

# Configuration
ONOS_API_BASE = os.environ.get("ONOS_API_BASE", "http://localhost:8181/onos/v1")
TE_HEAD_API_BASE = os.environ.get("TE_HEAD_API_BASE", "http://localhost:5001")
ONOS_USERNAME = os.environ.get("ONOS_USERNAME", "onos")
ONOS_PASSWORD = os.environ.get("ONOS_PASSWORD", "rocks")
HTTP_TIMEOUT = 30.0  # seconds


async def make_onos_request(
    method: str,
    path: str,
    json: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Make a request to the ONOS REST API with proper authentication and error handling."""
    url = f"{ONOS_API_BASE}{path}"
    auth = (ONOS_USERNAME, ONOS_PASSWORD)

    async with httpx.AsyncClient() as client:
        try:
            if method.lower() == "get":
                response = await client.get(
                    url, auth=auth, params=params, timeout=HTTP_TIMEOUT
                )
            elif method.lower() == "post":
                response = await client.post(
                    url, auth=auth, json=json, timeout=HTTP_TIMEOUT
                )
            elif method.lower() == "put":
                response = await client.put(
                    url, auth=auth, json=json, timeout=HTTP_TIMEOUT
                )
            elif method.lower() == "delete":
                response = await client.delete(url, auth=auth, timeout=HTTP_TIMEOUT)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response.json() if response.content else {}
        except httpx.HTTPStatusError as e:
            error_msg = f"ONOS API error: {e.response.status_code} - {e.response.text}"
            raise ValueError(error_msg)
        except Exception as e:
            raise ValueError(f"Error connecting to ONOS: {str(e)}")

async def make_tellm_request(
    method: str,
    path: str,
    api_base: str = TE_HEAD_API_BASE,
    json: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Any:
    """
    Make a request to TeLLM TE-head API with proper error handling.
    
    Args:
        method: HTTP method (get, post, put, delete)
        path: API endpoint path
        api_base: Base URL for the API (defaults to TE_HEAD_API_BASE)
        json: JSON payload for POST/PUT requests
        params: Query parameters for GET requests
        headers: Custom headers for the request
    
    Returns:
        Response data (JSON if content exists, otherwise empty dict)
    
    Raises:
        ValueError: If the request fails or returns an error status
    """
    url = f"{api_base}{path}"
    request_headers = headers or {"Content-Type": "application/json"}
    async with httpx.AsyncClient() as client:
        try:
            if method.lower() == "get":
                response = await client.get(
                    url, headers=request_headers, params=params, timeout=HTTP_TIMEOUT
                )
            elif method.lower() == "post":
                response = await client.post(
                    url, headers=request_headers, json=json, timeout=HTTP_TIMEOUT
                )
            elif method.lower() == "put":
                response = await client.put(
                    url, headers=request_headers, json=json, timeout=HTTP_TIMEOUT
                )
            elif method.lower() == "delete":
                response = await client.delete(
                    url, headers=request_headers, timeout=HTTP_TIMEOUT
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            response.raise_for_status()
            
            # Return appropriate response based on content type
            if response.headers.get("content-type", "").startswith("application/json"):
                return response.json() if response.content else {}
            else:
                return response.text if response.content else ""
                
        except httpx.HTTPStatusError as e:
            error_msg = f"TeLLM API error: {e.response.status_code} - {e.response.text}"
            raise ValueError(error_msg)
        except Exception as e:
            raise ValueError(f"Error connecting to TeLLM API: {str(e)}")