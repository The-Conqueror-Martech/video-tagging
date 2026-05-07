"""RunPod API client for pod management using GraphQL.

Manages GPU pods and dynamically retrieves vLLM endpoint URLs.
"""

import logging
from dataclasses import dataclass

import requests

from videotagger.config import get_settings

logger = logging.getLogger(__name__)

RUNPOD_API_URL = "https://api.runpod.io/graphql"


@dataclass
class PodPort:
    """Pod network port information."""

    private_port: int
    public_port: int | None
    ip: str | None
    is_public: bool
    type: str  # "http", "tcp", etc.


@dataclass
class PodStatus:
    """Status of a RunPod pod."""

    pod_id: str
    name: str
    status: str  # "RUNNING", "STOPPED", etc.
    gpu_type: str | None = None
    uptime_seconds: int | None = None
    cost_per_hour: float | None = None
    ports: list[PodPort] | None = None

    def get_vllm_endpoint(self, private_port: int = 8000) -> str | None:
        """Get vLLM OpenAI-compatible endpoint URL.

        Args:
            private_port: The private port where vLLM is running (default 8000).

        Returns:
            Full HTTP URL like https://{pod_id}-{private_port}.proxy.runpod.net/v1
            or None if not found.
        """
        if not self.ports:
            return None

        for port in self.ports:
            if port.private_port == private_port and port.type == "http":
                # RunPod proxy URL format uses pod_id and private_port
                # The proxy works even when is_public is False
                return f"https://{self.pod_id}-{private_port}.proxy.runpod.net/v1"
        return None


def _make_graphql_request(query: str, variables: dict | None = None) -> dict:
    """Make a GraphQL request to RunPod API.

    Args:
        query: GraphQL query string.
        variables: Optional variables for the query.

    Returns:
        Response data dictionary.

    Raises:
        RuntimeError: If the request fails.
    """
    config = get_settings().runpod_api

    headers = {"Content-Type": "application/json"}

    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    response = requests.post(
        f"{RUNPOD_API_URL}?api_key={config.api_key}",
        json=payload,
        headers=headers,
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(f"RunPod API request failed: {response.status_code} {response.text}")

    data = response.json()

    if "errors" in data:
        raise RuntimeError(f"RunPod API returned errors: {data['errors']}")

    return data.get("data", {})


def get_pod(pod_id: str) -> PodStatus | None:
    """Get detailed status of a specific pod.

    Args:
        pod_id: The pod ID to query.

    Returns:
        PodStatus with runtime details including ports, or None if not found.
    """
    query = """
    query Pod($podId: String!) {
        pod(input: {podId: $podId}) {
            id
            name
            desiredStatus
            costPerHr
            machine {
                gpuDisplayName
            }
            runtime {
                uptimeInSeconds
                ports {
                    ip
                    isIpPublic
                    privatePort
                    publicPort
                    type
                }
            }
        }
    }
    """

    try:
        data = _make_graphql_request(query, {"podId": pod_id})
        pod_data = data.get("pod")

        if not pod_data:
            logger.warning(f"Pod {pod_id} not found")
            return None

        # Parse runtime info
        runtime = pod_data.get("runtime")
        ports = None
        uptime = None

        if runtime:
            uptime = runtime.get("uptimeInSeconds")
            port_data = runtime.get("ports", [])
            ports = [
                PodPort(
                    private_port=p["privatePort"],
                    public_port=p.get("publicPort"),
                    ip=p.get("ip"),
                    is_public=p.get("isIpPublic", False),
                    type=p.get("type", "tcp"),
                )
                for p in port_data
            ]

        # Determine status
        desired = pod_data.get("desiredStatus", "").upper()
        if runtime and uptime and uptime > 0:
            status = "RUNNING"
        elif desired in ["RUNNING", "EXITED", "STOPPED"]:
            status = desired
        else:
            status = "UNKNOWN"

        return PodStatus(
            pod_id=pod_data["id"],
            name=pod_data.get("name", "Unknown"),
            status=status,
            gpu_type=pod_data.get("machine", {}).get("gpuDisplayName"),
            uptime_seconds=uptime,
            cost_per_hour=pod_data.get("costPerHr"),
            ports=ports,
        )

    except Exception as e:
        logger.error(f"Failed to get pod {pod_id}: {e}")
        return None


def get_pods() -> list[PodStatus]:
    """Get list of all pods (basic info only, no ports).

    Returns:
        List of PodStatus objects.
    """
    query = """
    query Pods {
        myself {
            pods {
                id
                name
                desiredStatus
                costPerHr
                machine {
                    gpuDisplayName
                }
            }
        }
    }
    """

    try:
        data = _make_graphql_request(query)
        pods_data = data.get("myself", {}).get("pods", [])

        result = []
        for pod in pods_data:
            result.append(
                PodStatus(
                    pod_id=pod["id"],
                    name=pod.get("name", "Unknown"),
                    status=pod.get("desiredStatus", "UNKNOWN").upper(),
                    gpu_type=pod.get("machine", {}).get("gpuDisplayName"),
                    cost_per_hour=pod.get("costPerHr"),
                )
            )

        logger.info(f"Found {len(result)} pods")
        return result

    except Exception as e:
        logger.error(f"Failed to get pods: {e}")
        return []


def start_pod(pod_id: str) -> bool:
    """Start/resume a stopped pod.

    Args:
        pod_id: The pod ID to start.

    Returns:
        True if successful, False otherwise.
    """
    mutation = """
    mutation ResumePod($podId: String!, $gpuCount: Int!) {
        podResume(input: {podId: $podId, gpuCount: $gpuCount}) {
            id
            desiredStatus
        }
    }
    """

    try:
        data = _make_graphql_request(mutation, {"podId": pod_id, "gpuCount": 1})
        result = data.get("podResume")

        if result:
            logger.info(f"Started pod: {pod_id}")
            return True
        else:
            logger.error(f"Failed to start pod {pod_id}: No result returned")
            return False

    except Exception as e:
        logger.error(f"Failed to start pod {pod_id}: {e}")
        return False


def stop_pod(pod_id: str) -> bool:
    """Stop a running pod.

    Args:
        pod_id: The pod ID to stop.

    Returns:
        True if successful, False otherwise.
    """
    mutation = """
    mutation StopPod($podId: String!) {
        podStop(input: {podId: $podId}) {
            id
            desiredStatus
        }
    }
    """

    try:
        data = _make_graphql_request(mutation, {"podId": pod_id})
        result = data.get("podStop")

        if result:
            logger.info(f"Stopped pod: {pod_id}")
            return True
        else:
            logger.error(f"Failed to stop pod {pod_id}: No result returned")
            return False

    except Exception as e:
        logger.error(f"Failed to stop pod {pod_id}: {e}")
        return False


def get_configured_pod_status() -> PodStatus | None:
    """Get status of the pod configured in settings.

    Returns:
        PodStatus of configured pod with ports, or None if not found.
    """
    pod_id = get_settings().runpod_ssh.pod_id
    return get_pod(pod_id)


def find_running_vllm_pod() -> PodStatus | None:
    """Find the first running pod with a vLLM endpoint (port 8000).

    Returns:
        PodStatus of a running pod with vLLM, or None if not found.
    """
    pods = get_pods()

    for pod_info in pods:
        if pod_info.status == "RUNNING":
            # Get full details including ports
            pod = get_pod(pod_info.pod_id)
            if pod and pod.get_vllm_endpoint():
                logger.info(f"Found running vLLM pod: {pod.name} ({pod.pod_id})")
                return pod

    return None


def ensure_pod_running() -> tuple[bool, str, str | None]:
    """Ensure a pod with vLLM is running and get its endpoint.

    Strategy:
    1. Look for any running pod with vLLM endpoint (port 8000)
    2. If none found, try to start the configured pod
    3. Return the vLLM endpoint URL

    Returns:
        Tuple of (success, message, vllm_endpoint_url).
    """
    # First, try to find any running pod with vLLM
    pod = find_running_vllm_pod()
    if pod:
        endpoint = pod.get_vllm_endpoint()
        return True, f"Using running pod: {pod.name}", endpoint

    # No running pod found, try to start the configured one
    logger.info("No running vLLM pod found, attempting to start configured pod...")

    try:
        pod_id = get_settings().runpod_ssh.pod_id
    except Exception:
        return False, "No running pods found and no pod configured in settings", None

    pod = get_pod(pod_id)
    if pod is None:
        return False, f"Configured pod {pod_id} not found", None

    if pod.status != "RUNNING":
        logger.info(f"Starting pod {pod.name}...")
        if not start_pod(pod_id):
            return False, f"Failed to start pod {pod.name}", None

        # Wait for pod to start and get ports
        import time

        logger.info("Waiting for pod to start...")
        for i in range(12):  # Wait up to 60 seconds
            time.sleep(5)
            pod = get_pod(pod_id)

            if pod and pod.status == "RUNNING" and pod.get_vllm_endpoint():
                break

        if not pod or pod.status != "RUNNING":
            return False, f"Pod {pod.name} did not start successfully", None

    # Get vLLM endpoint
    endpoint = pod.get_vllm_endpoint()
    if not endpoint:
        return False, f"Pod {pod.name} is running but vLLM endpoint not found (port 8000)", None

    return True, f"Started pod: {pod.name}", endpoint
