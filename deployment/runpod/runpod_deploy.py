#!/usr/bin/env python3
"""
RunPod deployment script for Voxtral 3B Real-Time Streaming
Handles automated deployment, configuration, and monitoring
"""

import os
import json
import time
import logging
import argparse
import subprocess
from typing import Dict, Any, Optional
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RunPodDeployer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.runpod.io/graphql"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def deploy_pod(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy a new pod with the specified configuration"""
        logger.info("🚀 Starting pod deployment...")

        # GraphQL mutation for creating a pod
        mutation = """
        mutation PodCreate($input: PodCreateInput!) {
            podCreate(input: $input) {
                id
                name
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
                machine {
                    podHostId
                }
            }
        }
        """

        with open("deployment/runpod/startup_no_docker.sh", "r") as f:
            startup_script = f.read()

        variables = {
            "input": {
                "name": config.get("name", "voxtral-streaming"),
                "imageName": config.get("base_image", "runpod/base:0.7.0-noble-cuda1290"),
                "gpuTypeId": config.get("gpu_type", "NVIDIA GeForce RTX 3090"),
                "cloudType": config.get("cloud_type", "COMMUNITY"),
                "supportPublicIp": True,
                "startJupyter": False,
                "startSsh": True,
                "dockerArgs": f'--entrypoint /bin/bash -c "{startup_script}"',
                "ports": config.get("ports", "8000/http,8001/http"),
                "volumeInGb": config.get("volume_size", 50),
                "containerDiskInGb": config.get("container_disk", 20),
                "env": [
                    {"key": "CUDA_VISIBLE_DEVICES", "value": "0"},
                    {"key": "TORCH_CUDA_ARCH_LIST", "value": "6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"},
                    {"key": "NVIDIA_VISIBLE_DEVICES", "value": "all"},
                    {"key": "MODEL_NAME", "value": "mistralai/Voxtral-Mini-3B-2507"},
                    {"key": "MAX_BATCH_SIZE", "value": str(config.get("max_batch_size", 4))},
                    {"key": "ENABLE_TORCH_COMPILE", "value": str(config.get("enable_torch_compile", True)).lower()},
                    *config.get("extra_env", [])
                ]
            }
        }

        try:
            response = requests.post(
                self.base_url,
                json={"query": mutation, "variables": variables},
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            if "errors" in result:
                raise Exception(f"GraphQL errors: {result['errors']}")

            pod_data = result["data"]["podCreate"]
            logger.info(f"✅ Pod created successfully: {pod_data['id']}")

            return pod_data

        except Exception as e:
            logger.error(f"❌ Pod deployment failed: {e}")
            raise

    def wait_for_pod_ready(self, pod_id: str, timeout: int = 600) -> Dict[str, Any]:
        """Wait for pod to be ready and return connection info"""
        logger.info(f"⏳ Waiting for pod {pod_id} to be ready...")

        query = """
        query Pod($podId: String!) {
            pod(input: {podId: $podId}) {
                id
                name
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
                lastStatusChange
            }
        }
        """

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.post(
                    self.base_url,
                    json={"query": query, "variables": {"podId": pod_id}},
                    headers=self.headers,
                    timeout=10
                )
                response.raise_for_status()

                result = response.json()
                if "errors" in result:
                    logger.warning(f"Query error: {result['errors']}")
                    time.sleep(10)
                    continue

                pod_data = result["data"]["pod"]
                if not pod_data:
                    logger.warning("Pod not found, retrying...")
                    time.sleep(10)
                    continue

                runtime = pod_data.get("runtime")
                if runtime and runtime.get("ports"):
                    logger.info("✅ Pod is ready!")
                    return pod_data

                logger.info("Pod starting up, checking again in 10 seconds...")
                time.sleep(10)

            except Exception as e:
                logger.warning(f"Status check failed: {e}, retrying...")
                time.sleep(10)

        raise TimeoutError(f"Pod {pod_id} did not become ready within {timeout} seconds")

    def get_pod_endpoints(self, pod_data: Dict[str, Any]) -> Dict[str, str]:
        """Extract endpoint URLs from pod data"""
        endpoints = {}

        runtime = pod_data.get("runtime", {})
        ports = runtime.get("ports", [])

        for port in ports:
            if port.get("isIpPublic") and port.get("publicPort"):
                port_num = port["privatePort"]
                public_port = port["publicPort"]
                ip = port["ip"]

                if port_num == 8000:
                    endpoints["api"] = f"https://{ip}:{public_port}"
                    endpoints["websocket"] = f"wss://{ip}:{public_port}"
                elif port_num == 8001:
                    endpoints["metrics"] = f"https://{ip}:{public_port}"

        return endpoints

    def test_deployment(self, endpoints: Dict[str, str]) -> bool:
        """Test the deployed service"""
        logger.info("🧪 Testing deployment...")

        if "api" not in endpoints:
            logger.error("❌ No API endpoint found")
            return False

        try:
            # Test health endpoint
            health_url = f"{endpoints['api']}/health"
            response = requests.get(health_url, timeout=30)
            response.raise_for_status()

            health_data = response.json()
            logger.info(f"✅ Health check passed: {health_data.get('status')}")

            # Test model loading
            if health_data.get("gpu_available"):
                logger.info(f"✅ GPU available: {health_data.get('gpu_count')} devices")
            else:
                logger.warning("⚠️ GPU not available")

            return True

        except Exception as e:
            logger.error(f"❌ Deployment test failed: {e}")
            return False

    def deploy_and_wait(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Complete deployment workflow"""
        logger.info("🌟 Starting complete deployment workflow...")

        # Deploy pod
        pod_data = self.deploy_pod(config)
        pod_id = pod_data["id"]

        # Wait for pod to be ready
        ready_pod_data = self.wait_for_pod_ready(pod_id)

        # Get endpoints
        endpoints = self.get_pod_endpoints(ready_pod_data)

        if not endpoints:
            raise Exception("No public endpoints found")

        logger.info("🔗 Endpoints:")
        for name, url in endpoints.items():
            logger.info(f"  {name}: {url}")

        # Test deployment
        if self.test_deployment(endpoints):
            logger.info("✅ Deployment successful!")
        else:
            logger.error("❌ Deployment test failed")

        return {
            "pod_id": pod_id,
            "pod_data": ready_pod_data,
            "endpoints": endpoints
        }

def main():
    parser = argparse.ArgumentParser(description="Deploy Voxtral 3B streaming service to RunPod")
    parser.add_argument("--api-key", required=True, help="RunPod API key")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--gpu-type", default="NVIDIA GeForce RTX 3090", help="GPU type")
    parser.add_argument("--cloud-type", default="COMMUNITY", choices=["COMMUNITY", "SECURE"], help="Cloud type")
    parser.add_argument("--name", default="voxtral-streaming", help="Pod name")
    parser.add_argument("--base-image", default="runpod/base:0.7.0-noble-cuda1290", help="Base Docker image to use for deployment")
    parser.add_argument("--image", default="rohithb/voxtral-streaming:latest", help="Docker image")
    parser.add_argument("--volume-size", type=int, default=50, help="Volume size in GB")
    parser.add_argument("--max-batch-size", type=int, default=4, help="Maximum batch size")
    parser.add_argument("--enable-torch-compile", action="store_true", default=True, help="Enable torch.compile")

    args = parser.parse_args()

    # Load configuration
    config = {
        "name": args.name,
        "base_image": args.base_image,
        "gpu_type": args.gpu_type,
        "cloud_type": args.cloud_type,
        "volume_size": args.volume_size,
        "max_batch_size": args.max_batch_size,
        "enable_torch_compile": args.enable_torch_compile
    }

    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            file_config = json.load(f)
            config.update(file_config)

    # Deploy
    deployer = RunPodDeployer(args.api_key)

    try:
        result = deployer.deploy_and_wait(config)

        # Save deployment info
        deployment_info = {
            "timestamp": time.time(),
            "config": config,
            **result
        }

        with open("deployment_info.json", "w") as f:
            json.dump(deployment_info, f, indent=2)

        logger.info("📝 Deployment info saved to deployment_info.json")

        print("\n" + "="*50)
        print("🎉 DEPLOYMENT SUCCESSFUL!")
        print("="*50)
        print(f"Pod ID: {result['pod_id']}")
        print(f"API Endpoint: {result['endpoints'].get('api', 'N/A')}")
        print(f"WebSocket: {result['endpoints'].get('websocket', 'N/A')}")
        print(f"Metrics: {result['endpoints'].get('metrics', 'N/A')}")
        print("="*50)

    except Exception as e:
        logger.error(f"💥 Deployment failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
