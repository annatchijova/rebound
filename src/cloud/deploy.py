
"""
Script de deploy para Alibaba Cloud ECS.

Prerequisitos:
1. Docker instalado localmente
2. Alibaba Cloud CLI configurado (aliyun configure)
3. Container Registry (ACR) creado en Alibaba Cloud

Uso:
    python3 -m src.cloud.deploy --registry <registry-url> --tag v0.1
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Execute a command and print it."""
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check)
    return result


def deploy(registry: str, tag: str = "latest") -> None:
    """Build and push Docker image."""
    image = f"{registry}/rebound:{tag}"

    print("=" * 50)
    print(f"Deploying REBOUND to {image}")
    print("=" * 50)

    # Build
    run(["docker", "build", "-t", image, "."])

    # Push
    run(["docker", "push", image])

    print(f"\nImagen publicada: {image}")
    print(f"\nPara correr en ECS:")
    print(f"  docker pull {image}")
    print(f"  docker run -d -p 8000:8000 -e DASHSCOPE_API_KEY=<key> {image}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy REBOUND a Alibaba Cloud")
    parser.add_argument("--registry", required=True, help="URL del registry ACR")
    parser.add_argument("--tag", default="latest", help="Tag de la imagen")
    args = parser.parse_args()

    deploy(args.registry, args.tag)
