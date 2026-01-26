"""Tests for Docker configuration files."""
from __future__ import annotations

from pathlib import Path

import pytest

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


class TestDockerfileExists:
    """Test that Docker files exist."""

    def test_dockerfile_exists(self):
        """Dockerfile exists in project root."""
        dockerfile = PROJECT_ROOT / "Dockerfile"
        assert dockerfile.exists(), "Dockerfile not found"

    def test_docker_compose_exists(self):
        """docker-compose.yml exists in project root."""
        compose = PROJECT_ROOT / "docker-compose.yml"
        assert compose.exists(), "docker-compose.yml not found"

    def test_nginx_conf_exists(self):
        """nginx/nginx.conf exists."""
        nginx_conf = PROJECT_ROOT / "nginx" / "nginx.conf"
        assert nginx_conf.exists(), "nginx/nginx.conf not found"


class TestDockerfileSyntax:
    """Test Dockerfile syntax and structure."""

    def test_dockerfile_has_from_instruction(self):
        """Dockerfile starts with FROM instruction."""
        dockerfile = PROJECT_ROOT / "Dockerfile"
        content = dockerfile.read_text()
        lines = [l.strip() for l in content.split("\n") if l.strip() and not l.strip().startswith("#")]
        # First non-comment, non-ARG line should be FROM or ARG
        assert any(l.startswith("FROM") or l.startswith("ARG") for l in lines[:5])

    def test_dockerfile_has_entrypoint_or_cmd(self):
        """Dockerfile has ENTRYPOINT or CMD."""
        dockerfile = PROJECT_ROOT / "Dockerfile"
        content = dockerfile.read_text()
        assert "ENTRYPOINT" in content or "CMD" in content

    def test_dockerfile_exposes_port(self):
        """Dockerfile exposes port 8000."""
        dockerfile = PROJECT_ROOT / "Dockerfile"
        content = dockerfile.read_text()
        assert "EXPOSE 8000" in content

    def test_dockerfile_has_healthcheck(self):
        """Dockerfile has HEALTHCHECK."""
        dockerfile = PROJECT_ROOT / "Dockerfile"
        content = dockerfile.read_text()
        assert "HEALTHCHECK" in content


class TestDockerComposeStructure:
    """Test docker-compose.yml structure."""

    def test_compose_is_valid_yaml(self):
        """docker-compose.yml is valid YAML."""
        import yaml

        compose = PROJECT_ROOT / "docker-compose.yml"
        content = compose.read_text()
        data = yaml.safe_load(content)
        assert data is not None
        assert isinstance(data, dict)

    def test_compose_has_services(self):
        """docker-compose.yml has services section."""
        import yaml

        compose = PROJECT_ROOT / "docker-compose.yml"
        data = yaml.safe_load(compose.read_text())
        assert "services" in data
        assert len(data["services"]) > 0

    def test_compose_has_tts_api_service(self):
        """docker-compose.yml has tts-api service."""
        import yaml

        compose = PROJECT_ROOT / "docker-compose.yml"
        data = yaml.safe_load(compose.read_text())
        assert "tts-api" in data["services"]

    def test_compose_has_nginx_service(self):
        """docker-compose.yml has nginx service."""
        import yaml

        compose = PROJECT_ROOT / "docker-compose.yml"
        data = yaml.safe_load(compose.read_text())
        assert "nginx" in data["services"]


class TestNginxConfig:
    """Test nginx configuration."""

    def test_nginx_has_upstream(self):
        """nginx.conf has upstream block for tts backend."""
        nginx_conf = PROJECT_ROOT / "nginx" / "nginx.conf"
        content = nginx_conf.read_text()
        assert "upstream" in content
        assert "tts" in content.lower()

    def test_nginx_has_location_health(self):
        """nginx.conf has /health location."""
        nginx_conf = PROJECT_ROOT / "nginx" / "nginx.conf"
        content = nginx_conf.read_text()
        assert "location /health" in content or "location = /health" in content

    def test_nginx_has_location_v1(self):
        """nginx.conf has /v1/ location."""
        nginx_conf = PROJECT_ROOT / "nginx" / "nginx.conf"
        content = nginx_conf.read_text()
        assert "location /v1" in content


class TestGitHubWorkflows:
    """Test GitHub Actions workflow files."""

    def test_ci_workflow_exists(self):
        """CI workflow file exists."""
        ci = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
        assert ci.exists(), "CI workflow not found"

    def test_docker_workflow_exists(self):
        """Docker workflow file exists."""
        docker = PROJECT_ROOT / ".github" / "workflows" / "docker.yml"
        assert docker.exists(), "Docker workflow not found"

    def test_ci_workflow_valid_yaml(self):
        """CI workflow is valid YAML."""
        import yaml

        ci = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
        data = yaml.safe_load(ci.read_text())
        assert "jobs" in data
        # YAML parses 'on:' as True (boolean), so check for True key
        assert "on" in data or True in data
