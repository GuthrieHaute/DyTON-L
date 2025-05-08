#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AURA Universal Adapter Bay - Git Repository Analyzer Node

This module provides capabilities for analyzing Git repositories to extract
structured information about tools, libraries, and applications.
It specializes in identifying Web UIs and CLI interfaces for AURA Phase 1 integration.
"""

import os
import sys
import git
import json
import logging
import argparse
import tempfile
import shutil
import subprocess
import re
import uuid
import datetime
import fnmatch
import glob
import hashlib
import importlib.util
import io
import pkgutil
import requests
import time
import yaml
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, Set, Callable
from urllib.parse import urlparse, urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional imports that may not be available
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from huggingface_hub import HfApi, HfFolder, Repository, list_models, list_datasets
    HUGGINGFACE_HUB_AVAILABLE = True
except ImportError:
    HUGGINGFACE_HUB_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("git_repo_analyzer.log")
    ]
)
logger = logging.getLogger("git_repo_analyzer")


class InterfaceType(Enum):
    """Types of interfaces a tool might expose."""
    WEB_UI = "web_ui"
    CLI = "cli"
    API = "api"
    LIBRARY_IMPORT = "library_import"
    UNKNOWN = "unknown"


class SkillType(Enum):
    """Types of skills/tools that might be identified."""
    GENERAL_TOOL = "general_tool"
    HUGGINGFACE_MODEL = "huggingface_model"
    HUGGINGFACE_DATASET = "huggingface_dataset"
    HUGGINGFACE_SPACE = "huggingface_space"
    LIBRARY = "library"
    WEB_APPLICATION = "web_application"
    CLI_TOOL = "cli_tool"
    UNKNOWN = "unknown"


@dataclass
class SecurityPolicy:
    """Security policies for repository analysis."""
    sandbox_execution: bool = True
    allow_script_execution: bool = False
    resource_limits: Dict[str, Any] = field(default_factory=lambda: {
        "max_memory_mb": 1024,
        "max_disk_space_mb": 5120,
        "max_execution_time_sec": 300
    })
    allowed_domains: List[str] = field(default_factory=list)
    restricted_commands: List[str] = field(default_factory=lambda: [
        "rm", "sudo", "chmod", "chown", "dd",
        "mkfs", "mount", "umount", "apt", "yum", "dnf"
    ])


@dataclass
class GitCredentials:
    """Credentials for accessing Git repositories."""
    username: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None
    ssh_key_path: Optional[str] = None


@dataclass
class GitRepoInfo:
    """Information about a Git repository."""
    url: str
    branch: str = "main"
    tag: Optional[str] = None
    commit: Optional[str] = None
    credentials: Optional[GitCredentials] = None
    clone_dir: Optional[str] = None


@dataclass
class Feature:
    """A specific feature or capability identified in the repository."""
    feature_id: str
    name: str
    description: str
    invocation_pattern: str = ""
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    platform_requirements: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    confidence_score: float = 0.0


@dataclass
class WebUIInfo:
    """Information about a Web UI interface."""
    launch_command: str = ""
    base_url: str = "http://localhost:8000"  # Default assumption
    port: int = 8000  # Default assumption
    routes: List[Dict[str, str]] = field(default_factory=list)
    auth_required: bool = False
    notes: str = ""


@dataclass
class CLIInfo:
    """Information about a CLI interface."""
    entry_point: str = ""
    binary_name: str = ""
    subcommands: List[Dict[str, Any]] = field(default_factory=list)
    global_options: List[Dict[str, str]] = field(default_factory=list)
    help_text: str = ""
    version_info: str = ""


@dataclass
class HuggingFaceInfo:
    """Information specific to Hugging Face models, datasets, or spaces."""
    model_id: str = ""
    model_task: str = ""
    space_url: str = ""
    dataset_id: str = ""
    api_url: str = ""
    model_requirements: List[str] = field(default_factory=list)
    model_architecture: str = ""


@dataclass
class SkillManifest:
    """The final output manifest describing the repository as an AURA skill."""
    skill_id: str
    name: str
    description: str
    version: str = "0.1.0"
    source_repository: str = ""
    clone_command: str = ""
    language: List[str] = field(default_factory=list)
    installation_commands: List[str] = field(default_factory=list)
    execution_environment: Dict[str, Any] = field(default_factory=dict)
    entry_point: str = ""
    skill_type: str = SkillType.UNKNOWN.value
    primary_interface_type: str = InterfaceType.UNKNOWN.value
    web_ui_info: Dict[str, Any] = field(default_factory=dict)
    cli_info: Dict[str, Any] = field(default_factory=dict)
    huggingface_info: Dict[str, Any] = field(default_factory=dict)
    features: List[Dict[str, Any]] = field(default_factory=list)
    confidence_score: float = 0.0
    analysis_notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    created_by: str = "AURA GitRepoAnalyzerNode"


class GitRepoHandler:
    """Handles Git repository operations like cloning and fetching."""
    
    def __init__(self, security_policy: SecurityPolicy = None):
        """Initialize with optional security policy."""
        self.security_policy = security_policy or SecurityPolicy()
        
    def validate_repo_url(self, repo_url: str) -> bool:
        """
        Validate if the given URL is a valid Git repository URL.
        
        Args:
            repo_url: The URL of the Git repository.
            
        Returns:
            bool: True if valid, False otherwise.
        """
        try:
            parsed = urlparse(repo_url)
            
            # Check for common Git hosting schemes
            valid_schemes = ["http", "https", "git", "ssh"]
            valid_hosts = ["github.com", "gitlab.com", "bitbucket.org", "huggingface.co"]
            
            if parsed.scheme not in valid_schemes:
                return False
            
            # For SSH URLs, further validate format
            if parsed.scheme == "ssh":
                if not parsed.netloc or "@" not in parsed.netloc:
                    return False
            
            # Additional security check: restrict to allowed domains if specified
            if self.security_policy.allowed_domains:
                if not any(parsed.netloc.endswith(domain) for domain in self.security_policy.allowed_domains):
                    logger.warning(f"Repository domain not in allowed list: {parsed.netloc}")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating repo URL: {e}")
            return False
    
    def clone_repository(self, repo_info: GitRepoInfo) -> Optional[str]:
        """
        Clone a Git repository to a temporary directory.
        
        Args:
            repo_info: GitRepoInfo object containing repository details.
            
        Returns:
            str: Path to cloned repository or None if failed.
        """
        if not self.validate_repo_url(repo_info.url):
            logger.error(f"Invalid repository URL: {repo_info.url}")
            return None
        
        try:
            # Create a temporary directory for the clone if not specified
            if not repo_info.clone_dir:
                repo_info.clone_dir = tempfile.mkdtemp(prefix="aura_git_")
            
            logger.info(f"Cloning repository {repo_info.url} to {repo_info.clone_dir}")
            
            # Handle credentials if provided
            git_env = os.environ.copy()
            if repo_info.credentials:
                if repo_info.credentials.username and repo_info.credentials.password:
                    # For HTTP(S) authentication
                    url_parts = urlparse(repo_info.url)
                    auth_url = f"{url_parts.scheme}://{repo_info.credentials.username}:{repo_info.credentials.password}@{url_parts.netloc}{url_parts.path}"
                    repo_info.url = auth_url
                
                if repo_info.credentials.token:
                    # For token-based authentication (GitHub, GitLab)
                    url_parts = urlparse(repo_info.url)
                    auth_url = f"{url_parts.scheme}://{repo_info.credentials.token}@{url_parts.netloc}{url_parts.path}"
                    repo_info.url = auth_url
                
                if repo_info.credentials.ssh_key_path:
                    # For SSH key authentication
                    git_env["GIT_SSH_COMMAND"] = f"ssh -i {repo_info.credentials.ssh_key_path}"
            
            # Clone the repository
            git_args = ["git", "clone", repo_info.url, repo_info.clone_dir]
            
            # Add depth=1 for faster cloning unless a specific commit is requested
            if not repo_info.commit:
                git_args.insert(2, "--depth=1")
            
            # Add branch/tag if specified
            if repo_info.branch and repo_info.branch != "main" and not repo_info.tag:
                git_args.extend(["-b", repo_info.branch])
            elif repo_info.tag:
                git_args.extend(["-b", repo_info.tag])
            
            logger.debug(f"Running: {' '.join([arg for arg in git_args if '@' not in arg])}")  # Log without credentials
            
            result = subprocess.run(
                git_args,
                env=git_env,
                check=True,
                text=True,
                capture_output=True
            )
            
            # Check out specific commit if requested
            if repo_info.commit:
                checkout_args = ["git", "checkout", repo_info.commit]
                subprocess.run(
                    checkout_args,
                    cwd=repo_info.clone_dir,
                    check=True,
                    text=True,
                    capture_output=True
                )
            
            logger.info(f"Successfully cloned repository to {repo_info.clone_dir}")
            return repo_info.clone_dir
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {e.stderr}")
            self.cleanup_repository(repo_info.clone_dir)
            return None
        except Exception as e:
            logger.error(f"Error cloning repository: {e}")
            self.cleanup_repository(repo_info.clone_dir)
            return None
    
    def cleanup_repository(self, repo_dir: str) -> None:
        """
        Clean up a cloned repository directory.
        
        Args:
            repo_dir: Directory to clean up.
        """
        if not repo_dir or not os.path.exists(repo_dir):
            return
        
        try:
            logger.info(f"Cleaning up repository directory: {repo_dir}")
            shutil.rmtree(repo_dir)
        except Exception as e:
            logger.error(f"Error cleaning up repository: {e}")


class GitRepoAnalyzerNode:
    """Main node for analyzing Git repositories and generating skill manifests."""
    
    def __init__(
        self, 
        security_policy: SecurityPolicy = None,
        llm_api_key: str = None,
        llm_endpoint: str = "https://api.openai.com/v1",
        llm_model: str = "gpt-4",
    ):
        """
        Initialize the Git Repository Analyzer Node.
        
        Args:
            security_policy: Security policies for repository analysis.
            llm_api_key: API key for LLM service.
            llm_endpoint: Endpoint URL for LLM service.
            llm_model: Model identifier for LLM service.
        """
        self.security_policy = security_policy or SecurityPolicy()
        self.repo_handler = GitRepoHandler(self.security_policy)
        self.llm_api_key = llm_api_key or os.environ.get("AURA_LLM_API_KEY")
        self.llm_endpoint = llm_endpoint
        self.llm_model = llm_model
        
    def analyze_repository(self, repo_info: GitRepoInfo) -> Optional[SkillManifest]:
        """
        Analyze a Git repository and generate a skill manifest.
        
        Args:
            repo_info: GitRepoInfo object containing repository details.
            
        Returns:
            SkillManifest: The generated skill manifest or None if analysis failed.
        """
        try:
            # Clone the repository
            repo_dir = self.repo_handler.clone_repository(repo_info)
            if not repo_dir:
                logger.error("Failed to clone repository")
                return None
            
            # Generate a unique skill ID
            skill_id = f"git-repo-{uuid.uuid4()}"
            
            # Extract repository name from URL for preliminary manifest
            repo_name = os.path.basename(repo_info.url)
            if repo_name.endswith('.git'):
                repo_name = repo_name[:-4]
                
            # Create a preliminary manifest
            manifest = SkillManifest(
                skill_id=skill_id,
                name=repo_name,
                description=f"Skill generated from Git repository: {repo_info.url}",
                source_repository=repo_info.url,
                clone_command=f"git clone {repo_info.url}"
            )
            
            # TODO: Implement the layered analysis engine
            # For now, return a minimal manifest
            
            logger.info(f"Successfully generated preliminary manifest for {repo_info.url}")
            return manifest
        
        except Exception as e:
            logger.error(f"Error analyzing repository: {e}")
            return None
        finally:
            # Clean up
            if repo_info.clone_dir and os.path.exists(repo_info.clone_dir):
                self.repo_handler.cleanup_repository(repo_info.clone_dir)
    
    def save_manifest(self, manifest: SkillManifest, output_file: str) -> bool:
        """
        Save a skill manifest to a JSON file.
        
        Args:
            manifest: The SkillManifest to save.
            output_file: Path to the output file.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Create directories if needed
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
            # Convert dataclass to dict and save as JSON
            with open(output_file, 'w') as f:
                json.dump(asdict(manifest), f, indent=2, default=str)
            
            logger.info(f"Manifest saved to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving manifest: {e}")
            return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AURA Git Repository Analyzer Node",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--repo-url",
        type=str,
        required=True,
        help="URL of the Git repository to analyze"
    )
    
    parser.add_argument(
        "--branch",
        type=str,
        default="main",
        help="Branch to analyze (default: main)"
    )
    
    parser.add_argument(
        "--tag",
        type=str,
        help="Tag to analyze (overrides branch if specified)"
    )
    
    parser.add_argument(
        "--commit",
        type=str,
        help="Specific commit to analyze"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        default="aura_skill.json",
        help="Path to output manifest file"
    )
    
    parser.add_argument(
        "--username",
        type=str,
        help="Username for repository authentication"
    )
    
    parser.add_argument(
        "--password",
        type=str,
        help="Password for repository authentication"
    )
    
    parser.add_argument(
        "--token",
        type=str,
        help="Token for repository authentication"
    )
    
    parser.add_argument(
        "--ssh-key",
        type=str,
        help="Path to SSH key for repository authentication"
    )
    
    parser.add_argument(
        "--llm-api-key",
        type=str,
        help="API key for LLM service"
    )
    
    parser.add_argument(
        "--llm-endpoint",
        type=str,
        default="https://api.openai.com/v1",
        help="Endpoint URL for LLM service"
    )
    
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4",
        help="Model identifier for LLM service"
    )
    
    parser.add_argument(
        "--sandbox-execution",
        type=bool,
        default=True,
        help="Enable/disable sandbox execution"
    )
    
    parser.add_argument(
        "--allow-script-execution",
        type=bool,
        default=False,
        help="Allow execution of scripts from the repository"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create security policy
    security_policy = SecurityPolicy(
        sandbox_execution=args.sandbox_execution,
        allow_script_execution=args.allow_script_execution
    )
    
    # Create credentials if provided
    credentials = None
    if any([args.username, args.password, args.token, args.ssh_key]):
        credentials = GitCredentials(
            username=args.username,
            password=args.password,
            token=args.token,
            ssh_key_path=args.ssh_key
        )
    
    # Create repository info
    repo_info = GitRepoInfo(
        url=args.repo_url,
        branch=args.branch,
        tag=args.tag,
        commit=args.commit,
        credentials=credentials
    )
    
    # Create analyzer node
    analyzer = GitRepoAnalyzerNode(
        security_policy=security_policy,
        llm_api_key=args.llm_api_key,
        llm_endpoint=args.llm_endpoint,
        llm_model=args.llm_model
    )
    
    # Analyze repository
    manifest = analyzer.analyze_repository(repo_info)
    
    if manifest:
        # Save manifest
        analyzer.save_manifest(manifest, args.output_file)
        logger.info(f"Analysis complete. Manifest saved to {args.output_file}")
        return 0
    else:
        logger.error("Analysis failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())