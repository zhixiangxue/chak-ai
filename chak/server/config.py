"""
Server configuration management.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class ServerSettings(BaseModel):
    """Server settings configuration."""
    
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")


class ServerConfig(BaseModel):
    """Complete server configuration."""
    
    api_keys: Dict[str, str] = Field(
        description="Provider API keys. Supports two formats: 'provider' or 'provider@base_url'"
    )
    server: ServerSettings = Field(default_factory=ServerSettings)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "ServerConfig":
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            ServerConfig instance
            
        Raises:
            FileNotFoundError: If config file not found
            ValueError: If config format is invalid
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if not data:
            raise ValueError(f"Empty configuration file: {config_path}")
        
        if 'api_keys' not in data:
            raise ValueError("Configuration must contain 'api_keys' section")
        
        return cls(**data)
    
    def get_provider_config(self, provider: str) -> Optional[Dict[str, str]]:
        """
        Get provider configuration (API key and optional base_url).
        
        Supports two config key formats:
        1. Simple: 'provider' -> uses default base_url
        2. Custom: 'provider@base_url' -> uses custom base_url
        
        Args:
            provider: Provider name (e.g., 'openai', 'ollama')
            
        Returns:
            Dict with 'api_key' and optional 'base_url', or None if not found
            
        Examples:
            # Simple format (default base_url)
            api_keys:
              openai: sk-xxx
            
            # Custom base_url format
            api_keys:
              "ollama@http://localhost:11434": "ollama"
        """
        # First try exact match (simple format)
        if provider in self.api_keys:
            api_key = self._resolve_api_key(self.api_keys[provider])
            if api_key:
                return {'api_key': api_key}
        
        # Then try matching provider@base_url format
        for config_key, config_value in self.api_keys.items():
            if '@' in config_key:
                # Parse "provider@base_url" format
                key_provider = config_key.split('@')[0]
                if key_provider == provider:
                    api_key = self._resolve_api_key(config_value)
                    if api_key:
                        base_url = config_key.split('@', 1)[1]
                        return {'api_key': api_key, 'base_url': base_url}
        
        return None
    
    def get_provider_entries(self) -> Dict[str, Dict[str, str]]:
        """
        Get all provider entries with their configurations.
        
        Returns:
            Dict mapping provider names to their config (api_key, optional base_url)
            For providers with custom base_url, the key is 'provider@base_url'
        """
        result = {}
        
        for config_key, config_value in self.api_keys.items():
            api_key = self._resolve_api_key(config_value)
            if not api_key:
                continue
                
            if '@' in config_key:
                # Custom base_url format: keep full key
                provider = config_key.split('@')[0]
                base_url = config_key.split('@', 1)[1]
                result[config_key] = {'api_key': api_key, 'base_url': base_url}
            else:
                # Simple format
                result[config_key] = {'api_key': api_key}
        
        return result
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Get API key for specified provider (backward compatible).
        
        Args:
            provider: Provider name or full config key
            
        Returns:
            API key or None if not found
        """
        config = self.get_provider_config(provider)
        return config['api_key'] if config else None
    
    def _resolve_api_key(self, config_value: str) -> Optional[str]:
        """
        Resolve API key from config value.
        
        Supports:
        1. ${ENV_VAR} syntax - references environment variable
        2. Plain text value (not recommended for production)
        
        Args:
            config_value: Value from config file
            
        Returns:
            Resolved API key or None
        """
        if not config_value:
            return None
        
        # Parse ${ENV_VAR} syntax if present
        if config_value.startswith("${") and config_value.endswith("}"):
            # Extract variable name from ${VAR_NAME}
            var_name = config_value[2:-1].strip()
            return os.getenv(var_name)
        
        # Return plain text value from config
        return config_value
