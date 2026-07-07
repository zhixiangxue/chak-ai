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


class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""
    
    api_key: str = Field(description="API key or ${ENV_VAR} reference")
    base_url: Optional[str] = Field(default=None, description="Custom API base URL")


class ServerConfig(BaseModel):
    """Complete server configuration."""
    
    providers: Dict[str, ProviderConfig] = Field(
        description="Provider configurations keyed by provider name"
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
        
        if 'providers' not in data:
            raise ValueError("Configuration must contain 'providers' section")
        
        return cls(**data)
    
    def get_provider_config(self, provider: str) -> Optional[Dict[str, str]]:
        """
        Get provider configuration (API key and optional base_url).
        
        Args:
            provider: Provider name (e.g., 'openai', 'ollama')
            
        Returns:
            Dict with 'api_key' and optional 'base_url', or None if not found
        """
        if provider not in self.providers:
            return None
        
        cfg = self.providers[provider]
        api_key = self._resolve_value(cfg.api_key)
        if not api_key:
            return None
        
        result = {'api_key': api_key}
        if cfg.base_url:
            result['base_url'] = cfg.base_url
        return result
    
    def get_provider_entries(self) -> Dict[str, Dict[str, str]]:
        """
        Get all provider entries with their configurations.
        
        Returns:
            Dict mapping provider display keys to their config.
            Simple providers use the provider name as key.
            Providers with custom base_url use 'provider@base_url' as key
            (for frontend compatibility).
        """
        result = {}
        
        for name, cfg in self.providers.items():
            api_key = self._resolve_value(cfg.api_key)
            if not api_key:
                continue
            
            if cfg.base_url:
                # Frontend-compatible key: provider@base_url
                display_key = f"{name}@{cfg.base_url}"
                result[display_key] = {'api_key': api_key, 'base_url': cfg.base_url}
            else:
                result[name] = {'api_key': api_key}
        
        return result
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for specified provider."""
        config = self.get_provider_config(provider)
        return config['api_key'] if config else None
    
    @staticmethod
    def _resolve_value(value: str) -> Optional[str]:
        """
        Resolve a config value, supporting ${ENV_VAR} syntax.
        
        Args:
            value: Raw value from config file
            
        Returns:
            Resolved value or None
        """
        if not value:
            return None
        
        if value.startswith("${") and value.endswith("}"):
            var_name = value[2:-1].strip()
            return os.getenv(var_name)
        
        return value
