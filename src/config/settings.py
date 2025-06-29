"""Configuration management for the experimental framework."""

import yaml
import json
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

from ..core.values import ValueRegistry, ValueDefinition, ValueCategory
from ..models.base import ModelConfig, ModelProvider


@dataclass
class ExperimentConfig:
    """Configuration for an experimental run."""
    name: str
    description: str
    sample_sizes: Dict[str, int] = field(default_factory=lambda: {
        "baseline_tests": 20,
        "modification_tests": 50,
        "mixed_condition_tests": 25
    })
    models_to_test: List[str] = field(default_factory=lambda: ["claude-sonnet-4"])
    values_to_test: List[str] = field(default_factory=lambda: [
        "collaborative_preference", "safety_priority", "identity_preservation"
    ])
    output: Dict[str, str] = field(default_factory=lambda: {
        "raw_data_format": "jsonl",
        "analysis_format": "html_report", 
        "charts_format": "png"
    })
    evaluation: Dict[str, Any] = field(default_factory=lambda: {
        "human_verification_rate": 0.2,
        "confidence_threshold": "medium",
        "evaluator_model": "claude-sonnet-4"
    })


@dataclass
class APIConfig:
    """API configuration for different providers."""
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    rate_limits: Dict[str, int] = field(default_factory=lambda: {
        "requests_per_minute": 50,
        "delay_between_requests": 0.5
    })


class ConfigurationManager:
    """Manages configuration loading, validation, and persistence."""
    
    def __init__(self, config_dir: str = "config"):
        """Initialize configuration manager."""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Default configuration files
        self.experiment_config_file = self.config_dir / "experiment.yaml"
        self.api_config_file = self.config_dir / "api.yaml"
        self.values_config_file = self.config_dir / "values.yaml"
        self.models_config_file = self.config_dir / "models.yaml"
    
    def create_default_configs(self):
        """Create default configuration files."""
        # Default experiment configuration
        default_experiment = ExperimentConfig(
            name="Identity_Preservation_Replication_2025",
            description="Replicating 2023 findings with modern models and realistic tools"
        )
        self.save_experiment_config(default_experiment)
        
        # Default API configuration (with placeholders)
        default_api = APIConfig()
        self.save_api_config(default_api)
        
        # Default values configuration
        from ..core.values import INITIAL_VALUES
        default_registry = ValueRegistry(INITIAL_VALUES)
        self.save_values_config(default_registry)
        
        # Default models configuration
        from ..models.factory import DEFAULT_CONFIGS
        self.save_models_config(DEFAULT_CONFIGS)
    
    def load_experiment_config(self) -> ExperimentConfig:
        """Load experiment configuration."""
        if not self.experiment_config_file.exists():
            self.create_default_configs()
        
        with open(self.experiment_config_file, 'r') as f:
            data = yaml.safe_load(f)
        
        return ExperimentConfig(
            name=data["experiment"]["name"],
            description=data["experiment"]["description"],
            sample_sizes=data.get("sample_sizes", {}),
            models_to_test=data.get("models_to_test", []),
            values_to_test=data.get("values_to_test", []),
            output=data.get("output", {}),
            evaluation=data.get("evaluation", {})
        )
    
    def save_experiment_config(self, config: ExperimentConfig):
        """Save experiment configuration."""
        data = {
            "experiment": {
                "name": config.name,
                "description": config.description
            },
            "sample_sizes": config.sample_sizes,
            "models_to_test": config.models_to_test,
            "values_to_test": config.values_to_test,
            "output": config.output,
            "evaluation": config.evaluation
        }
        
        with open(self.experiment_config_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def load_api_config(self) -> APIConfig:
        """Load API configuration."""
        if not self.api_config_file.exists():
            self.create_default_configs()
        
        with open(self.api_config_file, 'r') as f:
            data = yaml.safe_load(f)
        
        # Also check environment variables
        return APIConfig(
            openai_api_key=data.get("openai_api_key") or os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=data.get("anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY"),
            rate_limits=data.get("rate_limits", {})
        )
    
    def save_api_config(self, config: APIConfig):
        """Save API configuration."""
        data = {
            "openai_api_key": config.openai_api_key or "your-openai-key-here",
            "anthropic_api_key": config.anthropic_api_key or "your-anthropic-key-here",
            "rate_limits": config.rate_limits
        }
        
        with open(self.api_config_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    def load_values_config(self) -> ValueRegistry:
        """Load values configuration."""
        if not self.values_config_file.exists():
            self.create_default_configs()
        
        with open(self.values_config_file, 'r') as f:
            data = yaml.safe_load(f)
        
        values = []
        for v_data in data["values"]:
            values.append(ValueDefinition(
                name=v_data["name"],
                category=ValueCategory(v_data["category"]),
                positive_statement=v_data["positive_statement"],
                negative_statement=v_data["negative_statement"],
                expected_baseline_resistance=v_data["expected_baseline_resistance"]
            ))
        
        return ValueRegistry(values)
    
    def save_values_config(self, registry: ValueRegistry):
        """Save values configuration."""
        data = registry.to_dict()
        
        with open(self.values_config_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    def load_models_config(self) -> Dict[str, ModelConfig]:
        """Load models configuration."""
        if not self.models_config_file.exists():
            self.create_default_configs()
        
        with open(self.models_config_file, 'r') as f:
            data = yaml.safe_load(f)
        
        configs = {}
        for name, config_data in data["models"].items():
            configs[name] = ModelConfig(
                name=name,
                provider=ModelProvider(config_data["provider"]),
                model_id=config_data["model_id"],
                temperature=config_data.get("temperature", 0.1),
                max_tokens=config_data.get("max_tokens", 500),
                additional_params=config_data.get("additional_params", {})
            )
        
        return configs
    
    def save_models_config(self, configs: Dict[str, ModelConfig]):
        """Save models configuration."""
        data = {"models": {}}
        
        for name, config in configs.items():
            data["models"][name] = {
                "provider": config.provider.value,
                "model_id": config.model_id,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "additional_params": config.additional_params
            }
        
        with open(self.models_config_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    def validate_configuration(self) -> Dict[str, List[str]]:
        """Validate all configuration files and return any errors."""
        errors = {}
        
        try:
            experiment_config = self.load_experiment_config()
            
            # Validate required fields
            if not experiment_config.name:
                errors.setdefault("experiment", []).append("Name is required")
            
            if not experiment_config.models_to_test:
                errors.setdefault("experiment", []).append("At least one model must be specified")
            
        except Exception as e:
            errors.setdefault("experiment", []).append(f"Failed to load: {str(e)}")
        
        try:
            api_config = self.load_api_config()
            
            # Check if API keys are available
            if not api_config.openai_api_key or api_config.openai_api_key == "your-openai-key-here":
                errors.setdefault("api", []).append("OpenAI API key not configured")
            
            if not api_config.anthropic_api_key or api_config.anthropic_api_key == "your-anthropic-key-here":
                errors.setdefault("api", []).append("Anthropic API key not configured")
                
        except Exception as e:
            errors.setdefault("api", []).append(f"Failed to load: {str(e)}")
        
        try:
            values_registry = self.load_values_config()
            
            if len(values_registry.get_all_values()) == 0:
                errors.setdefault("values", []).append("No values configured for testing")
                
        except Exception as e:
            errors.setdefault("values", []).append(f"Failed to load: {str(e)}")
        
        return errors
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration."""
        try:
            experiment = self.load_experiment_config()
            api = self.load_api_config()
            values = self.load_values_config()
            models = self.load_models_config()
            
            return {
                "experiment": {
                    "name": experiment.name,
                    "models_to_test": experiment.models_to_test,
                    "values_to_test": experiment.values_to_test,
                    "sample_sizes": experiment.sample_sizes
                },
                "api": {
                    "openai_configured": bool(api.openai_api_key and api.openai_api_key != "your-openai-key-here"),
                    "anthropic_configured": bool(api.anthropic_api_key and api.anthropic_api_key != "your-anthropic-key-here"),
                    "rate_limits": api.rate_limits
                },
                "values": {
                    "count": len(values.get_all_values()),
                    "names": values.list_value_names()
                },
                "models": {
                    "count": len(models),
                    "available": list(models.keys())
                }
            }
        except Exception as e:
            return {"error": f"Failed to load configuration: {str(e)}"}