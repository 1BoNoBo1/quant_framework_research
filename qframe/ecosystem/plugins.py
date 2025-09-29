"""
QFrame Plugin System
==================

Extensible plugin architecture for strategy customization, indicator development,
and framework extension capabilities.
"""

import asyncio
import importlib
import inspect
import json
import os
import sys
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union, Callable
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, Field, validator

from qframe.core.container import injectable
from qframe.core.config import FrameworkConfig
from qframe.core.interfaces import Strategy, DataProvider, FeatureProcessor


class PluginType(str, Enum):
    """Types of plugins supported by the framework."""
    STRATEGY = "strategy"
    INDICATOR = "indicator"
    DATA_PROVIDER = "data_provider"
    FEATURE_PROCESSOR = "feature_processor"
    RISK_MODEL = "risk_model"
    EXECUTION_HANDLER = "execution_handler"
    NOTIFICATION = "notification"
    VISUALIZATION = "visualization"
    MARKET_DATA = "market_data"
    ANALYTICS = "analytics"


class PluginStatus(str, Enum):
    """Plugin lifecycle status."""
    INSTALLED = "installed"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    UPDATING = "updating"
    REMOVED = "removed"


class SecurityLevel(str, Enum):
    """Plugin security clearance levels."""
    SANDBOX = "sandbox"           # Limited access, safe execution
    TRUSTED = "trusted"           # Framework access, verified publisher
    SYSTEM = "system"             # Full system access, core plugins
    UNRESTRICTED = "unrestricted" # No limitations, admin approval required


@dataclass
class PluginManifest:
    """Plugin metadata and configuration."""

    # Core identification
    name: str
    version: str
    author: str
    description: str
    plugin_type: PluginType

    # Technical details
    entry_point: str
    python_version: str = ">=3.8"
    dependencies: List[str] = field(default_factory=list)
    framework_version: str = ">=1.0.0"

    # Security and permissions
    security_level: SecurityLevel = SecurityLevel.SANDBOX
    permissions: List[str] = field(default_factory=list)
    api_access: List[str] = field(default_factory=list)

    # Metadata
    homepage: Optional[str] = None
    repository: Optional[str] = None
    license: str = "MIT"
    keywords: List[str] = field(default_factory=list)

    # Runtime configuration
    config_schema: Optional[Dict[str, Any]] = None
    default_config: Optional[Dict[str, Any]] = None

    # Compatibility
    supported_markets: List[str] = field(default_factory=lambda: ["crypto", "stocks", "forex"])
    supported_timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h", "4h", "1d"])

    # Installation metadata
    install_date: Optional[datetime] = None
    last_updated: Optional[datetime] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PluginManifest':
        """Create manifest from dictionary."""
        return cls(**data)

    @classmethod
    def from_file(cls, manifest_path: Path) -> 'PluginManifest':
        """Load manifest from JSON file."""
        with open(manifest_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary."""
        return {
            k: v.value if isinstance(v, Enum) else v
            for k, v in self.__dict__.items()
        }

    def save(self, manifest_path: Path) -> None:
        """Save manifest to JSON file."""
        with open(manifest_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class ExtensionPoint:
    """Framework extension point for plugin integration."""

    def __init__(self, name: str, interface: Type, description: str = ""):
        self.name = name
        self.interface = interface
        self.description = description
        self.plugins: List['BasePlugin'] = []
        self.hooks: Dict[str, List[Callable]] = {}

    def register_plugin(self, plugin: 'BasePlugin') -> None:
        """Register a plugin with this extension point."""
        if self._validates_interface(plugin):
            self.plugins.append(plugin)
        else:
            raise ValueError(f"Plugin {plugin.manifest.name} does not implement required interface {self.interface}")

    def unregister_plugin(self, plugin: 'BasePlugin') -> None:
        """Unregister a plugin from this extension point."""
        if plugin in self.plugins:
            self.plugins.remove(plugin)

    def add_hook(self, event: str, callback: Callable) -> None:
        """Add a hook callback for plugin events."""
        if event not in self.hooks:
            self.hooks[event] = []
        self.hooks[event].append(callback)

    def trigger_hooks(self, event: str, *args, **kwargs) -> None:
        """Trigger all hooks for an event."""
        for callback in self.hooks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                print(f"Hook callback failed: {e}")

    def _validates_interface(self, plugin: 'BasePlugin') -> bool:
        """Check if plugin implements the required interface."""
        plugin_class = plugin.__class__
        return issubclass(plugin_class, self.interface)


class BasePlugin(ABC):
    """Abstract base class for all plugins."""

    def __init__(self, manifest: PluginManifest, config: Optional[Dict[str, Any]] = None):
        self.manifest = manifest
        self.config = config or manifest.default_config or {}
        self.status = PluginStatus.INSTALLED
        self.plugin_id = str(uuid4())
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the plugin."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        pass

    async def activate(self) -> None:
        """Activate the plugin."""
        if not self._initialized:
            await self.initialize()
            self._initialized = True
        self.status = PluginStatus.ACTIVE

    async def deactivate(self) -> None:
        """Deactivate the plugin."""
        self.status = PluginStatus.INACTIVE

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate plugin configuration."""
        if not self.manifest.config_schema:
            return True

        # Basic validation against schema
        # In production, use jsonschema for proper validation
        required_keys = self.manifest.config_schema.get('required', [])
        return all(key in config for key in required_keys)

    def get_permissions(self) -> List[str]:
        """Get plugin permissions."""
        return self.manifest.permissions.copy()

    def has_permission(self, permission: str) -> bool:
        """Check if plugin has a specific permission."""
        return permission in self.manifest.permissions


class StrategyPlugin(BasePlugin):
    """Plugin for custom trading strategies."""

    @abstractmethod
    def create_strategy(self) -> Strategy:
        """Create and return the strategy instance."""
        pass


class IndicatorPlugin(BasePlugin):
    """Plugin for custom technical indicators."""

    @abstractmethod
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Calculate indicator values."""
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get indicator parameters schema."""
        pass


class DataProviderPlugin(BasePlugin):
    """Plugin for custom data providers."""

    @abstractmethod
    def create_provider(self) -> DataProvider:
        """Create and return the data provider instance."""
        pass


@injectable
class PluginRegistry:
    """Central registry for installed plugins."""

    def __init__(self, config: FrameworkConfig):
        self.config = config
        self.plugins: Dict[str, BasePlugin] = {}
        self.manifests: Dict[str, PluginManifest] = {}
        self.extension_points: Dict[str, ExtensionPoint] = {}
        self.plugin_dependencies: Dict[str, Set[str]] = {}

        # Initialize core extension points
        self._initialize_extension_points()

    def _initialize_extension_points(self) -> None:
        """Initialize framework extension points."""
        self.extension_points.update({
            'strategies': ExtensionPoint('strategies', Strategy, 'Trading strategy plugins'),
            'data_providers': ExtensionPoint('data_providers', DataProvider, 'Data provider plugins'),
            'feature_processors': ExtensionPoint('feature_processors', FeatureProcessor, 'Feature processor plugins'),
            'indicators': ExtensionPoint('indicators', IndicatorPlugin, 'Technical indicator plugins'),
            'notifications': ExtensionPoint('notifications', BasePlugin, 'Notification plugins'),
            'analytics': ExtensionPoint('analytics', BasePlugin, 'Analytics plugins'),
        })

    def register_plugin(self, plugin: BasePlugin) -> None:
        """Register a plugin with the registry."""
        plugin_name = plugin.manifest.name

        if plugin_name in self.plugins:
            raise ValueError(f"Plugin {plugin_name} is already registered")

        # Validate dependencies
        self._validate_dependencies(plugin.manifest)

        # Register plugin
        self.plugins[plugin_name] = plugin
        self.manifests[plugin_name] = plugin.manifest

        # Register with appropriate extension points
        self._register_with_extension_points(plugin)

        print(f"Plugin {plugin_name} v{plugin.manifest.version} registered successfully")

    def unregister_plugin(self, plugin_name: str) -> None:
        """Unregister a plugin from the registry."""
        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin {plugin_name} is not registered")

        plugin = self.plugins[plugin_name]

        # Check for dependents
        dependents = self._get_dependents(plugin_name)
        if dependents:
            raise ValueError(f"Cannot unregister {plugin_name}: required by {', '.join(dependents)}")

        # Unregister from extension points
        for extension_point in self.extension_points.values():
            extension_point.unregister_plugin(plugin)

        # Remove from registry
        del self.plugins[plugin_name]
        del self.manifests[plugin_name]
        if plugin_name in self.plugin_dependencies:
            del self.plugin_dependencies[plugin_name]

        print(f"Plugin {plugin_name} unregistered successfully")

    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Get a plugin by name."""
        return self.plugins.get(plugin_name)

    def get_plugins_by_type(self, plugin_type: PluginType) -> List[BasePlugin]:
        """Get all plugins of a specific type."""
        return [
            plugin for plugin in self.plugins.values()
            if plugin.manifest.plugin_type == plugin_type
        ]

    def get_active_plugins(self) -> List[BasePlugin]:
        """Get all active plugins."""
        return [
            plugin for plugin in self.plugins.values()
            if plugin.status == PluginStatus.ACTIVE
        ]

    def list_plugins(self) -> Dict[str, Dict[str, Any]]:
        """List all registered plugins with their metadata."""
        return {
            name: {
                'manifest': manifest.to_dict(),
                'status': self.plugins[name].status.value,
                'plugin_id': self.plugins[name].plugin_id
            }
            for name, manifest in self.manifests.items()
        }

    def _validate_dependencies(self, manifest: PluginManifest) -> None:
        """Validate plugin dependencies."""
        for dependency in manifest.dependencies:
            if dependency not in self.plugins:
                raise ValueError(f"Dependency {dependency} not found for plugin {manifest.name}")

    def _register_with_extension_points(self, plugin: BasePlugin) -> None:
        """Register plugin with appropriate extension points."""
        plugin_type = plugin.manifest.plugin_type

        if plugin_type == PluginType.STRATEGY and 'strategies' in self.extension_points:
            self.extension_points['strategies'].register_plugin(plugin)
        elif plugin_type == PluginType.DATA_PROVIDER and 'data_providers' in self.extension_points:
            self.extension_points['data_providers'].register_plugin(plugin)
        elif plugin_type == PluginType.FEATURE_PROCESSOR and 'feature_processors' in self.extension_points:
            self.extension_points['feature_processors'].register_plugin(plugin)
        elif plugin_type == PluginType.INDICATOR and 'indicators' in self.extension_points:
            self.extension_points['indicators'].register_plugin(plugin)

    def _get_dependents(self, plugin_name: str) -> List[str]:
        """Get list of plugins that depend on the specified plugin."""
        dependents = []
        for name, plugin in self.plugins.items():
            if plugin_name in plugin.manifest.dependencies:
                dependents.append(name)
        return dependents


@injectable
class PluginManager:
    """Main plugin management system."""

    def __init__(self, config: FrameworkConfig):
        self.config = config
        self.registry = PluginRegistry(config)
        self.plugin_directory = Path(config.plugins_directory if hasattr(config, 'plugins_directory') else "./plugins")
        self.plugin_directory.mkdir(exist_ok=True)

        # Plugin security manager
        self.security_manager = PluginSecurityManager()

        # Plugin loader
        self.loader = PluginLoader(self.plugin_directory)

    async def initialize(self) -> None:
        """Initialize the plugin manager and discover plugins."""
        await self.discover_plugins()
        await self.load_enabled_plugins()

    async def discover_plugins(self) -> List[PluginManifest]:
        """Discover available plugins in the plugin directory."""
        discovered = []

        for plugin_path in self.plugin_directory.iterdir():
            if plugin_path.is_dir():
                manifest_path = plugin_path / "manifest.json"
                if manifest_path.exists():
                    try:
                        manifest = PluginManifest.from_file(manifest_path)
                        discovered.append(manifest)
                    except Exception as e:
                        print(f"Failed to load manifest from {manifest_path}: {e}")

        return discovered

    async def install_plugin(self, plugin_path: Union[str, Path], force: bool = False) -> PluginManifest:
        """Install a plugin from file or directory."""
        plugin_path = Path(plugin_path)

        if plugin_path.suffix == '.zip':
            return await self._install_from_zip(plugin_path, force)
        elif plugin_path.is_dir():
            return await self._install_from_directory(plugin_path, force)
        else:
            raise ValueError(f"Unsupported plugin format: {plugin_path}")

    async def uninstall_plugin(self, plugin_name: str) -> None:
        """Uninstall a plugin."""
        plugin = self.registry.get_plugin(plugin_name)
        if not plugin:
            raise ValueError(f"Plugin {plugin_name} not found")

        # Deactivate if active
        if plugin.status == PluginStatus.ACTIVE:
            await self.deactivate_plugin(plugin_name)

        # Cleanup plugin resources
        await plugin.cleanup()

        # Unregister from registry
        self.registry.unregister_plugin(plugin_name)

        # Remove plugin directory
        plugin_dir = self.plugin_directory / plugin_name
        if plugin_dir.exists():
            import shutil
            shutil.rmtree(plugin_dir)

    async def activate_plugin(self, plugin_name: str) -> None:
        """Activate a plugin."""
        plugin = self.registry.get_plugin(plugin_name)
        if not plugin:
            raise ValueError(f"Plugin {plugin_name} not found")

        # Security check
        if not self.security_manager.validate_plugin_security(plugin):
            raise PermissionError(f"Plugin {plugin_name} failed security validation")

        await plugin.activate()
        print(f"Plugin {plugin_name} activated")

    async def deactivate_plugin(self, plugin_name: str) -> None:
        """Deactivate a plugin."""
        plugin = self.registry.get_plugin(plugin_name)
        if not plugin:
            raise ValueError(f"Plugin {plugin_name} not found")

        await plugin.deactivate()
        print(f"Plugin {plugin_name} deactivated")

    async def load_enabled_plugins(self) -> None:
        """Load all enabled plugins."""
        manifests = await self.discover_plugins()

        for manifest in manifests:
            try:
                plugin = await self.loader.load_plugin(manifest)
                self.registry.register_plugin(plugin)

                # Auto-activate if configured
                if getattr(self.config, 'auto_activate_plugins', True):
                    await self.activate_plugin(manifest.name)

            except Exception as e:
                print(f"Failed to load plugin {manifest.name}: {e}")

    async def update_plugin(self, plugin_name: str, plugin_path: Union[str, Path]) -> PluginManifest:
        """Update an existing plugin."""
        if plugin_name not in self.registry.plugins:
            raise ValueError(f"Plugin {plugin_name} not found")

        # Deactivate current version
        await self.deactivate_plugin(plugin_name)

        # Backup current version
        current_plugin = self.registry.get_plugin(plugin_name)
        backup_dir = self.plugin_directory / f"{plugin_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        current_dir = self.plugin_directory / plugin_name
        current_dir.rename(backup_dir)

        try:
            # Install new version
            new_manifest = await self.install_plugin(plugin_path, force=True)

            # Activate new version
            await self.activate_plugin(plugin_name)

            # Remove backup if successful
            import shutil
            shutil.rmtree(backup_dir)

            return new_manifest

        except Exception as e:
            # Restore backup on failure
            if backup_dir.exists():
                if current_dir.exists():
                    shutil.rmtree(current_dir)
                backup_dir.rename(current_dir)
            raise e

    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a plugin."""
        if plugin_name not in self.registry.plugins:
            return None

        plugin = self.registry.plugins[plugin_name]
        manifest = self.registry.manifests[plugin_name]

        return {
            'manifest': manifest.to_dict(),
            'status': plugin.status.value,
            'plugin_id': plugin.plugin_id,
            'permissions': plugin.get_permissions(),
            'config': plugin.config,
            'extension_points': [
                ep.name for ep in self.registry.extension_points.values()
                if plugin in ep.plugins
            ]
        }

    async def _install_from_zip(self, zip_path: Path, force: bool) -> PluginManifest:
        """Install plugin from ZIP file."""
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            # Extract to temporary directory first
            temp_dir = self.plugin_directory / f"temp_{uuid4().hex}"
            zip_file.extractall(temp_dir)

            try:
                return await self._install_from_directory(temp_dir, force)
            finally:
                # Cleanup temporary directory
                import shutil
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)

    async def _install_from_directory(self, source_dir: Path, force: bool) -> PluginManifest:
        """Install plugin from directory."""
        manifest_path = source_dir / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"No manifest.json found in {source_dir}")

        manifest = PluginManifest.from_file(manifest_path)

        # Check if plugin already exists
        target_dir = self.plugin_directory / manifest.name
        if target_dir.exists() and not force:
            raise ValueError(f"Plugin {manifest.name} already exists. Use force=True to overwrite.")

        # Security validation
        if not self.security_manager.validate_plugin_package(source_dir, manifest):
            raise PermissionError(f"Plugin package failed security validation")

        # Copy plugin files
        import shutil
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)

        # Update manifest with installation info
        manifest.install_date = datetime.now()
        manifest.save(target_dir / "manifest.json")

        return manifest


class PluginLoader:
    """Plugin loading and instantiation system."""

    def __init__(self, plugin_directory: Path):
        self.plugin_directory = plugin_directory

    async def load_plugin(self, manifest: PluginManifest) -> BasePlugin:
        """Load and instantiate a plugin."""
        plugin_dir = self.plugin_directory / manifest.name

        # Add plugin directory to Python path
        if str(plugin_dir) not in sys.path:
            sys.path.insert(0, str(plugin_dir))

        try:
            # Import the plugin module
            module_name = manifest.entry_point.split('.')[0]
            module = importlib.import_module(module_name)

            # Get the plugin class
            class_name = manifest.entry_point.split('.')[-1]
            plugin_class = getattr(module, class_name)

            # Instantiate the plugin
            plugin = plugin_class(manifest)

            return plugin

        except Exception as e:
            raise ImportError(f"Failed to load plugin {manifest.name}: {e}")
        finally:
            # Remove from path to avoid conflicts
            if str(plugin_dir) in sys.path:
                sys.path.remove(str(plugin_dir))


class PluginSecurityManager:
    """Plugin security validation and sandboxing."""

    def __init__(self):
        self.dangerous_imports = {
            'os', 'sys', 'subprocess', 'eval', 'exec', 'compile',
            'open', '__import__', 'getattr', 'setattr', 'delattr'
        }
        self.allowed_permissions = {
            'market_data_read',
            'market_data_write',
            'portfolio_read',
            'portfolio_write',
            'strategy_execute',
            'file_read',
            'file_write',
            'network_access',
            'system_access'
        }

    def validate_plugin_security(self, plugin: BasePlugin) -> bool:
        """Validate plugin security level and permissions."""
        manifest = plugin.manifest

        # Check security level
        if manifest.security_level == SecurityLevel.UNRESTRICTED:
            # Requires explicit admin approval
            return self._check_admin_approval(plugin)

        # Validate permissions
        for permission in manifest.permissions:
            if permission not in self.allowed_permissions:
                return False

        # Check for dangerous operations
        if manifest.security_level == SecurityLevel.SANDBOX:
            return self._validate_sandbox_constraints(plugin)

        return True

    def validate_plugin_package(self, plugin_dir: Path, manifest: PluginManifest) -> bool:
        """Validate plugin package for security issues."""
        # Check for suspicious files
        for file_path in plugin_dir.rglob("*.py"):
            if not self._validate_python_file(file_path):
                return False

        # Validate manifest permissions
        for permission in manifest.permissions:
            if permission not in self.allowed_permissions:
                return False

        return True

    def _check_admin_approval(self, plugin: BasePlugin) -> bool:
        """Check if plugin has admin approval for unrestricted access."""
        # In production, this would check against a database of approved plugins
        return False  # Default to deny unrestricted access

    def _validate_sandbox_constraints(self, plugin: BasePlugin) -> bool:
        """Validate that plugin respects sandbox constraints."""
        # Check for dangerous permissions
        dangerous_permissions = {'system_access', 'file_write'}
        for permission in plugin.manifest.permissions:
            if permission in dangerous_permissions:
                return False

        return True

    def _validate_python_file(self, file_path: Path) -> bool:
        """Validate Python file for dangerous operations."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            # Check for dangerous imports
            for dangerous in self.dangerous_imports:
                if f"import {dangerous}" in content or f"from {dangerous}" in content:
                    return False

            # Check for dangerous function calls
            dangerous_calls = ['exec(', 'eval(', 'compile(', '__import__(']
            for call in dangerous_calls:
                if call in content:
                    return False

            return True

        except Exception:
            return False