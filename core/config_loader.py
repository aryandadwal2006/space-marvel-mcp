import json
import os
import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from jsonschema import validate, ValidationError, Draft7Validator
import streamlit as st
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigChangeHandler(FileSystemEventHandler):
    """Handles filesystem events for config file changes"""
    
    def __init__(self, config_loader):
        self.config_loader = config_loader
        self.last_reload = 0
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        if event.src_path.endswith('.json'):
            # Debounce rapid file changes
            current_time = time.time()
            if current_time - self.last_reload > 1:
                self.last_reload = current_time
                logger.info(f"Config file changed: {event.src_path}")
                self.config_loader.reload_configs()

class ConfigLoader:
    """Loads, validates, and manages MCP server configurations"""
    
    def __init__(self, base_config_dir: str = "configs", schema_path: str = "schemas/server.schema.json"):
        self.base_config_dir = Path(base_config_dir)
        self.schema_path = Path(schema_path)
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.validation_errors: Dict[str, str] = {}
        self.last_reload_time: Optional[datetime] = None
        self._observer: Optional[Observer] = None
        self._lock = threading.Lock()
        
        # Ensure directories exist
        self.base_config_dir.mkdir(exist_ok=True)
        self.schema_path.parent.mkdir(exist_ok=True)
        
        # Load schema
        self._load_schema()
        
        # Initial config load
        self.reload_configs()
        
        # Start file watcher for hot reload
        self._start_file_watcher()
    
    def _load_schema(self):
        """Load and validate the JSON schema"""
        try:
            if self.schema_path.exists():
                with open(self.schema_path, 'r') as f:
                    self.schema = json.load(f)
                    # Validate the schema itself
                    Draft7Validator.check_schema(self.schema)
                    logger.info(f"Schema loaded from {self.schema_path}")
            else:
                # Create default schema if it doesn't exist
                self._create_default_schema()
        except Exception as e:
            logger.error(f"Failed to load schema: {e}")
            self.schema = {}
    
    def _create_default_schema(self):
        """Create default schema file"""
        default_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "MCP Server Configuration",
            "type": "object",
            "required": ["id", "name", "transport", "tools"],
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
                "transport": {"type": "object"},
                "tools": {"type": "array"}
            }
        }
        
        with open(self.schema_path, 'w') as f:
            json.dump(default_schema, f, indent=2)
        
        self.schema = default_schema
        logger.info(f"Created default schema at {self.schema_path}")
    
    def _start_file_watcher(self):
        """Start watching config directory for changes"""
        try:
            self._observer = Observer()
            handler = ConfigChangeHandler(self)
            self._observer.schedule(handler, str(self.base_config_dir), recursive=True)
            self._observer.start()
            logger.info(f"Started file watcher for {self.base_config_dir}")
        except Exception as e:
            logger.error(f"Failed to start file watcher: {e}")
    
    def stop_file_watcher(self):
        """Stop the file watcher"""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            logger.info("Stopped file watcher")
    
    def get_user_config_dir(self, user_id: str) -> Path:
        """Get config directory for specific user"""
        user_dir = self.base_config_dir / user_id
        user_dir.mkdir(exist_ok=True)
        return user_dir
    
    def reload_configs(self, user_id: str = "global"):
        """Reload all configuration files"""
        with self._lock:
            try:
                config_dir = self.get_user_config_dir(user_id) if user_id != "global" else self.base_config_dir
                
                new_configs = {}
                new_errors = {}
                
                # Find all JSON files in config directory
                pattern = str(config_dir / "*.json")
                config_files = glob.glob(pattern)
                
                logger.info(f"Loading {len(config_files)} config files from {config_dir}")
                
                for config_file in config_files:
                    try:
                        config_id = Path(config_file).stem
                        config_data = self._load_and_validate_config(config_file)
                        
                        if config_data:
                            new_configs[config_id] = config_data
                            logger.info(f"Loaded config: {config_id}")
                        
                    except Exception as e:
                        error_msg = f"Failed to load {config_file}: {str(e)}"
                        new_errors[config_file] = error_msg
                        logger.error(error_msg)
                
                # Update configs atomically
                if user_id not in st.session_state:
                    st.session_state[f'configs_{user_id}'] = {}
                    st.session_state[f'errors_{user_id}'] = {}
                
                st.session_state[f'configs_{user_id}'] = new_configs
                st.session_state[f'errors_{user_id}'] = new_errors
                
                self.last_reload_time = datetime.now()
                
                logger.info(f"Reloaded {len(new_configs)} configs with {len(new_errors)} errors")
                
            except Exception as e:
                logger.error(f"Failed to reload configs: {e}")
    
    def _load_and_validate_config(self, config_file: str) -> Optional[Dict[str, Any]]:
        """Load and validate a single config file"""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Validate against schema
            if self.schema:
                validate(instance=config_data, schema=self.schema)
            
            # Additional custom validations
            self._validate_config_logic(config_data)
            
            # Add metadata
            config_data['_metadata'] = {
                'file_path': config_file,
                'loaded_at': datetime.now().isoformat(),
                'file_size': os.path.getsize(config_file)
            }
            
            return config_data
            
        except ValidationError as e:
            raise ValueError(f"Schema validation failed: {e.message}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
        except Exception as e:
            raise ValueError(f"Unexpected error: {e}")
    
    def _validate_config_logic(self, config: Dict[str, Any]):
        """Perform additional logical validation"""
        # Check transport type matches required fields
        transport = config.get('transport', {})
        transport_type = transport.get('type')
        
        if transport_type == 'stdio' and not transport.get('command'):
            raise ValueError("stdio transport requires 'command' field")
        
        if transport_type in ['sse', 'http'] and not transport.get('url'):
            raise ValueError(f"{transport_type} transport requires 'url' field")
        
        # Validate tool names are unique
        tools = config.get('tools', [])
        tool_names = [tool['name'] for tool in tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("Tool names must be unique within a server")
        
        # Validate environment variable format
        env_vars = transport.get('env', {})
        for var_name in env_vars:
            if not var_name.replace('_', '').isalnum():
                raise ValueError(f"Invalid environment variable name: {var_name}")
    
    def get_configs(self, user_id: str = "global") -> Dict[str, Dict[str, Any]]:
        """Get all loaded configurations for a user"""
        return st.session_state.get(f'configs_{user_id}', {})
    
    def get_config(self, config_id: str, user_id: str = "global") -> Optional[Dict[str, Any]]:
        """Get a specific configuration"""
        configs = self.get_configs(user_id)
        return configs.get(config_id)
    
    def get_validation_errors(self, user_id: str = "global") -> Dict[str, str]:
        """Get validation errors for configs"""
        return st.session_state.get(f'errors_{user_id}', {})
    
    def save_config(self, config_id: str, config_data: Dict[str, Any], user_id: str = "global") -> bool:
        """Save a configuration to file"""
        try:
            # Validate before saving
            if self.schema:
                validate(instance=config_data, schema=self.schema)
            
            self._validate_config_logic(config_data)
            
            # Save to appropriate directory
            config_dir = self.get_user_config_dir(user_id) if user_id != "global" else self.base_config_dir
            config_file = config_dir / f"{config_id}.json"
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Saved config {config_id} to {config_file}")
            
            # Trigger reload
            self.reload_configs(user_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save config {config_id}: {e}")
            return False
    
    def delete_config(self, config_id: str, user_id: str = "global") -> bool:
        """Delete a configuration file"""
        try:
            config_dir = self.get_user_config_dir(user_id) if user_id != "global" else self.base_config_dir
            config_file = config_dir / f"{config_id}.json"
            
            if config_file.exists():
                config_file.unlink()
                logger.info(f"Deleted config {config_id}")
                
                # Trigger reload
                self.reload_configs(user_id)
                return True
            else:
                logger.warning(f"Config file not found: {config_file}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete config {config_id}: {e}")
            return False
    
    def get_all_tools(self, user_id: str = "global") -> Dict[str, List[Dict[str, Any]]]:
        """Get all tools from all servers"""
        tools_by_server = {}
        configs = self.get_configs(user_id)
        
        for server_id, config in configs.items():
            tools = config.get('tools', [])
            tools_by_server[server_id] = tools
        
        return tools_by_server
    
    def search_tools(self, query: str, user_id: str = "global") -> List[Dict[str, Any]]:
        """Search for tools by name or description"""
        results = []
        configs = self.get_configs(user_id)
        
        query_lower = query.lower()
        
        for server_id, config in configs.items():
            server_name = config.get('name', server_id)
            tools = config.get('tools', [])
            
            for tool in tools:
                tool_name = tool.get('name', '').lower()
                tool_desc = tool.get('description', '').lower()
                
                if query_lower in tool_name or query_lower in tool_desc:
                    tool_info = tool.copy()
                    tool_info['server_id'] = server_id
                    tool_info['server_name'] = server_name
                    results.append(tool_info)
        
        return results
    
    def get_stats(self, user_id: str = "global") -> Dict[str, Any]:
        """Get statistics about loaded configurations"""
        configs = self.get_configs(user_id)
        errors = self.get_validation_errors(user_id)
        
        total_tools = sum(len(config.get('tools', [])) for config in configs.values())
        
        transport_types = {}
        for config in configs.values():
            transport_type = config.get('transport', {}).get('type', 'unknown')
            transport_types[transport_type] = transport_types.get(transport_type, 0) + 1
        
        return {
            'total_servers': len(configs),
            'total_tools': total_tools,
            'validation_errors': len(errors),
            'transport_types': transport_types,
            'last_reload': self.last_reload_time.isoformat() if self.last_reload_time else None
        }
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop_file_watcher()