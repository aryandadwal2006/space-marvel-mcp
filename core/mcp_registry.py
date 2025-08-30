import asyncio
import logging
import threading
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

import streamlit as st
from mcp.types import Tool, CallToolResult

from core.config_loader import ConfigLoader
from core.transport_adapters import ConnectionManager, MCPTransportAdapter

logger = logging.getLogger(__name__)

@dataclass
class ServerInfo:
    """Information about a registered MCP server"""
    id: str
    name: str
    description: str
    config: Dict[str, Any]
    adapter: Optional[MCPTransportAdapter] = None
    tools: List[Tool] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    connection_status: str = "disconnected"
    error_message: Optional[str] = None
    call_count: int = 0
    last_call_time: Optional[datetime] = None

@dataclass
class ToolCall:
    """Record of a tool call"""
    tool_name: str
    server_id: str
    arguments: Dict[str, Any]
    result: Any
    success: bool
    duration: float
    timestamp: datetime
    error_message: Optional[str] = None

class MCPRegistry:
    """Central registry for MCP servers and tools"""
    
    def __init__(self, config_loader: ConfigLoader):
        self.config_loader = config_loader
        self.connection_manager = ConnectionManager()
        self.servers: Dict[str, ServerInfo] = {}
        self.tool_calls: List[ToolCall] = []
        self.max_call_history = 1000
        self._lock = threading.Lock()
        
        # Background task management
        self._background_tasks: Dict[str, asyncio.Task] = {}
        self._shutdown_event = asyncio.Event()
        self._tasks_started = False
        
        self._registry_stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'average_response_time': 0.0
        }
    
    async def initialize(self, user_id: str = "global") -> None:
        """Initialize the registry with configs from loader"""
        try:
            # Ensure any existing tasks are cleaned up first
            await self._cleanup_background_tasks()
            
            # Load all configs
            configs = self.config_loader.get_configs(user_id)
            
            logger.info(f"Initializing registry with {len(configs)} servers for user {user_id}")
            
            # Connect to all servers concurrently
            connection_tasks = []
            for config_id, config in configs.items():
                task = asyncio.create_task(self._add_server_async(config))
                connection_tasks.append(task)
            
            # Wait for all connections with timeout
            if connection_tasks:
                results = await asyncio.gather(*connection_tasks, return_exceptions=True)
                
                successful = sum(1 for r in results if r is True)
                failed = len(results) - successful
                
                logger.info(f"Registry initialization complete: {successful} successful, {failed} failed")
            
            # Start background monitoring
            await self._start_background_tasks()
            
        except Exception as e:
            logger.error(f"Failed to initialize registry: {e}")
            # Ensure cleanup even on failure
            await self._cleanup_background_tasks()
            raise
    
    async def reload_from_config(self, user_id: str = "global") -> None:
        """Reload registry from updated configs"""
        try:
            # Get current configs
            current_configs = self.config_loader.get_configs(user_id)
            current_server_ids = set(current_configs.keys())
            registered_server_ids = set(self.servers.keys())
            
            # Find servers to add, update, and remove
            to_add = current_server_ids - registered_server_ids
            to_remove = registered_server_ids - current_server_ids
            to_update = current_server_ids & registered_server_ids
            
            logger.info(f"Reloading registry: +{len(to_add)} -{len(to_remove)} ~{len(to_update)}")
            
            # Remove old servers
            for server_id in to_remove:
                await self.remove_server(server_id)
            
            # Add new servers
            for server_id in to_add:
                config = current_configs[server_id]
                await self._add_server_async(config)
            
            # Update existing servers (check if config changed)
            for server_id in to_update:
                current_config = current_configs[server_id]
                existing_server = self.servers.get(server_id)
                
                if existing_server and existing_server.config != current_config:
                    logger.info(f"Config changed for server {server_id}, reconnecting")
                    await self.remove_server(server_id)
                    await self._add_server_async(current_config)
            
        except Exception as e:
            logger.error(f"Failed to reload registry: {e}")
    
    async def _add_server_async(self, config: Dict[str, Any]) -> bool:
        """Add a server asynchronously"""
        try:
            server_id = config['id']
            
            # Create server info
            server_info = ServerInfo(
                id=server_id,
                name=config['name'],
                description=config.get('description', ''),
                config=config
            )
            
            # Add to connection manager
            if await self.connection_manager.add_server(config):
                adapter = self.connection_manager.get_adapter(server_id)
                if adapter:
                    server_info.adapter = adapter
                    server_info.connection_status = "connected"
                    
                    # Get tools
                    try:
                        tools = await adapter.get_tools()
                        server_info.tools = tools
                        logger.info(f"Loaded {len(tools)} tools from {server_id}")
                    except Exception as e:
                        logger.warning(f"Failed to load tools from {server_id}: {e}")
                
                # Register server
                with self._lock:
                    self.servers[server_id] = server_info
                
                logger.info(f"Successfully added server {server_id}")
                return True
            else:
                logger.error(f"Failed to connect to server {server_id}")
                
                # Still register server with error status
                server_info.connection_status = "error"
                server_info.error_message = "Connection failed"
                
                with self._lock:
                    self.servers[server_id] = server_info
                
                return False
                
        except Exception as e:
            logger.error(f"Error adding server {config.get('id', 'unknown')}: {e}")
            return False
    
    async def add_server(self, config: Dict[str, Any]) -> bool:
        """Add a new server to the registry"""
        return await self._add_server_async(config)
    
    async def remove_server(self, server_id: str) -> bool:
        """Remove a server from the registry"""
        try:
            # Remove from connection manager
            await self.connection_manager.remove_server(server_id)
            
            # Remove from registry
            with self._lock:
                if server_id in self.servers:
                    del self.servers[server_id]
                    logger.info(f"Removed server {server_id} from registry")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error removing server {server_id}: {e}")
            return False
    
    def get_server(self, server_id: str) -> Optional[ServerInfo]:
        """Get server information"""
        return self.servers.get(server_id)
    
    def get_all_servers(self) -> Dict[str, ServerInfo]:
        """Get all registered servers"""
        return self.servers.copy()
    
    def get_server_tools(self, server_id: str) -> List[Tool]:
        """Get tools for a specific server"""
        server = self.servers.get(server_id)
        return server.tools if server else []
    
    def get_all_tools(self) -> Dict[str, List[Tool]]:
        """Get all tools from all servers"""
        all_tools = {}
        for server_id, server_info in self.servers.items():
            all_tools[server_id] = server_info.tools
        return all_tools
    
    def search_tools(self, query: str, category: Optional[str] = None) -> List[Tuple[str, Tool]]:
        """Search for tools across all servers"""
        results = []
        query_lower = query.lower()
        
        for server_id, server_info in self.servers.items():
            for tool in server_info.tools:
                # Check name and description
                name_match = query_lower in tool.name.lower()
                desc_match = query_lower in (tool.description or '').lower()
                
                # Check category if specified
                category_match = True
                if category:
                    tool_category = getattr(tool, 'category', None)
                    category_match = tool_category and tool_category.lower() == category.lower()
                
                if (name_match or desc_match) and category_match:
                    results.append((server_id, tool))
        
        return results
    
    def get_tool_by_name(self, tool_name: str) -> List[Tuple[str, Tool]]:
        """Find tool by exact name across all servers"""
        results = []
        for server_id, server_info in self.servers.items():
            for tool in server_info.tools:
                if tool.name == tool_name:
                    results.append((server_id, tool))
        return results
    
    async def call_tool(self, server_id: str, tool_name: str, arguments: Dict[str, Any]) -> ToolCall:
        """Call a tool on a specific server"""
        start_time = time.time()
        
        try:
            # Get server
            server = self.servers.get(server_id)
            if not server:
                raise ValueError(f"Server {server_id} not found")
            
            if not server.adapter:
                raise ValueError(f"Server {server_id} not connected")
            
            # Validate tool exists
            tool_exists = any(tool.name == tool_name for tool in server.tools)
            if not tool_exists:
                raise ValueError(f"Tool {tool_name} not found on server {server_id}")
            
            # Call the tool
            result = await server.adapter.call_tool(tool_name, arguments)
            duration = time.time() - start_time
            
            # Create call record
            tool_call = ToolCall(
                tool_name=tool_name,
                server_id=server_id,
                arguments=arguments,
                result=result,
                success=True,
                duration=duration,
                timestamp=datetime.now()
            )
            
            # Update server stats
            server.call_count += 1
            server.last_call_time = datetime.now()
            
            # Update registry stats
            self._update_stats(tool_call)
            
            # Store call history
            with self._lock:
                self.tool_calls.append(tool_call)
                
                # Trim history if needed
                if len(self.tool_calls) > self.max_call_history:
                    self.tool_calls = self.tool_calls[-self.max_call_history:]
            
            logger.info(f"Tool call successful: {server_id}.{tool_name} in {duration:.3f}s")
            return tool_call
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            
            # Create failed call record
            tool_call = ToolCall(
                tool_name=tool_name,
                server_id=server_id,
                arguments=arguments,
                result=None,
                success=False,
                duration=duration,
                timestamp=datetime.now(),
                error_message=error_msg
            )
            
            # Update stats
            self._update_stats(tool_call)
            
            # Store call history
            with self._lock:
                self.tool_calls.append(tool_call)
            
            logger.error(f"Tool call failed: {server_id}.{tool_name}: {error_msg}")
            raise
    
    def _update_stats(self, tool_call: ToolCall) -> None:
        """Update registry statistics"""
        self._registry_stats['total_calls'] += 1
        
        if tool_call.success:
            self._registry_stats['successful_calls'] += 1
        else:
            self._registry_stats['failed_calls'] += 1
        
        # Update rolling average response time
        total_calls = self._registry_stats['total_calls']
        current_avg = self._registry_stats['average_response_time']
        new_avg = ((current_avg * (total_calls - 1)) + tool_call.duration) / total_calls
        self._registry_stats['average_response_time'] = new_avg
    
    def get_call_history(self, limit: Optional[int] = None, server_id: Optional[str] = None) -> List[ToolCall]:
        """Get tool call history"""
        with self._lock:
            calls = self.tool_calls.copy()
        
        # Filter by server if specified
        if server_id:
            calls = [call for call in calls if call.server_id == server_id]
        
        # Sort by timestamp (most recent first)
        calls.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            calls = calls[:limit]
        
        return calls
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        with self._lock:
            server_count = len(self.servers)
            connected_servers = sum(1 for s in self.servers.values() if s.connection_status == "connected")
            total_tools = sum(len(s.tools) for s in self.servers.values())
            
            # Recent activity (last 24 hours)
            recent_cutoff = datetime.now() - timedelta(hours=24)
            recent_calls = [call for call in self.tool_calls if call.timestamp >= recent_cutoff]
            
            return {
                'servers': {
                    'total': server_count,
                    'connected': connected_servers,
                    'disconnected': server_count - connected_servers
                },
                'tools': {
                    'total': total_tools
                },
                'calls': {
                    'total': self._registry_stats['total_calls'],
                    'successful': self._registry_stats['successful_calls'],
                    'failed': self._registry_stats['failed_calls'],
                    'success_rate': (self._registry_stats['successful_calls'] / max(1, self._registry_stats['total_calls'])) * 100,
                    'average_response_time': self._registry_stats['average_response_time'],
                    'recent_24h': len(recent_calls)
                }
            }
    
    def get_server_stats(self, server_id: str) -> Dict[str, Any]:
        """Get statistics for a specific server"""
        server = self.servers.get(server_id)
        if not server:
            return {}
        
        # Get calls for this server
        server_calls = [call for call in self.tool_calls if call.server_id == server_id]
        successful_calls = sum(1 for call in server_calls if call.success)
        
        # Tool usage statistics
        tool_usage = defaultdict(int)
        for call in server_calls:
            tool_usage[call.tool_name] += 1
        
        return {
            'connection_status': server.connection_status,
            'error_message': server.error_message,
            'tool_count': len(server.tools),
            'total_calls': server.call_count,
            'successful_calls': successful_calls,
            'failed_calls': len(server_calls) - successful_calls,
            'last_call_time': server.last_call_time.isoformat() if server.last_call_time else None,
            'tool_usage': dict(tool_usage),
            'uptime': (datetime.now() - server.last_updated).total_seconds() if server.last_updated else 0
        }
    
    async def health_check(self, server_id: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Perform health check on servers"""
        servers_to_check = [server_id] if server_id else list(self.servers.keys())
        results = {}
        
        for sid in servers_to_check:
            server = self.servers.get(sid)
            if not server or not server.adapter:
                results[sid] = {
                    'healthy': False,
                    'error': 'Server not found or not connected'
                }
                continue
            
            try:
                start_time = time.time()
                healthy = await server.adapter.ping()
                response_time = time.time() - start_time
                
                results[sid] = {
                    'healthy': healthy,
                    'response_time': response_time,
                    'tool_count': len(server.tools),
                    'status': server.connection_status
                }
                
                # Update server status
                if healthy:
                    server.connection_status = "connected"
                    server.error_message = None
                else:
                    server.connection_status = "error"
                    server.error_message = "Health check failed"
                
            except Exception as e:
                results[sid] = {
                    'healthy': False,
                    'error': str(e),
                    'response_time': None
                }
                
                server.connection_status = "error"
                server.error_message = str(e)
        
        return results
    
    async def refresh_tools(self, server_id: Optional[str] = None) -> Dict[str, int]:
        """Refresh tool list from servers"""
        servers_to_refresh = [server_id] if server_id else list(self.servers.keys())
        results = {}
        
        for sid in servers_to_refresh:
            server = self.servers.get(sid)
            if not server or not server.adapter:
                results[sid] = 0
                continue
            
            try:
                tools = await server.adapter.get_tools(force_refresh=True)
                server.tools = tools
                server.last_updated = datetime.now()
                results[sid] = len(tools)
                
                logger.info(f"Refreshed {len(tools)} tools for server {sid}")
                
            except Exception as e:
                logger.error(f"Failed to refresh tools for {sid}: {e}")
                results[sid] = 0
        
        return results
    
    async def _start_background_tasks(self) -> None:
        """Start background monitoring tasks"""
        if self._tasks_started:
            logger.warning("Background tasks already started")
            return
            
        try:
            # Clear shutdown event
            self._shutdown_event.clear()
            
            # Start periodic health checks
            health_task = asyncio.create_task(
                self._periodic_health_check(),
                name="registry_health_check"
            )
            self._background_tasks["health_check"] = health_task
            
            # Start periodic tool refresh
            refresh_task = asyncio.create_task(
                self._periodic_tool_refresh(),
                name="registry_tool_refresh"
            )
            self._background_tasks["tool_refresh"] = refresh_task
            
            self._tasks_started = True
            logger.info("Started background registry tasks")
            
        except Exception as e:
            logger.error(f"Failed to start background tasks: {e}")
            await self._cleanup_background_tasks()
            raise
    
    async def _cleanup_background_tasks(self) -> None:
        """Clean up all background tasks properly"""
        if not self._background_tasks:
            return
        
        try:
            logger.info("Cleaning up background registry tasks")
            
            # Signal shutdown
            self._shutdown_event.set()
            
            # Cancel all tasks
            for task_name, task in self._background_tasks.items():
                if not task.done():
                    logger.debug(f"Cancelling task: {task_name}")
                    task.cancel()
            
            # Wait for all tasks to complete with timeout
            if self._background_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*self._background_tasks.values(), return_exceptions=True),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("Some background tasks did not complete within timeout")
                except Exception as e:
                    logger.error(f"Error waiting for background tasks: {e}")
            
            # Clear the tasks dictionary
            self._background_tasks.clear()
            self._tasks_started = False
            
            logger.info("Background registry tasks cleaned up")
            
        except Exception as e:
            logger.error(f"Error during background task cleanup: {e}")
    
    async def _periodic_health_check(self) -> None:
        """Periodic health check background task"""
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Wait for interval or shutdown signal
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=300.0)  # 5 minutes
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    # Timeout reached, perform health check
                    pass
                
                if not self._shutdown_event.is_set():
                    logger.debug("Performing periodic health check")
                    await self.health_check()
                    
        except asyncio.CancelledError:
            logger.debug("Health check task cancelled")
            raise
        except Exception as e:
            logger.error(f"Periodic health check error: {e}")
        finally:
            logger.debug("Health check task finished")
    
    async def _periodic_tool_refresh(self) -> None:
        """Periodic tool refresh background task"""
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Wait for interval or shutdown signal
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=1800.0)  # 30 minutes
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    # Timeout reached, perform refresh
                    pass
                
                if not self._shutdown_event.is_set():
                    logger.debug("Performing periodic tool refresh")
                    await self.refresh_tools()
                    
        except asyncio.CancelledError:
            logger.debug("Tool refresh task cancelled")
            raise
        except Exception as e:
            logger.error(f"Periodic tool refresh error: {e}")
        finally:
            logger.debug("Tool refresh task finished")
    
    def get_categories(self) -> List[str]:
        """Get all available tool categories"""
        categories = set()
        
        for server in self.servers.values():
            for tool in server.tools:
                # Check if tool has category in input schema or metadata
                if hasattr(tool, 'category') and tool.category:
                    categories.add(tool.category)
                elif hasattr(tool.inputSchema, 'properties') and 'category' in tool.inputSchema.properties:
                    categories.add(tool.inputSchema.properties['category'])
        
        return sorted(list(categories))
    
    def get_tools_by_category(self) -> Dict[str, List[Tuple[str, Tool]]]:
        """Group tools by category"""
        categorized = defaultdict(list)
        
        for server_id, server in self.servers.items():
            for tool in server.tools:
                category = getattr(tool, 'category', 'General')
                if not category:
                    category = 'General'
                categorized[category].append((server_id, tool))
        
        return dict(categorized)
    
    def export_config(self, server_id: str) -> Optional[Dict[str, Any]]:
        """Export server configuration"""
        server = self.servers.get(server_id)
        return server.config if server else None
    
    async def test_tool(self, server_id: str, tool_name: str, test_arguments: Optional[Dict[str, Any]] = None) -> ToolCall:
        """Test a tool with sample arguments"""
        # Use provided test arguments or try to generate safe defaults
        if test_arguments is None:
            # Find tool definition
            server = self.servers.get(server_id)
            if not server:
                raise ValueError(f"Server {server_id} not found")
            
            tool = next((t for t in server.tools if t.name == tool_name), None)
            if not tool:
                raise ValueError(f"Tool {tool_name} not found")
            
            # Generate test arguments based on schema
            test_arguments = self._generate_test_arguments(tool)
        
        return await self.call_tool(server_id, tool_name, test_arguments)
    
    def _generate_test_arguments(self, tool: Tool) -> Dict[str, Any]:
        """Generate safe test arguments for a tool"""
        # This is a simple implementation - in practice you'd want more sophisticated
        # argument generation based on the tool's input schema
        test_args = {}
        
        if hasattr(tool, 'inputSchema') and tool.inputSchema:
            schema = tool.inputSchema
            if hasattr(schema, 'properties') and schema.properties:
                for prop_name, prop_schema in schema.properties.items():
                    prop_type = prop_schema.get('type', 'string')
                    
                    if prop_type == 'string':
                        test_args[prop_name] = f"test_{prop_name}"
                    elif prop_type == 'integer':
                        test_args[prop_name] = 1
                    elif prop_type == 'number':
                        test_args[prop_name] = 1.0
                    elif prop_type == 'boolean':
                        test_args[prop_name] = True
                    elif prop_type == 'array':
                        test_args[prop_name] = []
                    elif prop_type == 'object':
                        test_args[prop_name] = {}
        
        return test_args
    
    async def cleanup(self) -> None:
        """Cleanup registry and all connections"""
        try:
            logger.info("Starting registry cleanup")
            
            # Clean up background tasks first
            await self._cleanup_background_tasks()
            
            # Cleanup connection manager
            await self.connection_manager.cleanup()
            
            # Clear registry
            with self._lock:
                self.servers.clear()
                self.tool_calls.clear()
            
            logger.info("Registry cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during registry cleanup: {e}")
    
    def __len__(self) -> int:
        """Return number of registered servers"""
        return len(self.servers)
    
    def __contains__(self, server_id: str) -> bool:
        """Check if server is registered"""
        return server_id in self.servers
    
    def __iter__(self):
        """Iterate over server IDs"""
        return iter(self.servers.keys())
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with proper cleanup"""
        await self.cleanup()