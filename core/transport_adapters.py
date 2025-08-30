import asyncio
import os
import subprocess
import json
import logging
import time
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
#from mcp.types import CallToolRequest, CallToolResult, Tool, GetToolsRequest
from mcp.types import CallToolResult, Tool
logger = logging.getLogger(__name__)

class TransportType(Enum):
    STDIO = "stdio"
    SSE = "sse" 
    HTTP = "http"

@dataclass
class ConnectionStatus:
    connected: bool = False
    last_ping: Optional[float] = None
    error_message: Optional[str] = None
    connection_time: Optional[float] = None
    tool_count: int = 0

class MCPTransportAdapter(ABC):
    """Abstract base class for MCP transport adapters"""
    
    def __init__(self, server_config: Dict[str, Any]):
        self.server_config = server_config
        self.server_id = server_config['id']
        self.transport_config = server_config['transport']
        self.session: Optional[ClientSession] = None
        self.status = ConnectionStatus()
        self._tools_cache: Optional[List[Tool]] = None
        self._last_tools_fetch: Optional[float] = None
        self._tools_cache_ttl = 300  # 5 minutes
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to MCP server"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to MCP server"""
        pass
    
    async def get_tools(self, force_refresh: bool = False) -> List[Tool]:
        """Get available tools from the server with caching"""
        current_time = time.time()
        
        # Check cache validity
        if (not force_refresh and 
            self._tools_cache and 
            self._last_tools_fetch and
            current_time - self._last_tools_fetch < self._tools_cache_ttl):
            return self._tools_cache
        
        try:
            if not self.session:
                raise Exception("No active session")
            
            # Fetch tools from server
            #request = GetToolsRequest()
            #response = await self.session.get_tools(request)
            
            #self._tools_cache = response.tools
            response = await self.session.get_tools()
            self._tools_cache = response
            self._last_tools_fetch = current_time
            self.status.tool_count = len(self._tools_cache)
            
            logger.info(f"Fetched {len(self._tools_cache)} tools from {self.server_id}")
            return self._tools_cache
            
        except Exception as e:
            logger.error(f"Failed to get tools from {self.server_id}: {e}")
            self.status.error_message = str(e)
            return self._tools_cache or []
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> CallToolResult:
        """Call a tool on the MCP server"""
        try:
            if not self.session:
                raise Exception("No active session")
            
            #request = CallToolRequest(name=tool_name,arguments=arguments)
            
            logger.info(f"Calling tool {tool_name} on {self.server_id} with args: {arguments}")
            #result = await self.session.call_tool(request)
            result = await self.session.call_tool(tool_name, arguments)
            logger.info(f"Tool call successful: {tool_name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to call tool {tool_name} on {self.server_id}: {e}")
            raise
    
    async def ping(self) -> bool:
        """Check if connection is alive"""
        try:
            # Try to get tools as a connectivity check
            await self.get_tools(force_refresh=True)
            self.status.last_ping = time.time()
            self.status.connected = True
            self.status.error_message = None
            return True
        except Exception as e:
            self.status.connected = False
            self.status.error_message = str(e)
            self.status.last_ping = time.time()
            return False
    
    def get_status(self) -> ConnectionStatus:
        """Get current connection status"""
        return self.status

class StdioTransportAdapter(MCPTransportAdapter):
    """Transport adapter for stdio-based MCP servers"""
    
    def __init__(self, server_config: Dict[str, Any]):
        super().__init__(server_config)
        self.process: Optional[subprocess.Popen] = None
    
    async def connect(self) -> bool:
        """Connect to stdio MCP server"""
        try:
            command = self.transport_config['command']
            args = self.transport_config.get('args', [])
            env_vars = self.transport_config.get('env', {})
            timeout = self.transport_config.get('timeout', 30)
            
            # Prepare environment
            env = os.environ.copy()
            for key, value in env_vars.items():
                # Handle environment variable substitution
                if value.startswith('${') and value.endswith('}'):
                    env_var_name = value[2:-1]
                    env_value = os.getenv(env_var_name)
                    if env_value is None:
                        logger.warning(f"Environment variable {env_var_name} not found for {self.server_id}")
                        env[key] = ""
                    else:
                        env[key] = env_value
                else:
                    env[key] = value
            
            # Create server parameters
            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=env
            )
            
            logger.info(f"Connecting to stdio server {self.server_id}: {command} {' '.join(args)}")
            
            # Connect with timeout
            connect_start = time.time()
            self.session = await asyncio.wait_for(
                stdio_client(server_params),
                timeout=timeout
            )
            
            self.status.connected = True
            self.status.connection_time = time.time() - connect_start
            self.status.error_message = None
            
            logger.info(f"Successfully connected to {self.server_id} in {self.status.connection_time:.2f}s")
            
            # Initial tool fetch
            await self.get_tools(force_refresh=True)
            
            return True
            
        except asyncio.TimeoutError:
            error_msg = f"Connection timeout for {self.server_id}"
            logger.error(error_msg)
            self.status.error_message = error_msg
            self.status.connected = False
            return False
            
        except Exception as e:
            error_msg = f"Failed to connect to {self.server_id}: {e}"
            logger.error(error_msg)
            self.status.error_message = error_msg
            self.status.connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from stdio server"""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            
            if self.process and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            
            self.status.connected = False
            self.status.error_message = None
            logger.info(f"Disconnected from {self.server_id}")
            
        except Exception as e:
            logger.error(f"Error disconnecting from {self.server_id}: {e}")

class SSETransportAdapter(MCPTransportAdapter):
    """Transport adapter for Server-Sent Events MCP servers"""
    
    async def connect(self) -> bool:
        """Connect to SSE MCP server"""
        try:
            url = self.transport_config['url']
            timeout = self.transport_config.get('timeout', 30)
            
            logger.info(f"Connecting to SSE server {self.server_id}: {url}")
            
            connect_start = time.time()
            self.session = await asyncio.wait_for(
                sse_client(url),
                timeout=timeout
            )
            
            self.status.connected = True
            self.status.connection_time = time.time() - connect_start
            self.status.error_message = None
            
            logger.info(f"Successfully connected to {self.server_id} in {self.status.connection_time:.2f}s")
            
            # Initial tool fetch
            await self.get_tools(force_refresh=True)
            
            return True
            
        except asyncio.TimeoutError:
            error_msg = f"Connection timeout for {self.server_id}"
            logger.error(error_msg)
            self.status.error_message = error_msg
            self.status.connected = False
            return False
            
        except Exception as e:
            error_msg = f"Failed to connect to {self.server_id}: {e}"
            logger.error(error_msg)
            self.status.error_message = error_msg
            self.status.connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from SSE server"""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            
            self.status.connected = False
            self.status.error_message = None
            logger.info(f"Disconnected from {self.server_id}")
            
        except Exception as e:
            logger.error(f"Error disconnecting from {self.server_id}: {e}")

class HTTPTransportAdapter(MCPTransportAdapter):
    """Transport adapter for HTTP-based MCP servers"""
    
    def __init__(self, server_config: Dict[str, Any]):
        super().__init__(server_config)
        self.client: Optional[httpx.AsyncClient] = None
    
    async def connect(self) -> bool:
        """Connect to HTTP MCP server"""
        try:
            url = self.transport_config['url']
            headers = self.transport_config.get('headers', {})
            timeout = self.transport_config.get('timeout', 30)
            
            logger.info(f"Connecting to HTTP server {self.server_id}: {url}")
            
            # Create HTTP client
            self.client = httpx.AsyncClient(
                base_url=url,
                headers=headers,
                timeout=timeout
            )
            
            connect_start = time.time()
            
            # Test connection with a simple request
            response = await self.client.get("/health", timeout=timeout)
            response.raise_for_status()
            
            self.status.connected = True
            self.status.connection_time = time.time() - connect_start
            self.status.error_message = None
            
            logger.info(f"Successfully connected to {self.server_id} in {self.status.connection_time:.2f}s")
            
            # Note: For HTTP transport, we would need to implement the MCP protocol
            # over HTTP. This is a placeholder implementation.
            # In practice, you'd need to implement the full MCP JSON-RPC protocol.
            
            return True
            
        except asyncio.TimeoutError:
            error_msg = f"Connection timeout for {self.server_id}"
            logger.error(error_msg)
            self.status.error_message = error_msg
            self.status.connected = False
            return False
            
        except Exception as e:
            error_msg = f"Failed to connect to {self.server_id}: {e}"
            logger.error(error_msg)
            self.status.error_message = error_msg
            self.status.connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from HTTP server"""
        try:
            if self.client:
                await self.client.aclose()
                self.client = None
            
            self.status.connected = False
            self.status.error_message = None
            logger.info(f"Disconnected from {self.server_id}")
            
        except Exception as e:
            logger.error(f"Error disconnecting from {self.server_id}: {e}")
    
    async def get_tools(self, force_refresh: bool = False) -> List[Tool]:
        """Get tools via HTTP (placeholder implementation)"""
        # This would need to implement MCP protocol over HTTP
        # For now, return empty list
        logger.warning(f"HTTP transport not fully implemented for {self.server_id}")
        return []
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> CallToolResult:
        """Call tool via HTTP (placeholder implementation)"""
        # This would need to implement MCP protocol over HTTP
        raise NotImplementedError("HTTP transport not fully implemented")

class TransportFactory:
    """Factory for creating transport adapters"""
    
    @staticmethod
    def create_adapter(server_config: Dict[str, Any]) -> MCPTransportAdapter:
        """Create appropriate transport adapter based on config"""
        transport_type = server_config['transport']['type']
        
        if transport_type == TransportType.STDIO.value:
            return StdioTransportAdapter(server_config)
        elif transport_type == TransportType.SSE.value:
            return SSETransportAdapter(server_config)
        elif transport_type == TransportType.HTTP.value:
            return HTTPTransportAdapter(server_config)
        else:
            raise ValueError(f"Unsupported transport type: {transport_type}")

class ConnectionManager:
    """Manages multiple transport connections"""
    
    def __init__(self):
        self.adapters: Dict[str, MCPTransportAdapter] = {}
        self._health_check_task: Optional[asyncio.Task] = None
        self._health_check_interval = 60  # seconds
    
    async def add_server(self, server_config: Dict[str, Any]) -> bool:
        """Add and connect to a new server"""
        server_id = server_config['id']
        
        try:
            # Remove existing connection if any
            if server_id in self.adapters:
                await self.remove_server(server_id)
            
            # Create new adapter
            adapter = TransportFactory.create_adapter(server_config)
            
            # Attempt connection
            if await adapter.connect():
                self.adapters[server_id] = adapter
                logger.info(f"Added server {server_id}")
                
                # Start health checks if this is the first server
                if len(self.adapters) == 1:
                    self._start_health_checks()
                
                return True
            else:
                logger.error(f"Failed to connect to server {server_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding server {server_id}: {e}")
            return False
    
    async def remove_server(self, server_id: str) -> bool:
        """Remove and disconnect from a server"""
        try:
            if server_id in self.adapters:
                adapter = self.adapters[server_id]
                await adapter.disconnect()
                del self.adapters[server_id]
                logger.info(f"Removed server {server_id}")
                
                # Stop health checks if no servers remain
                if not self.adapters:
                    self._stop_health_checks()
                
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error removing server {server_id}: {e}")
            return False
    
    def get_adapter(self, server_id: str) -> Optional[MCPTransportAdapter]:
        """Get transport adapter for a server"""
        return self.adapters.get(server_id)
    
    def get_all_adapters(self) -> Dict[str, MCPTransportAdapter]:
        """Get all transport adapters"""
        return self.adapters.copy()
    
    async def get_all_tools(self) -> Dict[str, List[Tool]]:
        """Get tools from all connected servers"""
        all_tools = {}
        
        for server_id, adapter in self.adapters.items():
            try:
                tools = await adapter.get_tools()
                all_tools[server_id] = tools
            except Exception as e:
                logger.error(f"Failed to get tools from {server_id}: {e}")
                all_tools[server_id] = []
        
        return all_tools
    
    def _start_health_checks(self):
        """Start periodic health checks for all connections"""
        if not self._health_check_task:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            logger.info("Started connection health checks")
    
    def _stop_health_checks(self):
        """Stop periodic health checks"""
        if self._health_check_task:
            self._health_check_task.cancel()
            self._health_check_task = None
            logger.info("Stopped connection health checks")
    
    async def _health_check_loop(self):
        """Periodic health check loop"""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                
                for server_id, adapter in self.adapters.items():
                    try:
                        healthy = await adapter.ping()
                        if not healthy:
                            logger.warning(f"Health check failed for {server_id}")
                    except Exception as e:
                        logger.error(f"Health check error for {server_id}: {e}")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
    
    async def cleanup(self):
        """Cleanup all connections"""
        self._stop_health_checks()
        
        for server_id in list(self.adapters.keys()):
            await self.remove_server(server_id)
        
        logger.info("Cleaned up all connections")