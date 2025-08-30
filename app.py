import streamlit as st
import asyncio
import json
import os
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our core modules
from core.config_loader import ConfigLoader
from core.mcp_registry import MCPRegistry
from core.orchestrator import GeminiOrchestrator, ExecutionMode

# Page configuration
st.set_page_config(
    page_title="Universal MCP Client",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .server-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: white;
    }
    .tool-card {
        background: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class MCPStreamlitApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        self.config_loader = None
        self.registry = None
        self.orchestrator = None
        self._initialize_session_state()
        self._setup_cleanup_handlers()

    def _setup_cleanup_handlers(self):
        """Set up cleanup handlers for proper shutdown"""
        # Register cleanup on normal exit
        atexit.register(self._cleanup_sync)

        # Register cleanup on SIGTERM/SIGINT
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, cleaning up...")
            self._cleanup_sync()
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def _cleanup_sync(self):
        """Synchronous cleanup wrapper"""
        try:
            if self.registry:
                # Create new event loop if none exists
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_closed():
                        raise RuntimeError("Event loop is closed")
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Run cleanup
                loop.run_until_complete(self.registry.cleanup())
                logger.info("Registry cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")    
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
        if 'user_id' not in st.session_state:
            st.session_state.user_id = "default_user"
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "chat"
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
        if 'execution_history' not in st.session_state:
            st.session_state.execution_history = []
        if 'auto_confirm' not in st.session_state:
            st.session_state.auto_confirm = False
        if 'gemini_api_key' not in st.session_state:
            st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        if 'initialization_attempted' not in st.session_state:
            st.session_state.initialization_attempted = False
    
    async def initialize_app(self):
        """Initialize the application components"""
        if st.session_state.initialized:
            return True
            
        # Don't re-attempt if already tried with current API key
        if st.session_state.initialization_attempted and not st.session_state.gemini_api_key:
            return False
            
        # Check for API key first
        if not st.session_state.gemini_api_key:
            return False
            
        try:
            with st.spinner("Initializing Universal MCP Client..."):
                # Initialize components
                self.config_loader = ConfigLoader()
                self.registry = MCPRegistry(self.config_loader)
                
                # Use session state API key
                self.orchestrator = GeminiOrchestrator(self.registry, st.session_state.gemini_api_key)
                
                # Initialize registry with configs
                await self.registry.initialize(st.session_state.user_id)
                
                st.session_state.initialized = True
                st.session_state.initialization_attempted = True
                st.success("✅ Universal MCP Client initialized successfully!")
                return True
                
        except Exception as e:
            st.error(f"❌ Initialization failed: {e}")
            logger.error(f"App initialization failed: {e}")
            st.session_state.initialization_attempted = True
            return False
    
    def render_sidebar(self):
        """Render the sidebar navigation"""
        with st.sidebar:
            st.markdown("### 🔧 Universal MCP Client")
            
            # API Key input
            new_api_key = st.text_input(
                "🔑 Gemini API Key",
                value=st.session_state.gemini_api_key,
                type="password",
                placeholder="Enter your Gemini API key...",
                help="Get your API key from Google AI Studio"
            )
            
            # Update API key if changed
            if new_api_key != st.session_state.gemini_api_key:
                st.session_state.gemini_api_key = new_api_key
                st.session_state.initialized = False
                st.session_state.initialization_attempted = False
                st.rerun()
            
            st.markdown("---")
            
            # Navigation - Always show buttons
            pages = {
                "💬 Chat Interface": "chat",
                "🏠 Dashboard": "dashboard", 
                "⚙️ Server Management": "servers",
                "🔍 Tool Explorer": "tools",
                "📊 Analytics": "analytics",
                "⚡ Config Uploader": "config"
            }
            
            for label, page_key in pages.items():
                is_current = st.session_state.current_page == page_key
                if st.button(
                    label, 
                    key=f"nav_{page_key}", 
                    use_container_width=True,
                    type="primary" if is_current else "secondary"
                ):
                    st.session_state.current_page = page_key
                    st.rerun()
            
            st.markdown("---")
            
            # Quick stats - only show if initialized
            if st.session_state.initialized and self.registry:
                try:
                    stats = self.registry.get_stats()
                    st.markdown("### 📈 Quick Stats")
                    st.metric("Connected Servers", stats['servers']['connected'])
                    st.metric("Total Tools", stats['tools']['total'])
                    st.metric("Success Rate", f"{stats['calls']['success_rate']:.1f}%")
                except Exception as e:
                    st.warning(f"Could not load stats: {e}")
            
            st.markdown("---")
            
            # Settings
            st.markdown("### ⚙️ Settings")
            st.session_state.auto_confirm = st.checkbox(
                "Auto-confirm operations",
                value=st.session_state.auto_confirm,
                help="Automatically confirm tool calls that normally require confirmation"
            )
            
            # Clear data options - only show if initialized
            if st.session_state.initialized:
                if st.button("🗑️ Clear Chat History"):
                    st.session_state.chat_messages = []
                    if self.orchestrator:
                        self.orchestrator.clear_conversation_history()
                    st.success("Chat history cleared!")
                
                if st.button("🔄 Reload Configs"):
                    if self.config_loader and self.registry:
                        try:
                            self.config_loader.reload_configs(st.session_state.user_id)
                            asyncio.run(self.registry.reload_from_config(st.session_state.user_id))
                            st.success("Configs reloaded!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Reload failed: {e}")
    
    def render_chat_interface(self):
        """Render the main chat interface"""
        st.markdown('<div class="main-header">💬 Chat with MCP Tools</div>', unsafe_allow_html=True)
        
        if not st.session_state.initialized:
            if not st.session_state.gemini_api_key:
                st.info("👈 Please enter your Gemini API key in the sidebar to start chatting!")
            else:
                st.info("⏳ Initializing... Please wait.")
            return
        
        # Chat messages container
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    
                    # Show execution details for assistant messages
                    if message["role"] == "assistant" and "execution_result" in message:
                        result = message["execution_result"]
                        
                        with st.expander("🔍 Execution Details", expanded=False):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Success", "✅" if result.success else "❌")
                            with col2:
                                st.metric("Tool Calls", len(result.tool_calls))
                            with col3:
                                st.metric("Duration", f"{result.total_duration:.2f}s")
                            
                            if result.tool_calls:
                                st.markdown("**Tool Calls:**")
                                for call in result.tool_calls:
                                    status_emoji = "✅" if call.success else "❌"
                                    st.markdown(f"{status_emoji} `{call.server_id}.{call.tool_name}` ({call.duration:.2f}s)")
                                    
                                    if not call.success and call.error_message:
                                        st.error(f"Error: {call.error_message}")
        
        # Chat input
        if user_input := st.chat_input("Ask me to use any MCP tool..."):
            # Add user message
            st.session_state.chat_messages.append({
                "role": "user", 
                "content": user_input,
                "timestamp": datetime.now()
            })
            
            # Process with orchestrator
            with st.chat_message("assistant"):
                with st.spinner("Thinking and planning..."):
                    try:
                        # Create an async function to run the orchestrator
                        async def process_message():
                            return await self.orchestrator.process_request(
                                user_input, 
                                auto_confirm=st.session_state.auto_confirm
                            )
                        
                        # Run the async function
                        result = asyncio.create_task(process_message())
                        result = asyncio.get_event_loop().run_until_complete(result)
                        
                        if result.success:
                            if result.plan and result.plan.mode == ExecutionMode.INTERACTIVE:
                                # Interactive response
                                response = result.results[0]["message"] if result.results else result.plan.reasoning
                            else:
                                # Successful execution
                                response = self._format_execution_response(result)
                            
                            st.markdown(response)
                            
                            # Show results
                            if result.results and result.plan.mode != ExecutionMode.INTERACTIVE:
                                st.markdown("**Results:**")
                                for i, res in enumerate(result.results):
                                    if res:
                                        st.json(res)
                        else:
                            if result.plan and result.plan.requires_confirmation and not st.session_state.auto_confirm:
                                st.warning("⚠️ This operation requires confirmation. Enable 'Auto-confirm operations' in the sidebar or confirm manually.")
                                response = f"Operation requires confirmation: {result.plan.reasoning}"
                            else:
                                st.error(f"❌ Execution failed: {result.error_message}")
                                response = f"I encountered an error: {result.error_message}"
                        
                        # Add assistant response
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": response,
                            "timestamp": datetime.now(),
                            "execution_result": result
                        })
                        
                    except Exception as e:
                        error_msg = f"Failed to process request: {e}"
                        st.error(error_msg)
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "timestamp": datetime.now()
                        })
                    
                    st.rerun()
    
    def _format_execution_response(self, result) -> str:
        """Format execution result into a readable response"""
        if not result.tool_calls:
            return "I completed the task but didn't make any tool calls."
        
        if len(result.tool_calls) == 1:
            call = result.tool_calls[0]
            return f"✅ I used `{call.server_id}.{call.tool_name}` to complete your request."
        else:
            successful = sum(1 for call in result.tool_calls if call.success)
            total = len(result.tool_calls)
            return f"✅ I executed {successful}/{total} tool calls to complete your request."
    
    def render_dashboard(self):
        """Render the main dashboard"""
        st.markdown('<div class="main-header">🏠 Dashboard</div>', unsafe_allow_html=True)
        
        if not st.session_state.initialized:
            st.info("👈 Please initialize the application first by entering your API key.")
            return
        
        if not self.registry:
            st.warning("Registry not initialized")
            return
        
        try:
            # Get statistics
            registry_stats = self.registry.get_stats()
            orchestrator_stats = self.orchestrator.get_stats() if self.orchestrator else {}
            
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Connected Servers", registry_stats['servers']['connected'])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Available Tools", registry_stats['tools']['total'])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Total Calls", registry_stats['calls']['total'])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Success Rate", f"{registry_stats['calls']['success_rate']:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Recent activity and charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Call Statistics")
                
                if registry_stats['calls']['total'] > 0:
                    # Success rate pie chart
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=['Successful', 'Failed'],
                        values=[registry_stats['calls']['successful'], registry_stats['calls']['failed']],
                        hole=0.3,
                        marker_colors=['#28a745', '#dc3545']
                    )])
                    fig_pie.update_layout(title="Success Rate", height=300)
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("No tool calls yet")
            
            with col2:
                st.subheader("🕐 Recent Activity")
                
                # Get recent call history
                recent_calls = self.registry.get_call_history(limit=10)
                
                if recent_calls:
                    for call in recent_calls:
                        status_emoji = "✅" if call.success else "❌"
                        time_ago = self._time_ago(call.timestamp)
                        
                        st.markdown(f"""
                        <div class="tool-card">
                            {status_emoji} <strong>{call.server_id}.{call.tool_name}</strong><br>
                            <small>{time_ago} • {call.duration:.2f}s</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No recent activity")
            
            st.markdown("---")
            
            # Server status overview
            st.subheader("🖥️ Server Status")
            
            servers = self.registry.get_all_servers()
            if servers:
                for server_id, server in servers.items():
                    status_color = "#28a745" if server.connection_status == "connected" else "#dc3545"
                    
                    st.markdown(f"""
                    <div class="server-card" style="background: linear-gradient(135deg, {status_color} 0%, {status_color}aa 100%);">
                        <h4>{server.name}</h4>
                        <p>{server.description}</p>
                        <small>Status: {server.connection_status} • Tools: {len(server.tools)} • Calls: {server.call_count}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No servers configured. Go to Server Management to add some!")
                
        except Exception as e:
            st.error(f"Error loading dashboard: {e}")
    
    def render_server_management(self):
        """Render server management interface"""
        st.markdown('<div class="main-header">⚙️ Server Management</div>', unsafe_allow_html=True)
        
        if not st.session_state.initialized:
            st.info("👈 Please initialize the application first by entering your API key.")
            return
        
        if not self.registry:
            st.warning("Registry not initialized")
            return
        
        try:
            # Server list
            servers = self.registry.get_all_servers()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📋 Registered Servers")
                
                if servers:
                    for server_id, server in servers.items():
                        with st.expander(f"🖥️ {server.name} ({server_id})", expanded=False):
                            col_info, col_actions = st.columns([3, 1])
                            
                            with col_info:
                                st.write(f"**Description:** {server.description}")
                                st.write(f"**Status:** {server.connection_status}")
                                st.write(f"**Tools:** {len(server.tools)}")
                                st.write(f"**Total Calls:** {server.call_count}")
                                
                                if server.error_message:
                                    st.error(f"Error: {server.error_message}")
                            
                            with col_actions:
                                if st.button(f"🔄 Refresh", key=f"refresh_{server_id}"):
                                    try:
                                        # Run async function synchronously
                                        loop = asyncio.new_event_loop()
                                        asyncio.set_event_loop(loop)
                                        loop.run_until_complete(self.registry.refresh_tools(server_id))
                                        loop.close()
                                        st.success(f"Refreshed {server_id}")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Refresh failed: {e}")
                                
                                if st.button(f"🗑️ Remove", key=f"remove_{server_id}"):
                                    try:
                                        # Run async function synchronously
                                        loop = asyncio.new_event_loop()
                                        asyncio.set_event_loop(loop)
                                        loop.run_until_complete(self.registry.remove_server(server_id))
                                        loop.close()
                                        st.success(f"Removed {server_id}")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Remove failed: {e}")
                                
                                if st.button(f"🔍 Health Check", key=f"health_{server_id}"):
                                    try:
                                        # Run async function synchronously
                                        loop = asyncio.new_event_loop()
                                        asyncio.set_event_loop(loop)
                                        health = loop.run_until_complete(self.registry.health_check(server_id))
                                        loop.close()
                                        
                                        if health[server_id]['healthy']:
                                            st.success("Healthy!")
                                        else:
                                            st.error(f"Unhealthy: {health[server_id].get('error', 'Unknown error')}")
                                    except Exception as e:
                                        st.error(f"Health check failed: {e}")
                else:
                    st.info("No servers registered. Upload a configuration to get started!")
            
            with col2:
                st.subheader("📊 Server Statistics")
                
                if servers:
                    stats_data = []
                    for server_id, server in servers.items():
                        stats_data.append({
                            'Server': server.name,
                            'Status': server.connection_status,
                            'Tools': len(server.tools),
                            'Calls': server.call_count
                        })
                    
                    df = pd.DataFrame(stats_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Tools per server chart
                    if len(stats_data) > 1:
                        fig = px.bar(df, x='Server', y='Tools', title="Tools per Server")
                        st.plotly_chart(fig, use_container_width=True)
                        
        except Exception as e:
            st.error(f"Error in server management: {e}")
    
    def render_tool_explorer(self):
        """Render tool exploration interface"""
        st.markdown('<div class="main-header">🔍 Tool Explorer</div>', unsafe_allow_html=True)
        
        if not st.session_state.initialized:
            st.info("👈 Please initialize the application first by entering your API key.")
            return
        
        if not self.registry:
            st.warning("Registry not initialized")
            return
        
        try:
            # Search and filters
            col1, col2 = st.columns([3, 1])
            
            with col1:
                search_query = st.text_input("🔍 Search tools by name or description", placeholder="e.g., create payment, file operations")
            
            with col2:
                categories = self.registry.get_categories()
                selected_category = st.selectbox("📂 Category", ["All"] + categories)
            
            # Get all tools
            all_tools = self.registry.get_all_tools()
            
            # Apply search and filters
            filtered_tools = []
            
            for server_id, tools in all_tools.items():
                server = self.registry.get_server(server_id)
                if not server:
                    continue
                
                for tool in tools:
                    # Apply search filter
                    if search_query:
                        query_lower = search_query.lower()
                        if (query_lower not in tool.name.lower() and 
                            query_lower not in (tool.description or "").lower()):
                            continue
                    
                    # Apply category filter
                    if selected_category != "All":
                        tool_category = getattr(tool, 'category', 'General')
                        if tool_category != selected_category:
                            continue
                    
                    filtered_tools.append((server_id, server, tool))
            
            # Display results
            st.write(f"Found {len(filtered_tools)} tools")
            
            if filtered_tools:
                # Group by server
                tools_by_server = {}
                for server_id, server, tool in filtered_tools:
                    if server_id not in tools_by_server:
                        tools_by_server[server_id] = []
                    tools_by_server[server_id].append((server, tool))
                
                for server_id, server_tools in tools_by_server.items():
                    server_name = server_tools[0][0].name
                    
                    with st.expander(f"🖥️ {server_name} ({len(server_tools)} tools)", expanded=True):
                        for server, tool in server_tools:
                            self._render_tool_card(server_id, tool)
            else:
                st.info("No tools found matching your criteria")
                
        except Exception as e:
            st.error(f"Error in tool explorer: {e}")
    
    def _render_tool_card(self, server_id: str, tool):
        """Render a single tool card"""
        try:
            with st.container():
                st.markdown(f"""
                <div class="tool-card">
                    <h4>🔧 {tool.name}</h4>
                    <p>{tool.description or 'No description available'}</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    # Show parameters if available
                    if hasattr(tool, 'inputSchema') and tool.inputSchema:
                        schema = tool.inputSchema
                        if hasattr(schema, 'properties') and schema.properties:
                            params = list(schema.properties.keys())
                            st.write(f"**Parameters:** {', '.join(params)}")
                
                with col2:
                    if st.button(f"📋 Schema", key=f"schema_{server_id}_{tool.name}"):
                        st.json(tool.inputSchema.__dict__ if hasattr(tool, 'inputSchema') else {})
                
                with col3:
                    if st.button(f"🧪 Test", key=f"test_{server_id}_{tool.name}"):
                        try:
                            # Run async function synchronously
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            test_result = loop.run_until_complete(self.registry.test_tool(server_id, tool.name))
                            loop.close()
                            
                            if test_result.success:
                                st.success("Test successful!")
                                st.json(test_result.result)
                            else:
                                st.error(f"Test failed: {test_result.error_message}")
                        except Exception as e:
                            st.error(f"Test error: {e}")
        except Exception as e:
            st.error(f"Error rendering tool card: {e}")
    
    def render_analytics(self):
        """Render analytics and insights"""
        st.markdown('<div class="main-header">📊 Analytics</div>', unsafe_allow_html=True)
        
        if not st.session_state.initialized:
            st.info("👈 Please initialize the application first by entering your API key.")
            return
        
        if not self.registry:
            st.warning("Registry not initialized")
            return
        
        try:
            # Time range selector
            col1, col2 = st.columns(2)
            
            with col1:
                time_range = st.selectbox("📅 Time Range", ["Last 24 hours", "Last 7 days", "Last 30 days", "All time"])
            
            with col2:
                refresh_data = st.button("🔄 Refresh Data")
            
            # Get call history
            call_history = self.registry.get_call_history()
            
            if not call_history:
                st.info("No data available yet. Make some tool calls to see analytics!")
                return
            
            # Filter by time range
            now = datetime.now()
            if time_range == "Last 24 hours":
                cutoff = now - timedelta(hours=24)
            elif time_range == "Last 7 days":
                cutoff = now - timedelta(days=7)
            elif time_range == "Last 30 days":
                cutoff = now - timedelta(days=30)
            else:
                cutoff = None
            
            if cutoff:
                filtered_calls = [call for call in call_history if call.timestamp >= cutoff]
            else:
                filtered_calls = call_history
            
            if not filtered_calls:
                st.info(f"No data available for {time_range.lower()}")
                return
            
            # Analytics metrics
            total_calls = len(filtered_calls)
            successful_calls = sum(1 for call in filtered_calls if call.success)
            avg_duration = sum(call.duration for call in filtered_calls) / total_calls
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Calls", total_calls)
            with col2:
                st.metric("Success Rate", f"{(successful_calls/total_calls)*100:.1f}%")
            with col3:
                st.metric("Avg Duration", f"{avg_duration:.2f}s")
            with col4:
                unique_tools = len(set(f"{call.server_id}.{call.tool_name}" for call in filtered_calls))
                st.metric("Unique Tools Used", unique_tools)
            
            st.markdown("---")
            
            # Charts would go here (simplified for now)
            st.subheader("📈 Call Statistics")
            st.info("Charts and detailed analytics will be displayed here.")
                    
        except Exception as e:
            st.error(f"Error in analytics: {e}")
    
    def render_config_uploader(self):
        """Render configuration upload interface"""
        st.markdown('<div class="main-header">⚡ Configuration Uploader</div>', unsafe_allow_html=True)
        
        st.markdown("""
        Upload JSON configuration files to add new MCP servers. Each file should follow the schema:
        - `id`: Unique server identifier
        - `name`: Human-readable server name  
        - `transport`: Connection details (stdio, sse, or http)
        - `tools`: Array of tool definitions
        """)
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose JSON configuration files",
            type=['json'],
            accept_multiple_files=True,
            help="Upload one or more JSON configuration files"
        )
        
        if uploaded_files:
            st.subheader("📄 Uploaded Files")
            
            for uploaded_file in uploaded_files:
                try:
                    # Parse JSON
                    file_content = uploaded_file.read()
                    config_data = json.loads(file_content)
                    config_id = config_data.get('id', uploaded_file.name.replace('.json', ''))
                    
                    # Validate config
                    if self.config_loader and st.session_state.initialized:
                        try:
                            # Save config
                            success = self.config_loader.save_config(
                                config_id, 
                                config_data, 
                                st.session_state.user_id
                            )
                            
                            if success:
                                st.success(f"✅ Saved configuration: {config_data.get('name', config_id)}")
                                
                                # Try to add to registry
                                if self.registry:
                                    # Run async function synchronously
                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)
                                    add_success = loop.run_until_complete(self.registry.add_server(config_data))
                                    loop.close()
                                    
                                    if add_success:
                                        st.success(f"🔗 Connected to server: {config_id}")
                                    else:
                                        st.warning(f"⚠️ Config saved but connection failed: {config_id}")
                            else:
                                st.error(f"❌ Failed to save configuration: {config_id}")
                                
                        except Exception as e:
                            st.error(f"❌ Validation failed for {uploaded_file.name}: {e}")
                    else:
                        st.warning("Please initialize the application first to save configurations.")
                    
                    # Show config preview
                    with st.expander(f"📋 Preview: {config_data.get('name', config_id)}", expanded=False):
                        st.json(config_data)
                        
                except json.JSONDecodeError as e:
                    st.error(f"❌ Invalid JSON in {uploaded_file.name}: {e}")
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {e}")
        
        st.markdown("---")
        
        # Example configurations (same as before - showing working examples)
        st.subheader("📖 Example Configurations")
        
        example_configs = {
            "File System": {
                "id": "filesystem",
                "name": "File System Operations", 
                "description": "Read, write, and manage files",
                "transport": {
                    "type": "stdio",
                    "command": "echo",
                    "args": ["Mock filesystem server - not actually connected"]
                },
                "tools": [
                    {
                        "name": "read_file",
                        "description": "Read contents of a file"
                    }
                ]
            }
        }
        
        for name, config in example_configs.items():
            with st.expander(f"📄 {name} Example", expanded=False):
                st.json(config)
    
    def _time_ago(self, timestamp: datetime) -> str:
        """Format timestamp as 'time ago' string"""
        try:
            now = datetime.now()
            diff = now - timestamp
            
            if diff.total_seconds() < 60:
                return "just now"
            elif diff.total_seconds() < 3600:
                minutes = int(diff.total_seconds() // 60)
                return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
            elif diff.days < 1:
                hours = int(diff.total_seconds() // 3600)
                return f"{hours} hour{'s' if hours != 1 else ''} ago"
            elif diff.days < 7:
                return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
            else:
                return timestamp.strftime("%Y-%m-%d %H:%M")
        except Exception:
            return "unknown"
    
    async def run_async_initialization(self):
        """Run async initialization properly"""
        try:
            return await self.initialize_app()
        except Exception as e:
            st.error(f"Async initialization failed: {e}")
            return False
    
    def run(self):
        """Run the Streamlit application"""
        try:
            # Render sidebar first
            self.render_sidebar()
            
            # Check if API key is provided
            if not st.session_state.gemini_api_key:
                # Show welcome message when no API key
                st.markdown('<div class="main-header">🔧 Universal MCP Client</div>', unsafe_allow_html=True)
                st.markdown("""
                ## Welcome to Universal MCP Client! 
                
                This tool lets you interact with any MCP (Model Context Protocol) server through a natural language interface powered by Google's Gemini AI.
                
                ### Getting Started:
                1. **Get a Gemini API key** from [Google AI Studio](https://aistudio.google.com/app/apikey)
                2. **Enter your API key** in the sidebar ➡️
                3. **Upload or configure** your MCP servers
                4. **Start chatting** with your tools!
                
                ### What you can do:
                - Connect to any MCP server (File System, Web Search, etc.)
                - Use natural language to interact with tools
                - Chain multiple operations together
                - Monitor usage and analytics
                """)
                return
            
            # Try to initialize if we have an API key
            if st.session_state.gemini_api_key and not st.session_state.initialized:
                try:
                    # Create and run event loop for initialization
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    initialized = loop.run_until_complete(self.initialize_app())
                    loop.close()
                    
                    if not initialized and not st.session_state.initialization_attempted:
                        st.warning("⚠️ Failed to initialize. Please check your API key.")
                        return
                except Exception as e:
                    st.error(f"Initialization error: {e}")
                    return
            
            # Render main content based on current page
            page = st.session_state.current_page
            
            try:
                if page == "chat":
                    self.render_chat_interface()
                elif page == "dashboard":
                    self.render_dashboard()
                elif page == "servers":
                    self.render_server_management()
                elif page == "tools":
                    self.render_tool_explorer()
                elif page == "analytics":
                    self.render_analytics()
                elif page == "config":
                    self.render_config_uploader()
                else:
                    st.error(f"Unknown page: {page}")
            except Exception as e:
                st.error(f"Error rendering page '{page}': {e}")
                logger.error(f"Page rendering error for '{page}': {e}")

        except Exception as e:
            st.error(f"Application error: {e}")
            logger.error(f"App error: {e}")

# Main entry point
if __name__ == "__main__":
    app = MCPStreamlitApp()
    app.run()