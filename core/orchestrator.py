import asyncio
import json
import logging
import os
import re
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from core.mcp_registry import MCPRegistry, ToolCall
from mcp.types import Tool

logger = logging.getLogger(__name__)

class ExecutionMode(Enum):
    SINGLE_TOOL = "single_tool"
    MULTI_TOOL = "multi_tool"
    WORKFLOW = "workflow"
    INTERACTIVE = "interactive"

@dataclass
class ExecutionPlan:
    """Plan for executing user request"""
    mode: ExecutionMode
    steps: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.0
    requires_confirmation: bool = False
    estimated_duration: Optional[float] = None

@dataclass
class ExecutionResult:
    """Result of executing a plan"""
    success: bool
    results: List[Any] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    total_duration: float = 0.0
    error_message: Optional[str] = None
    plan: Optional[ExecutionPlan] = None

class GeminiOrchestrator:
    """Orchestrates MCP tool calls using Gemini AI"""
    
    def __init__(self, registry: MCPRegistry, api_key: Optional[str] = None):
        self.registry = registry
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
        
        # Configure Gemini
        try:
            genai.configure(api_key=self.api_key)
            
            # Initialize model with safety settings
            self.model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",  # Use more reliable model
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                }
            )
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise ValueError(f"Failed to initialize Gemini with provided API key: {e}")
        
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history_length = 50
        
        # Execution statistics
        self.execution_stats = {
            'total_requests': 0,
            'successful_plans': 0,
            'failed_plans': 0,
            'total_tool_calls': 0,
            'average_planning_time': 0.0,
            'average_execution_time': 0.0
        }
    
    def _get_available_tools_context(self) -> str:
        """Generate context string of all available tools"""
        try:
            all_tools = self.registry.get_all_tools()
            
            if not all_tools:
                return "No MCP servers are currently connected."
            
            context_parts = ["Available MCP Servers and Tools:"]
            
            for server_id, tools in all_tools.items():
                server = self.registry.get_server(server_id)
                if not server:
                    continue
                    
                context_parts.append(f"\n## Server: {server.name} (ID: {server_id})")
                context_parts.append(f"Description: {server.description}")
                context_parts.append(f"Status: {server.connection_status}")
                context_parts.append(f"Tools ({len(tools)}):")
                
                for tool in tools:
                    context_parts.append(f"  - {tool.name}: {tool.description or 'No description'}")
                    
                    # Add parameter info if available
                    if hasattr(tool, 'inputSchema') and tool.inputSchema:
                        schema = tool.inputSchema
                        if hasattr(schema, 'properties') and schema.properties:
                            params = list(schema.properties.keys())
                            if params:
                                context_parts.append(f"    Parameters: {', '.join(params)}")
            
            return "\n".join(context_parts)
        except Exception as e:
            logger.error(f"Error getting tools context: {e}")
            return "Error retrieving available tools."
    
    def _create_planning_prompt(self, user_request: str, context: Dict[str, Any]) -> str:
        """Create prompt for Gemini to plan tool execution"""
        
        tools_context = self._get_available_tools_context()
        
        # Recent conversation context
        recent_history = ""
        if self.conversation_history:
            recent_history = "\n\nRecent Conversation:\n"
            for entry in self.conversation_history[-5:]:  # Last 5 exchanges
                recent_history += f"User: {entry.get('user', '')}\n"
                recent_history += f"Assistant: {entry.get('assistant', '')}\n"
        
        # Additional context from user
        user_context = ""
        if context:
            user_context = f"\n\nAdditional Context:\n{json.dumps(context, indent=2)}"
        
        prompt = f"""You are an AI assistant that helps users accomplish tasks by orchestrating calls to MCP (Model Context Protocol) tools. 

{tools_context}
{recent_history}
{user_context}

User Request: "{user_request}"

Analyze the user's request and create an execution plan. Consider:
1. Which tools are needed to fulfill this request
2. What order operations should be performed in
3. What parameters each tool call requires
4. Whether multiple tools need to be chained together
5. If any operations might be dangerous and need confirmation

Respond with a JSON object in this exact format:
{{
  "mode": "single_tool|multi_tool|workflow|interactive",
  "reasoning": "Explain your analysis and approach",
  "confidence": 0.0-1.0,
  "requires_confirmation": true/false,
  "estimated_duration": seconds_estimate,
  "steps": [
    {{
      "server_id": "server_identifier",
      "tool_name": "tool_name",
      "arguments": {{"param1": "value1", "param2": "value2"}},
      "description": "What this step accomplishes",
      "depends_on": ["step_1", "step_2"]
    }}
  ]
}}

Guidelines:
- Use "single_tool" for requests requiring one tool call
- Use "multi_tool" for requests requiring multiple independent tool calls
- Use "workflow" for requests requiring dependent/chained tool calls
- Use "interactive" if you need more information from the user
- Set requires_confirmation=true for potentially destructive operations
- Provide realistic parameter values based on the user's request
- If the request cannot be fulfilled with available tools, explain in reasoning and use empty steps array
- Be specific about server_id and tool_name - they must exactly match available tools
"""
        
        return prompt
    
    async def analyze_request(self, user_request: str, context: Dict[str, Any] = None) -> ExecutionPlan:
        """Analyze user request and create execution plan"""
        start_time = time.time()
        
        try:
            # Generate planning prompt
            prompt = self._create_planning_prompt(user_request, context or {})
            
            # Get plan from Gemini
            logger.info(f"Analyzing request: {user_request[:100]}...")
            
            try:
                response = self.model.generate_content(prompt)
                
                if not response.text:
                    raise ValueError("Empty response from Gemini")
                    
            except Exception as e:
                logger.error(f"Gemini API error: {e}")
                # Return interactive plan for error
                return ExecutionPlan(
                    mode=ExecutionMode.INTERACTIVE,
                    reasoning=f"I'm having trouble connecting to the AI service: {str(e)}. Please try again.",
                    confidence=0.0,
                    steps=[]
                )
            
            planning_time = time.time() - start_time
            self._update_planning_stats(planning_time)
            
            # Parse JSON response
            response_text = response.text.strip()
            
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                # Try to find JSON object directly
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
                else:
                    raise ValueError("No valid JSON found in response")
            
            try:
                plan_data = json.loads(json_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}")
                logger.error(f"Response text: {response_text}")
                return ExecutionPlan(
                    mode=ExecutionMode.INTERACTIVE,
                    reasoning="I had trouble understanding how to help you. Could you please rephrase your request?",
                    confidence=0.0,
                    steps=[]
                )
            
            # Validate plan structure
            required_fields = ['mode', 'reasoning', 'confidence', 'steps']
            for field in required_fields:
                if field not in plan_data:
                    logger.warning(f"Missing required field: {field}")
                    plan_data[field] = self._get_default_value(field)
            
            # Create execution plan
            try:
                plan = ExecutionPlan(
                    mode=ExecutionMode(plan_data['mode']),
                    steps=plan_data['steps'],
                    reasoning=plan_data['reasoning'],
                    confidence=float(plan_data['confidence']),
                    requires_confirmation=plan_data.get('requires_confirmation', False),
                    estimated_duration=plan_data.get('estimated_duration')
                )
            except ValueError as e:
                # Invalid mode, default to interactive
                plan = ExecutionPlan(
                    mode=ExecutionMode.INTERACTIVE,
                    reasoning=plan_data.get('reasoning', 'I need more information to help you.'),
                    confidence=0.0,
                    steps=[]
                )
            
            logger.info(f"Created {plan.mode.value} plan with {len(plan.steps)} steps (confidence: {plan.confidence:.2f})")
            
            return plan
            
        except Exception as e:
            logger.error(f"Failed to analyze request: {e}")
            
            # Return interactive plan to ask for clarification
            return ExecutionPlan(
                mode=ExecutionMode.INTERACTIVE,
                reasoning=f"I encountered an error analyzing your request: {str(e)}. Could you please rephrase or provide more details?",
                confidence=0.0,
                steps=[]
            )
    
    def _get_default_value(self, field: str):
        """Get default value for missing fields"""
        defaults = {
            'mode': 'interactive',
            'reasoning': 'Processing request...',
            'confidence': 0.0,
            'steps': [],
            'requires_confirmation': False
        }
        return defaults.get(field, '')
    
    async def execute_plan(self, plan: ExecutionPlan, user_confirmation: bool = False) -> ExecutionResult:
        """Execute the given plan"""
        start_time = time.time()
        self.execution_stats['total_requests'] += 1
        
        try:
            if plan.requires_confirmation and not user_confirmation:
                return ExecutionResult(
                    success=False,
                    error_message="This operation requires user confirmation",
                    plan=plan
                )
            
            if plan.mode == ExecutionMode.INTERACTIVE:
                return ExecutionResult(
                    success=True,
                    results=[{"message": plan.reasoning}],
                    plan=plan
                )
            
            # Execute based on mode
            if plan.mode == ExecutionMode.SINGLE_TOOL:
                return await self._execute_single_tool(plan)
            elif plan.mode == ExecutionMode.MULTI_TOOL:
                return await self._execute_multi_tool(plan)
            elif plan.mode == ExecutionMode.WORKFLOW:
                return await self._execute_workflow(plan)
            else:
                raise ValueError(f"Unsupported execution mode: {plan.mode}")
                
        except Exception as e:
            logger.error(f"Plan execution failed: {e}")
            self.execution_stats['failed_plans'] += 1
            
            return ExecutionResult(
                success=False,
                error_message=str(e),
                total_duration=time.time() - start_time,
                plan=plan
            )
        finally:
            execution_time = time.time() - start_time
            self._update_execution_stats(execution_time)
    
    async def _execute_single_tool(self, plan: ExecutionPlan) -> ExecutionResult:
        """Execute single tool call"""
        if not plan.steps:
            raise ValueError("No steps in plan")
        
        step = plan.steps[0]
        
        # Validate step has required fields
        if 'server_id' not in step or 'tool_name' not in step or 'arguments' not in step:
            raise ValueError("Invalid step format - missing required fields")
        
        try:
            tool_call = await self.registry.call_tool(
                step['server_id'],
                step['tool_name'],
                step['arguments']
            )
            
            self.execution_stats['successful_plans'] += 1
            self.execution_stats['total_tool_calls'] += 1
            
            return ExecutionResult(
                success=tool_call.success,
                results=[tool_call.result],
                tool_calls=[tool_call],
                total_duration=tool_call.duration,
                plan=plan
            )
        except Exception as e:
            self.execution_stats['failed_plans'] += 1
            raise
    
    async def _execute_multi_tool(self, plan: ExecutionPlan) -> ExecutionResult:
        """Execute multiple independent tool calls"""
        tool_calls = []
        results = []
        
        # Execute all tools concurrently
        tasks = []
        for step in plan.steps:
            if 'server_id' not in step or 'tool_name' not in step or 'arguments' not in step:
                logger.warning(f"Skipping invalid step: {step}")
                continue
                
            task = self.registry.call_tool(
                step['server_id'],
                step['tool_name'],
                step['arguments']
            )
            tasks.append(task)
        
        if not tasks:
            raise ValueError("No valid steps to execute")
        
        completed_calls = await asyncio.gather(*tasks, return_exceptions=True)
        
        for call in completed_calls:
            if isinstance(call, Exception):
                tool_calls.append(ToolCall(
                    tool_name="unknown",
                    server_id="unknown",
                    arguments={},
                    result=None,
                    success=False,
                    duration=0.0,
                    timestamp=datetime.now(),
                    error_message=str(call)
                ))
                results.append(None)
            else:
                tool_calls.append(call)
                results.append(call.result)
        
        success = all(call.success for call in tool_calls)
        total_duration = sum(call.duration for call in tool_calls)
        
        if success:
            self.execution_stats['successful_plans'] += 1
        else:
            self.execution_stats['failed_plans'] += 1
        
        self.execution_stats['total_tool_calls'] += len(tool_calls)
        
        return ExecutionResult(
            success=success,
            results=results,
            tool_calls=tool_calls,
            total_duration=total_duration,
            plan=plan
        )
    
    async def _execute_workflow(self, plan: ExecutionPlan) -> ExecutionResult:
        """Execute workflow with dependencies"""
        tool_calls = []
        results = []
        step_results = {}  # Store results by step for dependency resolution
        
        # Sort steps by dependencies (simple topological sort)
        sorted_steps = self._sort_workflow_steps(plan.steps)
        
        for i, step in enumerate(sorted_steps):
            if 'server_id' not in step or 'tool_name' not in step or 'arguments' not in step:
                logger.warning(f"Skipping invalid step: {step}")
                continue
                
            try:
                # Resolve arguments with results from previous steps
                resolved_args = self._resolve_step_arguments(
                    step['arguments'], 
                    step_results,
                    step.get('depends_on', [])
                )
                
                tool_call = await self.registry.call_tool(
                    step['server_id'],
                    step['tool_name'],
                    resolved_args
                )
                
                tool_calls.append(tool_call)
                results.append(tool_call.result)
                step_results[f"step_{i+1}"] = tool_call.result
                
                if not tool_call.success:
                    # Stop workflow on failure
                    break
                    
            except Exception as e:
                logger.error(f"Workflow step {i+1} failed: {e}")
                failed_call = ToolCall(
                    tool_name=step.get('tool_name', 'unknown'),
                    server_id=step.get('server_id', 'unknown'),
                    arguments=step.get('arguments', {}),
                    result=None,
                    success=False,
                    duration=0.0,
                    timestamp=datetime.now(),
                    error_message=str(e)
                )
                tool_calls.append(failed_call)
                results.append(None)
                break
        
        success = all(call.success for call in tool_calls)
        total_duration = sum(call.duration for call in tool_calls)
        
        if success:
            self.execution_stats['successful_plans'] += 1
        else:
            self.execution_stats['failed_plans'] += 1
        
        self.execution_stats['total_tool_calls'] += len(tool_calls)
        
        return ExecutionResult(
            success=success,
            results=results,
            tool_calls=tool_calls,
            total_duration=total_duration,
            plan=plan
        )
    
    def _sort_workflow_steps(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simple topological sort for workflow steps"""
        # For now, just return steps as-is
        # In a full implementation, you'd do proper dependency resolution
        return steps
    
    def _resolve_step_arguments(self, arguments: Dict[str, Any], step_results: Dict[str, Any], dependencies: List[str]) -> Dict[str, Any]:
        """Resolve step arguments with results from previous steps"""
        resolved = arguments.copy()
        
        # Simple variable substitution
        for key, value in resolved.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                var_name = value[2:-1]
                if var_name in step_results:
                    resolved[key] = step_results[var_name]
        
        return resolved
    
    async def process_request(self, user_request: str, context: Dict[str, Any] = None, auto_confirm: bool = False) -> ExecutionResult:
        """Process a user request end-to-end"""
        try:
            # Add to conversation history
            self._add_to_history(user_request, "")
            
            # Analyze request
            plan = await self.analyze_request(user_request, context)
            
            # Execute plan
            result = await self.execute_plan(plan, user_confirmation=auto_confirm)
            
            # Update conversation history with result
            if result.success:
                summary = f"Successfully executed {len(result.tool_calls)} tool calls"
            else:
                summary = f"Execution failed: {result.error_message}"
            
            self._update_history_response(summary)
            
            return result
            
        except Exception as e:
            logger.error(f"Request processing failed: {e}")
            return ExecutionResult(
                success=False,
                error_message=str(e)
            )
    
    def _add_to_history(self, user_message: str, assistant_response: str):
        """Add exchange to conversation history"""
        self.conversation_history.append({
            'user': user_message,
            'assistant': assistant_response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Trim history if needed
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def _update_history_response(self, response: str):
        """Update the last history entry with response"""
        if self.conversation_history:
            self.conversation_history[-1]['assistant'] = response
    
    def _update_planning_stats(self, planning_time: float):
        """Update planning statistics"""
        total = self.execution_stats['total_requests']
        if total > 0:
            current_avg = self.execution_stats['average_planning_time']
            new_avg = ((current_avg * (total - 1)) + planning_time) / total
            self.execution_stats['average_planning_time'] = new_avg
    
    def _update_execution_stats(self, execution_time: float):
        """Update execution statistics"""
        total = self.execution_stats['total_requests']
        if total > 0:
            current_avg = self.execution_stats['average_execution_time']
            new_avg = ((current_avg * (total - 1)) + execution_time) / total
            self.execution_stats['average_execution_time'] = new_avg
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        return self.execution_stats.copy()
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
    
    async def suggest_tools(self, user_query: str, limit: int = 5) -> List[Tuple[str, Tool, float]]:
        """Suggest relevant tools for a user query"""
        try:
            # Simple keyword matching for now
            # In production, you might use embeddings or more sophisticated matching
            suggestions = []
            
            query_words = set(user_query.lower().split())
            all_tools = self.registry.get_all_tools()
            
            for server_id, tools in all_tools.items():
                for tool in tools:
                    # Calculate relevance score
                    tool_words = set((tool.name + " " + (tool.description or "")).lower().split())
                    overlap = len(query_words & tool_words)
                    score = overlap / max(len(query_words), 1)
                    
                    if score > 0:
                        suggestions.append((server_id, tool, score))
            
            # Sort by relevance and return top results
            suggestions.sort(key=lambda x: x[2], reverse=True)
            return suggestions[:limit]
            
        except Exception as e:
            logger.error(f"Tool suggestion failed: {e}")
            return []