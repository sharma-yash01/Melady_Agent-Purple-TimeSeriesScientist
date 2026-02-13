"""
Prompt Post-Processing Module

This module provides extensible prompt processing for workshop participants.
Modify the post_process_prompt() function to customize how the raw prompt
from the green agent is processed before being sent to the LLM.

This is the extension point for adding:
- Additional context or instructions
- Data extraction and enhancement
- Custom transformations
- Tool integration results
"""

from typing import Dict, Any


def post_process_prompt(raw_prompt: str, task_info: Dict[str, Any]):
    """
    Post-process the task prompt from the green agent.
    
    This is the extension point for workshop participants to:
    - Add additional context or instructions
    - Extract and enhance time series data
    - Apply custom transformations
    - Integrate tool results (e.g., statistical analysis, forecasts)
    
    Args:
        raw_prompt: The original prompt text from the green agent (contains task description, data, etc.)
        task_info: Extracted task information dictionary with keys:
            - task_type: Task type (T1, T2, T3, T4)
            - dataset: Dataset name (freshretailnet, causal_chambers, MIMIC, PSML)
            - has_context: Whether task includes context
            - has_mcq: Whether task includes MCQ questions
        
    Returns:
        Processed prompt string to send to LLM
        
    Example:
        # Default: passthrough (no modification)
        return raw_prompt
        
        # Enhanced: Add custom instructions based on task type
        if task_info["task_type"] == "T2":
            return f"{raw_prompt}\n\nRemember to forecast {num_steps} future values."
        return raw_prompt
    """
    # Default implementation: passthrough (no modification)
    # Workshop participants should modify this function to customize prompt processing
    return raw_prompt

