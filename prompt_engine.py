"""
Prompt Engineering Module for Time Series Tasks

This module provides optimized prompt structures for different time series task types:
- T1: Signal description (accuracy-based)
- T2: Forecasting (regression metrics)
- T3: Description with context (accuracy-based)
- T4: Forecasting with context (regression + MCQ)

The prompts are designed to elicit optimal performance from LLMs without tools.
Workshop participants can modify these prompts to experiment with different strategies.
"""

import json
import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger("prompt_engine")


def extract_task_info(prompt_text: str):
    """
    Extract task information from the prompt text.
    
    This helps identify the task type and structure the response accordingly.
    
    Args:
        prompt_text: The full prompt from the green agent
        
    Returns:
        Dict with task_type, dataset, and other extracted info
    """
    info = {
        "task_type": "unknown",
        "dataset": "unknown",
        "has_context": False,
        "has_mcq": False,
    }
    
    # Try to identify task type from prompt structure
    prompt_lower = prompt_text.lower()
    
    if "task type: t1" in prompt_lower or "task_type: t1" in prompt_lower:
        info["task_type"] = "T1"
    elif "task type: t2" in prompt_lower or "task_type: t2" in prompt_lower:
        info["task_type"] = "T2"
    elif "task type: t3" in prompt_lower or "task_type: t3" in prompt_lower:
        info["task_type"] = "T3"
        info["has_context"] = True
    elif "task type: t4" in prompt_lower or "task_type: t4" in prompt_lower:
        info["task_type"] = "T4"
        info["has_context"] = True
        info["has_mcq"] = True
    
    # Detect dataset from prompt
    for dataset in ["freshretailnet", "causal_chambers", "mimic", "psml"]:
        if dataset in prompt_lower:
            info["dataset"] = dataset
            break
    
    # Check for forecasting keywords
    if "forecast" in prompt_lower or "predict" in prompt_lower:
        if info["task_type"] == "unknown":
            info["task_type"] = "T2"  # Default to T2 if unclear
    
    # Check for MCQ keywords
    if "multiple choice" in prompt_lower or "mcq" in prompt_lower:
        info["has_mcq"] = True
    
    return info


def build_system_prompt(task_type: str, task_info: Dict[str, Any]):
    """
    Build an optimized system prompt for the given task type.
    
    These prompts are designed to:
    1. Guide the model to produce the correct output format
    2. Emphasize timeseries-specific reasoning
    3. Encourage careful analysis of patterns and trends
    
    Args:
        task_type: Task type (T1, T2, T3, T4)
        task_info: Extracted task information
        
    Returns:
        System prompt string
    """
    base_prompt = """You are an expert time series analyst. Your task is to analyze time series data and provide accurate responses.

CRITICAL: You must respond with valid JSON only. Do not include any explanatory text outside the JSON structure.

"""
    
    if task_type in ["T1", "T3"]:
        # Accuracy-based tasks (description/classification)
        return base_prompt + """For this task, you need to answer questions about the time series data.

Your response must be a JSON object with an "answers" field containing a dictionary:
{
  "answers": {
    "question_id_1": "answer_1",
    "question_id_2": "answer_2",
    ...
  }
}

Guidelines:
- Analyze the time series patterns carefully (trends, seasonality, cycles, anomalies)
- Look for relationships between different signals if multiple series are provided
- For classification questions, consider the statistical properties (mean, variance, distribution)
- Be precise and consistent with your answers
- Use the exact question IDs provided in the prompt

Example response format:
{
  "answers": {
    "q1": "increasing",
    "q2": "stationary",
    "q3": "yes"
  }
}
"""
    
    elif task_type in ["T2", "T4"]:
        # Forecasting tasks
        forecast_prompt = base_prompt + """For this task, you need to forecast future values of the time series.

Your response must be a JSON object with a "forecasts" field containing an array of numerical values:
{
  "forecasts": [value1, value2, value3, ...]
}

Guidelines:
- Carefully analyze the historical data patterns (trend, seasonality, cycles)
- Consider the scale and units of the input data
- Forecast values should be in the same scale/units as the input
- The number of forecast values must match the number of future time steps requested
- For time series with trends, extrapolate the trend appropriately
- For seasonal patterns, continue the seasonal cycle
- For stationary series, forecast values near the mean with appropriate variance
- Be mindful of potential outliers or regime changes in the data

"""
        
        if task_type == "T4":
            # T4 also includes MCQ
            forecast_prompt += """Additionally, this task includes multiple-choice questions (MCQ).

Include an "mcq" field in your response:
{
  "forecasts": [value1, value2, ...],
  "mcq": {
    "question_id_1": "option_a",
    "question_id_2": "option_b",
    ...
  }
}

For MCQ questions:
- Analyze the contextual information provided
- Consider how the context affects the time series behavior
- Select the most appropriate answer based on both the data and context
"""
        
        forecast_prompt += """
Example response format:
{
  "forecasts": [10.5, 11.2, 10.8, 11.5, 11.0]
}
"""
        
        return forecast_prompt
    
    else:
        # Unknown task type - generic prompt
        return base_prompt + """Analyze the time series data and provide your response in JSON format.

If the task asks for answers to questions, use:
{
  "answers": {
    "question_id": "answer"
  }
}

If the task asks for forecasts, use:
{
  "forecasts": [value1, value2, ...]
}

Always respond with valid JSON only.
"""


def format_response(response_text: str, task_type: str, task_info: Dict[str, Any]):
    """
    Format the LLM response to ensure it's valid JSON in the expected format.
    
    This function:
    1. Extracts JSON from the response (handles cases where LLM adds explanation)
    2. Validates the structure matches the expected format
    3. Ensures required fields are present
    
    Args:
        response_text: Raw response from LLM
        task_type: Task type (T1, T2, T3, T4)
        task_info: Extracted task information
        
    Returns:
        Formatted JSON string
    """
    # Try to extract JSON from the response
    # Handle cases where LLM wraps JSON in markdown or adds explanation
    
    # First, try to parse as-is
    try:
        parsed = json.loads(response_text.strip())
        return json.dumps(parsed)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            return json.dumps(parsed)
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON object in the text
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            return json.dumps(parsed)
        except json.JSONDecodeError:
            pass
    
    # If all parsing fails, create a default structure based on task type
    logger.warning(f"Failed to parse JSON from response, creating default structure for {task_type}")
    
    if task_type in ["T1", "T3"]:
        return json.dumps({"answers": {}})
    else:  # T2, T4
        result = {"forecasts": []}
        if task_type == "T4":
            result["mcq"] = {}
        return json.dumps(result)

