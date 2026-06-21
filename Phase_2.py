import json
import logging
from typing import List, Dict, TypedDict

import openai
import instructor
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class DeepInsight(BaseModel):
    category: str = Field(
        ..., 
        description="Must be one of: 'Behavioral Pattern', 'Cognitive Load', 'Physiological Risk', or 'Protective Factor'."
    )
    observation: str = Field(
        ..., 
        description="What the data reveals based on the mathematical drivers (e.g., 'Sleep debt is compounding anxiety')."
    )
    clinical_rationale: str = Field(
        ..., 
        description="The psychological or physiological reason why this matters to the user's stress."
    )
    action_step: str = Field(
        ..., 
        description="One highly specific, achievable micro-action the user can take today."
    )

class AdvancedDashboardInsights(BaseModel):
    daily_summary: str = Field(
        ..., 
        description="A warm, compassionate 2-sentence summary of the user's day based on the data."
    )
    core_bottleneck: str = Field(
        ..., 
        description="The primary driver of stress today (e.g., 'Sleep deprivation combined with Exam Anxiety')."
    )
    deep_analysis: List[DeepInsight] = Field(
        ..., 
        description="Exactly 4 distinct, deep insights analyzing the intersection of their habits and stressors.",
        min_items=4,
        max_items=5
    )

class PipelineState(TypedDict):
    """The data structure that flows through our pipeline."""
    raw_metrics: dict              
    analysis_plan: str             
    safety_trigger: bool           
    final_insights_json: dict      

# Connect to local Ollama instance (e.g., running phi4 or llama3)
client = instructor.from_openai(
    openai.OpenAI(
        base_url="http://localhost:11434/v1", 
        api_key="ollama" 
    ),
    mode=instructor.Mode.JSON,
)

def planner_node(state: PipelineState) -> Dict:
    """Evaluates raw metrics to determine the analytical strategy."""
    logging.info("Executing Planner Node...")
    metrics = state.get("raw_metrics", {})
    score = metrics.get("predicted_stress_score", 0.0)
    
    if score > 0.75:
        plan = (
            "CRITICAL: High stress detected. Focus analysis on immediate risk mitigation, "
            "identifying the core bottleneck, and provide calming, low-effort action steps."
        )
        safety = True
    else:
        plan = (
            "STANDARD: Normal stress variance. Focus on baseline habit tracking, "
            "positive reinforcement for protective factors, and optimizing daily routines."
        )
        safety = False
        
    return {"analysis_plan": plan, "safety_trigger": safety}

def analyzer_node(state: PipelineState) -> Dict:
    """Uses the LLM to translate metrics into structured clinical insights."""
    logging.info("Executing Analyzer Node...")
    metrics = state["raw_metrics"]
    plan = state["analysis_plan"]
    
    prompt = f"""
    You are the clinical statistics engine of Serene, a mental health platform.
    
    Execution Strategy: {plan}
    
    User Metrics & Mathematical Drivers:
    {json.dumps(metrics, indent=2)}
    
    Rules for Analysis:
    1. Positive driver values mean the feature INCREASES stress.
    2. Negative driver values mean the feature DECREASES stress.
    3. Generate exactly 4 deep insights synthesizing these numbers.
    """
    
    insights_response: AdvancedDashboardInsights = client.chat.completions.create(
        model="phi4", 
        response_model=AdvancedDashboardInsights,
        messages=[
            {"role": "system", "content": "You are a precise, clinical JSON generator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2 
    )
    
    return {"final_insights_json": insights_response.model_dump()}

# Building Lang graph
def build_insights_pipeline():
    """Constructs the LangGraph state machine."""
    workflow = StateGraph(PipelineState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("analyzer", analyzer_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "analyzer")
    workflow.add_edge("analyzer", END)

    return workflow.compile()

# Returns final insights in JSON formats
def generate_insights(ml_payload: dict) -> dict:
    """
    Main entry point for external files (like FastAPI or Component 3).
    Takes raw ML data, runs the pipeline, and returns the parsed JSON dict.
    """
    pipeline = build_insights_pipeline()
    # logging.info("Starting pipeline execution via generate_insights()...")
    
    final_state = pipeline.invoke({"raw_metrics": ml_payload["raw_metrics"]})
    return final_state.get("final_insights_json", {})

# Testing code

# if __name__ == "__main__":
#     print("\n--- Serene Component 2: Direct Execution Test ---")
    
#     # Mock payload simulating Component 1's output
#     mock_ml_payload = {
#         "raw_metrics": {
#             "predicted_stress_score": 0.82,
#             "top_mathematical_drivers": {
#                 "work_study_hours": 0.18,      
#                 "sleep_avg_hours": 0.14,       
#                 "Exam Anxiety": 0.08,          
#                 "wellness_points": -0.05       
#             }
#         }
#     }

#     # Execute the active function
#     returned_insights = generate_insights(mock_ml_payload)
    
#     print("\n--- Final Validated Output Returned ---")
#     print(json.dumps(returned_insights, indent=2))