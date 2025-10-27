"""
LLM Helper Module for AURA Preprocessor 2.0

This module provides explanation generation and logging capabilities.
Simulates LLM responses for now, with plans to integrate with actual LLM APIs later.
"""

import json
import os
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Union
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLMHelper:
    """
    Handles explanation generation and logging for AURA pipeline steps.
    
    This class simulates LLM responses and saves explanations to JSON files
    for later integration with actual LLM APIs.
    """
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize the LLM helper.
        
        Args:
            output_dir: Directory to save explanation files
        """
        self.output_dir = output_dir
        self.explanations = []
        self.ensure_output_dir()
    
    def ensure_output_dir(self) -> None:
        """Ensure the output directory exists."""
        os.makedirs(self.output_dir, exist_ok=True)
    
    def explain_step(self, 
                    step_name: str, 
                    data_sample: Optional[pd.DataFrame] = None,
                    additional_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate and log an explanation for a pipeline step.
        
        Args:
            step_name: Name/description of the step performed
            data_sample: Sample of data after the step (optional)
            additional_info: Additional context information (optional)
            
        Returns:
            Generated explanation string
        """
        try:
            # Generate explanation based on step type
            explanation = self._generate_explanation(step_name, data_sample, additional_info)
            
            # Create explanation record
            explanation_record = {
                "timestamp": datetime.now().isoformat(),
                "step_name": step_name,
                "explanation": explanation,
                "data_shape": data_sample.shape if data_sample is not None else None,
                "additional_info": additional_info or {}
            }
            
            # Add to explanations list
            self.explanations.append(explanation_record)
            
            # Log the explanation
            logger.info(f"Generated explanation for: {step_name}")
            print(f"\n🤖 AI Explanation: {explanation}")
            
            return explanation
            
        except Exception as e:
            error_msg = f"Error generating explanation for {step_name}: {str(e)}"
            logger.error(error_msg)
            print(f"⚠️ {error_msg}")
            return f"Explanation generation failed for {step_name}"
    
    def _generate_explanation(self, 
                            step_name: str, 
                            data_sample: Optional[pd.DataFrame],
                            additional_info: Optional[Dict[str, Any]]) -> str:
        """
        Generate explanation based on step type and data.
        
        This simulates LLM response generation.
        """
        explanations = {
            "Missing values handled": self._explain_missing_values(data_sample),
            "Label encoded": self._explain_label_encoding(data_sample),
            "One-hot encoded": self._explain_onehot_encoding(data_sample),
            "StandardScaler applied": self._explain_scaling(data_sample, "StandardScaler"),
            "MinMaxScaler applied": self._explain_scaling(data_sample, "MinMaxScaler"),
            "No scaling applied": self._explain_no_scaling(data_sample),
            "Trained RandomForest model": self._explain_model_training(additional_info),
        }
        
        # Check for partial matches
        for key, explanation in explanations.items():
            if key.lower() in step_name.lower():
                return explanation
        
        # Default explanation
        return self._explain_generic_step(step_name, data_sample)
    
    def _explain_missing_values(self, data_sample: Optional[pd.DataFrame]) -> str:
        """Explain missing value handling."""
        if data_sample is None:
            return "Missing values have been processed. This step ensures data quality by handling incomplete records."
        
        return f"""Missing values have been successfully handled! Here's what happened:

• **Data Quality**: Missing values can cause errors in machine learning models
• **Strategy Applied**: We filled missing values using appropriate methods:
  - Numeric columns: Filled with mean/median values
  - Categorical columns: Filled with most frequent value (mode)
• **Result**: Your dataset now has complete data for all {data_sample.shape[1]} features
• **Next Steps**: The data is now ready for feature encoding and scaling"""
    
    def _explain_label_encoding(self, data_sample: Optional[pd.DataFrame]) -> str:
        """Explain label encoding."""
        return f"""Label encoding has been applied! Here's what this means:

• **Purpose**: Converts text categories into numbers (e.g., 'Male'→0, 'Female'→1)
• **Benefits**: 
  - Reduces memory usage
  - Works well with tree-based models
  - Maintains ordinal relationships
• **Use Case**: Best for categorical variables with inherent order
• **Note**: This creates a single column per original categorical feature"""
    
    def _explain_onehot_encoding(self, data_sample: Optional[pd.DataFrame]) -> str:
        """Explain one-hot encoding."""
        return f"""One-hot encoding has been applied! Here's the breakdown:

• **Purpose**: Creates binary columns for each category (e.g., 'Male'→[1,0], 'Female'→[0,1])
• **Benefits**:
  - No ordinal assumptions
  - Works well with linear models
  - Prevents model bias from category ordering
• **Trade-off**: Increases dataset width (more columns)
• **Best For**: Nominal categorical variables without inherent order"""
    
    def _explain_scaling(self, data_sample: Optional[pd.DataFrame], scaler_type: str) -> str:
        """Explain feature scaling."""
        return f"""{scaler_type} has been applied to your features! Here's why this matters:

• **Problem Solved**: Different features have different scales (e.g., age: 0-100, income: 0-100000)
• **{scaler_type} Effect**:
  - Transforms all features to have similar ranges
  - Prevents large-scale features from dominating the model
  - Improves convergence speed for many algorithms
• **Impact**: Your model will now treat all features equally
• **Result**: Better model performance and more stable training"""
    
    def _explain_no_scaling(self, data_sample: Optional[pd.DataFrame]) -> str:
        """Explain why scaling was skipped."""
        return """Scaling was skipped for this dataset. Here's why this might be appropriate:

• **Tree-based Models**: Random Forest and similar algorithms are scale-invariant
• **Already Normalized**: Features might already be in similar ranges
• **Domain Knowledge**: Sometimes raw scales are meaningful
• **Note**: If you switch to linear models later, consider applying scaling"""
    
    def _explain_model_training(self, additional_info: Optional[Dict[str, Any]]) -> str:
        """Explain model training results."""
        accuracy = additional_info.get('accuracy', 'N/A') if additional_info else 'N/A'
        return f"""Model training completed successfully! Here's your performance summary:

• **Algorithm**: Random Forest Classifier
• **Accuracy**: {accuracy} (on test set)
• **What This Means**:
  - Random Forest combines multiple decision trees for robust predictions
  - The accuracy shows how well your model performs on unseen data
  - Values above 0.7 are generally considered good for classification
• **Next Steps**: You can now use this model to make predictions on new data!"""
    
    def _explain_generic_step(self, step_name: str, data_sample: Optional[pd.DataFrame]) -> str:
        """Generic explanation for unknown steps."""
        return f"""Step '{step_name}' has been completed successfully!

• **Purpose**: This step improves your dataset for machine learning
• **Impact**: Your data is now better prepared for model training
• **Data Status**: {f'Dataset shape: {data_sample.shape}' if data_sample is not None else 'Data processed'}
• **Next**: Continue with the next step in your pipeline"""
    
    def save_explanations(self, filename: str = "aura_explanations.json") -> None:
        """
        Save all explanations to a JSON file.
        
        Args:
            filename: Name of the file to save explanations
        """
        try:
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.explanations, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(self.explanations)} explanations to {filepath}")
            print(f"💾 Saved explanations to {filepath}")
            
        except Exception as e:
            error_msg = f"Error saving explanations: {str(e)}"
            logger.error(error_msg)
            print(f"⚠️ {error_msg}")
    
    def get_explanations_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all explanations.
        
        Returns:
            Dictionary containing explanation summary
        """
        return {
            "total_explanations": len(self.explanations),
            "steps_covered": [exp["step_name"] for exp in self.explanations],
            "timestamp_range": {
                "start": self.explanations[0]["timestamp"] if self.explanations else None,
                "end": self.explanations[-1]["timestamp"] if self.explanations else None
            }
        }


# Global instance for easy access
_llm_helper = None

def get_llm_helper(output_dir: str = "outputs") -> LLMHelper:
    """
    Get or create the global LLM helper instance.
    
    Args:
        output_dir: Directory for output files
        
    Returns:
        LLMHelper instance
    """
    global _llm_helper
    if _llm_helper is None:
        _llm_helper = LLMHelper(output_dir)
    return _llm_helper

def explain_step(step_name: str, 
                data_sample: Optional[pd.DataFrame] = None,
                additional_info: Optional[Dict[str, Any]] = None) -> str:
    """
    Convenience function to explain a step.
    
    Args:
        step_name: Name/description of the step
        data_sample: Sample data after the step
        additional_info: Additional context
        
    Returns:
        Generated explanation
    """
    helper = get_llm_helper()
    return helper.explain_step(step_name, data_sample, additional_info)

def save_explanations(filename: str = "aura_explanations.json") -> None:
    """
    Convenience function to save explanations.
    
    Args:
        filename: Name of the file to save
    """
    helper = get_llm_helper()
    helper.save_explanations(filename)

