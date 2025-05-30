import pandas as pd
import numpy as np
from typing import Any, Dict, List
import json

def save_analysis_results(results: Dict[str, Any], filename: str):
    """
    Save analysis results to a JSON file.
    
    Args:
        results: Dictionary containing analysis results
        filename: Output filename
    """
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    converted_results = convert_numpy_types(results)
    
    with open(filename, 'w') as f:
        json.dump(converted_results, f, indent=2, default=str)

def format_large_numbers(num: int) -> str:
    """
    Format large numbers with appropriate suffixes.
    
    Args:
        num: Number to format
        
    Returns:
        str: Formatted number string
    """
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return str(num)

def get_data_quality_report(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive data quality report.
    
    Args:
        data: DataFrame to analyze
        
    Returns:
        Dict: Data quality metrics
    """
    report = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'missing_values': {},
        'duplicate_rows': len(data) - len(data.drop_duplicates()),
        'data_types': data.dtypes.to_dict(),
        'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024 * 1024)
    }
    
    # Missing values analysis
    for col in data.columns:
        missing_count = data[col].isnull().sum()
        missing_pct = (missing_count / len(data)) * 100
        report['missing_values'][col] = {
            'count': int(missing_count),
            'percentage': round(missing_pct, 2)
        }
    
    return report 