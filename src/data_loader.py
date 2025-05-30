import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialDataLoader:
    """Class to handle loading and basic preprocessing of financial news data."""
    
    def __init__(self, data_path: str):
        """
        Initialize the data loader.
        
        Args:
            data_path (str): Path to the CSV file
        """
        self.data_path = data_path
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the financial news dataset.
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            logger.info(f"Loading data from {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get basic information about the dataset.
        
        Returns:
            Dict: Basic dataset information
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum()
        }
        
        return info
    
    def get_sample_data(self, n: int = 5) -> pd.DataFrame:
        """
        Get a sample of the data.
        
        Args:
            n (int): Number of samples to return
            
        Returns:
            pd.DataFrame: Sample data
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        return self.data.head(n) 