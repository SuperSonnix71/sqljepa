import pandas as pd
from pathlib import Path
from logger import logger

def load_data(data_path, file_type=None):
    """
    Load data from various file formats into a pandas DataFrame.
    
    Args:
        data_path (str): Path to the data file or directory
        file_type (str, optional): Type of file ('csv', 'parquet', 'sql'). 
                                 If None, will be inferred from file extension.
    
    Returns:
        pd.DataFrame: Loaded data
    """
    data_path = Path(data_path)
    
    # Infer file type from extension if not provided
    if file_type is None:
        file_type = data_path.suffix.lower()[1:]  # Remove the dot
    
    try:
        if file_type == 'csv':
            logger.info(f"Loading CSV file from {data_path}")
            return pd.read_csv(data_path)
            
        elif file_type == 'parquet':
            logger.info(f"Loading Parquet file from {data_path}")
            return pd.read_parquet(data_path)
            
        elif file_type == 'sql':
            from db import fetch_data
            logger.info(f"Loading data from SQL database")
            return fetch_data()
            
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def load_multiple_files(data_paths, file_types=None):
    """
    Load and concatenate multiple data files.
    
    Args:
        data_paths (list): List of paths to data files
        file_types (list, optional): List of file types corresponding to data_paths.
                                   If None, will be inferred from file extensions.
    
    Returns:
        pd.DataFrame: Concatenated data
    """
    if file_types is None:
        file_types = [None] * len(data_paths)
    
    dfs = []
    for path, file_type in zip(data_paths, file_types):
        df = load_data(path, file_type)
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True) 