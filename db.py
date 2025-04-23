from sqlalchemy import create_engine
import pandas as pd
from config import DATABASE_URI, SCHEMA_NAME, TABLE_NAME
from logger import logger

def fetch_data():
    engine = create_engine(DATABASE_URI)
    query = f'SELECT * FROM "{SCHEMA_NAME}"."{TABLE_NAME}"'
    df = pd.read_sql(query, engine)
    logger.info(f"Data loaded: {df.shape}")
    return df
 