# Data source configuration
DATA_PATH = "data/your_data.csv"  # Path to your data file
DATA_TYPE = "csv"  # Type of data source: 'csv', 'parquet', or 'sql'

# Database configuration (only needed if DATA_TYPE is 'sql')
DATABASE_URI = "postgresql://username:password@host:port/dbname"
SCHEMA_NAME = "public"
TABLE_NAME = "your_table"

# Training configuration
BATCH_SIZE = 256
EPOCHS = 50
LR = 1e-4

# Model configuration
EMBED_DIM = 64
NUM_HEADS = 4
ENCODER_LAYERS = 4
MOMENTUM = 0.996  # EMA momentum for target encoder

# Masking configuration
MASK_MIN_CONTEXT = 0.07  # Minimum ratio of features to keep in context
MASK_MAX_CONTEXT = 0.15  # Maximum ratio of features to keep in context

# Device configuration
DEVICE = "cuda"  # or "cpu"

# Logging configuration
MLFLOW_URI = "mlruns" 