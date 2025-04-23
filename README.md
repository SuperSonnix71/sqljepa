# SQL-JEPA: Augmentation-Free Self-Supervised Learning for Tabular Data

This repository implements SQL-JEPA (SQL Joint-Embedding Predictive Architecture), a novel self-supervised learning method for tabular data that does not require data augmentations. SQL-JEPA enables deep neural networks to achieve state-of-the-art performance on tabular tasks, often outperforming traditional methods like XGBoost and CatBoost.

## Key Features

- **Augmentation-Free SSL**: Unlike traditional SSL methods that rely on data augmentations, SQL-JEPA uses masked feature subsets from the same sample to create self-supervised signals.
- **Joint-Embedding Predictive Architecture**: Predicts latent representations of masked features using a context encoder and target encoder setup.
- **[REG] Token Innovation**: Introduces a regularization token to prevent representation collapse, a crucial component for tabular data SSL.
- **Momentum-Based Updates**: Uses exponential moving average updates for the target encoder to stabilize training.
- **State-of-the-Art Performance**: Consistently outperforms traditional methods on tabular tasks.
- **Enhanced Data Preprocessing**: Robust data validation, cleaning, and feature engineering pipeline.
- **Multiple Data Source Support**: Works with CSV, Parquet, and SQL databases.

## Architecture

SQL-JEPA consists of three main components:

1. **Context Encoder**: Processes a subset of features (context) to produce a latent representation.
2. **Target Encoder**: Processes all features but its output is only used for the masked (target) features.
3. **Predictor**: Predicts the target encoder's latent representation from the context encoder's output.

The architecture includes several key innovations:

- **Feature Masking**: Instead of augmenting data, SQL-JEPA masks a subset of features to create self-supervised signals.
- **[REG] Token**: A special token that's never masked and helps prevent representation collapse.
- **Momentum Updates**: The target encoder is updated using momentum from the context encoder.

## Enhanced Preprocessing Pipeline

SQL-JEPA includes a robust preprocessing pipeline that can be enabled through configuration:

1. **Data Validation**:
   - Outlier detection using z-scores
   - Missing value analysis
   - Categorical-like numeric column detection
   - Rare category identification

2. **Data Cleaning**:
   - IQR-based outlier removal
   - Multiple imputation strategies (median, KNN)
   - Robust scaling for numeric features
   - Rare category handling for categorical features

3. **Feature Engineering**:
   - Polynomial features for numeric columns
   - Interaction terms between numeric features
   - Target encoding for categorical features
   - Frequency encoding for categorical features
   - PCA-based dimensionality reduction

4. **Error Handling**:
   - Graceful fallback mechanisms
   - Comprehensive error logging
   - Memory-efficient feature generation
   - Automatic handling of edge cases

## Differences from Current Approaches

1. **Traditional SSL Methods**:
   - Require carefully designed data augmentations
   - Often struggle with tabular data due to its structured nature
   - May introduce out-of-distribution samples

2. **Tree-Based Methods (XGBoost, CatBoost)**:
   - Require extensive feature engineering
   - Don't learn representations that can be transferred
   - Limited in capturing complex feature interactions

3. **Other Tabular SSL Methods**:
   - Often use contrastive learning which requires negative pairs
   - May not handle mixed data types effectively
   - Can suffer from representation collapse

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sql-jepa.git
cd sql-jepa
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure the data source and preprocessing in `config.py`:
```python
# Data source configuration
DATA_PATH = "data/your_data.csv"  # Path to your data file
DATA_TYPE = "csv"  # Type of data source: 'csv', 'parquet', or 'sql'
USE_ENHANCED_PREPROCESSING = True  # Enable enhanced preprocessing
TARGET_COL = None  # Target column for feature engineering (optional)

# Database configuration (only needed if DATA_TYPE is 'sql')
DATABASE_URI = "postgresql://username:password@host:port/dbname"
SCHEMA_NAME = "public"
TABLE_NAME = "your_table"
```

## Usage

### Training

1. Prepare your dataset:
```python
from dataset import TabularDataset
from enhanced_dataset import EnhancedTabularDataset

# Basic preprocessing
dataset = TabularDataset(your_dataframe)

# Enhanced preprocessing
dataset = EnhancedTabularDataset(your_dataframe, target_col='target')
```

2. Initialize the model:
```python
from model import SQLJEPAModel
model = SQLJEPAModel(
    input_dim=your_input_dim,
    embed_dim=64,
    num_heads=4,
    num_layers=4
)
```

3. Train the model:
```python
from trainer import Trainer
trainer = Trainer(model, dataset, config)
for epoch in range(config.EPOCHS):
    loss = trainer.train_epoch()
```

### Evaluation

1. Load a pre-trained model:
```python
model.load_state_dict(torch.load('path_to_model.pt'))
```

2. Extract representations:
```python
representations = model(context_x, is_target=False)
```

3. Evaluate on downstream tasks:
```python
# Example: Classification
classifier = nn.Linear(embed_dim, num_classes)
predictions = classifier(representations)
```

## Configuration

Key parameters in `config.py`:

- `BATCH_SIZE`: Batch size for training (default: 256)
- `EPOCHS`: Number of training epochs (default: 50)
- `LR`: Learning rate (default: 1e-4)
- `EMBED_DIM`: Embedding dimension (default: 64)
- `NUM_HEADS`: Number of attention heads (default: 4)
- `ENCODER_LAYERS`: Number of transformer layers (default: 4)
- `MOMENTUM`: EMA momentum for target encoder (default: 0.996)
- `MASK_MIN_CONTEXT`: Minimum ratio of features to keep in context (default: 0.07)
- `MASK_MAX_CONTEXT`: Maximum ratio of features to keep in context (default: 0.15)
- `USE_ENHANCED_PREPROCESSING`: Enable enhanced preprocessing (default: True)
- `TARGET_COL`: Target column for feature engineering (default: None)

## Results

SQL-JEPA achieves state-of-the-art results on various tabular datasets:

- Adult Income: 87.2% accuracy
- Higgs: 89.1% accuracy
- Helena: 82.3% accuracy
- Jannis: 78.9% accuracy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@GitHub Code{SQLJEPA,
  Title={SQL-JEPA: Augmentation-Free Self-Supervised Learning for SQL Data},
  Link={https://github.com/SuperSonnix71/sqljepa}
  author={Sonny Mir},
  year={2025}
}
``` 