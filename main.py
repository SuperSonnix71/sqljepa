import mlflow
from data_loader import load_data
from dataset import TabularDataset
from enhanced_dataset import EnhancedTabularDataset
from model import SQLJEPAModel
from trainer import Trainer
from evaluate import evaluate
import config as cfg
from logger import logger

mlflow.set_tracking_uri(cfg.MLFLOW_URI)
mlflow.start_run()

# Load data from the configured source
df = load_data(cfg.DATA_PATH, cfg.DATA_TYPE)

if cfg.USE_ENHANCED_PREPROCESSING:
    dataset = EnhancedTabularDataset(df, cfg.TARGET_COL)
else:
    dataset = TabularDataset(df)

input_dim = dataset.data.shape[1]

model = SQLJEPAModel(input_dim, cfg.EMBED_DIM, cfg.NUM_HEADS, cfg.ENCODER_LAYERS)
trainer = Trainer(model, dataset, cfg)

for epoch in range(cfg.EPOCHS):
    loss = trainer.train_epoch()
    mlflow.log_metric("train_loss", loss, step=epoch)

metrics = evaluate(model, dataset, cfg.DEVICE)
mlflow.log_metrics(metrics)
mlflow.end_run()

logger.info(f"Evaluation metrics: {metrics}")
 