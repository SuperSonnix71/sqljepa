from dataset import TabularDataset
from enhanced_preprocessing import EnhancedPreprocessor
from logger import logger

class EnhancedTabularDataset(TabularDataset):
    def __init__(self, df, target_col=None):
        try:
            self.preprocessor = EnhancedPreprocessor()
            processed_df, validation_results = self.preprocessor.preprocess(df, target_col)
            
            for col, results in validation_results.items():
                logger.info(f"Validation results for {col}: {results}")
            
            super().__init__(processed_df)
        except Exception as e:
            logger.error(f"Enhanced preprocessing failed: {str(e)}. Falling back to basic preprocessing.")
            super().__init__(df) 