import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from logger import logger

class DataValidator:
    def __init__(self):
        self.validation_rules = {
            'numeric': {
                'min_value': float('-inf'),
                'max_value': float('inf'),
                'allow_null': False,
                'unique_threshold': 0.9
            },
            'categorical': {
                'max_categories': 100,
                'min_frequency': 0.001,
                'allow_null': False
            }
        }
    
    def validate_column(self, series, col_type):
        if series.empty:
            return {
                'empty': True,
                'missing_values': 0,
                'outliers': 0,
                'is_categorical_like': False,
                'rare_categories': 0,
                'unique_categories': 0
            }
            
        if col_type == 'numeric':
            if series.std() == 0:
                return {
                    'constant': True,
                    'missing_values': series.isnull().sum(),
                    'outliers': 0,
                    'is_categorical_like': False
                }
                
            z_scores = np.abs((series - series.mean()) / series.std())
            outliers = z_scores > 3
            unique_ratio = series.nunique() / len(series)
            is_categorical_like = unique_ratio < self.validation_rules['numeric']['unique_threshold']
            return {
                'outliers': outliers.sum(),
                'is_categorical_like': is_categorical_like,
                'missing_values': series.isnull().sum()
            }
        elif col_type == 'categorical':
            value_counts = series.value_counts(normalize=True)
            rare_categories = value_counts[value_counts < self.validation_rules['categorical']['min_frequency']]
            return {
                'rare_categories': len(rare_categories),
                'missing_values': series.isnull().sum(),
                'unique_categories': series.nunique()
            }

class DataCleaner:
    def __init__(self):
        self.cleaning_strategies = {
            'numeric': {
                'outlier_method': 'iqr',
                'imputation_method': 'median',
                'scaling_method': 'robust'
            },
            'categorical': {
                'rare_category_threshold': 0.01,
                'rare_category_strategy': 'merge',
                'imputation_method': 'mode'
            }
        }
        self.scaler = RobustScaler()
        self.knn_imputer = KNNImputer(n_neighbors=5)
    
    def clean_numeric_column(self, series, strategy):
        if series.empty:
            return series
            
        if series.std() == 0:
            return series
            
        if strategy['outlier_method'] == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:  # Avoid division by zero
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                series = series.clip(lower_bound, upper_bound)
        
        if strategy['imputation_method'] == 'median':
            series = series.fillna(series.median())
        elif strategy['imputation_method'] == 'knn':
            try:
                series = pd.Series(self.knn_imputer.fit_transform(series.values.reshape(-1, 1)).ravel())
            except Exception as e:
                logger.warning(f"KNN imputation failed: {str(e)}. Falling back to median imputation.")
                series = series.fillna(series.median())
        
        if strategy['scaling_method'] == 'robust':
            try:
                series = pd.Series(self.scaler.fit_transform(series.values.reshape(-1, 1)).ravel())
            except Exception as e:
                logger.warning(f"Robust scaling failed: {str(e)}. Skipping scaling.")
        
        return series
    
    def clean_categorical_column(self, series, strategy):
        if series.empty:
            return series
            
        value_counts = series.value_counts(normalize=True)
        rare_categories = value_counts[value_counts < strategy['rare_category_threshold']].index
        
        if strategy['rare_category_strategy'] == 'merge':
            series = series.replace(rare_categories, 'Other')
        elif strategy['rare_category_strategy'] == 'drop':
            series = series.replace(rare_categories, np.nan)
        
        if strategy['imputation_method'] == 'mode':
            if not series.empty and not series.isna().all():
                series = series.fillna(series.mode()[0])
            else:
                series = series.fillna('Missing')
        
        return series

class FeatureEngineer:
    def __init__(self):
        self.feature_generators = {
            'numeric': {
                'polynomial': True,
                'interaction': True,
                'rolling_stats': True
            },
            'categorical': {
                'target_encoding': True,
                'frequency_encoding': True,
                'one_hot_encoding': True
            }
        }
        self.pca = PCA(n_components=0.95)
        self.variance_threshold = VarianceThreshold(threshold=0.01)
    
    def generate_numeric_features(self, df, numeric_cols):
        new_features = {}
        
        if self.feature_generators['numeric']['polynomial']:
            for col in numeric_cols:
                if df[col].std() > 0:  # Avoid constant columns
                    new_features[f'{col}_squared'] = df[col] ** 2
                    new_features[f'{col}_cubed'] = df[col] ** 3
        
        if self.feature_generators['numeric']['interaction']:
            max_interactions = 100  # Limit to prevent memory issues
            interaction_count = 0
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    if interaction_count >= max_interactions:
                        break
                    if df[col1].std() > 0 and df[col2].std() > 0:  # Avoid constant columns
                        new_features[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]
                        interaction_count += 1
        
        return pd.DataFrame(new_features)
    
    def generate_categorical_features(self, df, categorical_cols, target=None):
        new_features = {}
        
        if self.feature_generators['categorical']['target_encoding'] and target is not None:
            if target in df.columns:
                for col in categorical_cols:
                    try:
                        new_features[f'{col}_target_encoded'] = df.groupby(col)[target].transform('mean')
                    except Exception as e:
                        logger.warning(f"Target encoding failed for {col}: {str(e)}")
        
        if self.feature_generators['categorical']['frequency_encoding']:
            for col in categorical_cols:
                try:
                    new_features[f'{col}_freq'] = df[col].map(df[col].value_counts(normalize=True))
                except Exception as e:
                    logger.warning(f"Frequency encoding failed for {col}: {str(e)}")
        
        return pd.DataFrame(new_features)

class EnhancedPreprocessor:
    def __init__(self):
        self.validator = DataValidator()
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()
    
    def preprocess(self, df, target_col=None):
        # Validate input
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty or None")
            
        if target_col is not None and target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        validation_results = {}
        for col in df.columns:
            col_type = 'numeric' if np.issubdtype(df[col].dtype, np.number) else 'categorical'
            validation_results[col] = self.validator.validate_column(df[col], col_type)
        
        cleaned_df = df.copy()
        for col in df.columns:
            col_type = 'numeric' if np.issubdtype(df[col].dtype, np.number) else 'categorical'
            strategy = self.cleaner.cleaning_strategies[col_type]
            if col_type == 'numeric':
                cleaned_df[col] = self.cleaner.clean_numeric_column(df[col], strategy)
            else:
                cleaned_df[col] = self.cleaner.clean_categorical_column(df[col], strategy)
        
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        categorical_cols = cleaned_df.select_dtypes(exclude=[np.number]).columns
        
        numeric_features = self.engineer.generate_numeric_features(cleaned_df, numeric_cols)
        categorical_features = self.engineer.generate_categorical_features(cleaned_df, categorical_cols, target_col)
        
        final_df = pd.concat([cleaned_df, numeric_features, categorical_features], axis=1)
        
        numeric_features_final = final_df.select_dtypes(include=[np.number])
        if not numeric_features_final.empty:
            try:
                numeric_features_final = pd.DataFrame(
                    self.variance_threshold.fit_transform(numeric_features_final),
                    columns=numeric_features_final.columns[self.variance_threshold.get_support()]
                )
                
                if numeric_features_final.shape[1] > 1:
                    pca_features = self.pca.fit_transform(numeric_features_final)
                    pca_df = pd.DataFrame(
                        pca_features,
                        columns=[f'pca_{i}' for i in range(pca_features.shape[1])]
                    )
                    final_df = pd.concat([final_df, pca_df], axis=1)
            except Exception as e:
                logger.warning(f"Dimensionality reduction failed: {str(e)}")
        
        return final_df, validation_results 