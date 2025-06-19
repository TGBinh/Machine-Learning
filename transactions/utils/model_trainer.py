"""
Model Training Module
Pre-train và lưu trữ models từ dữ liệu training
"""
import pandas as pd
import numpy as np
import pickle
import joblib
import os
import hashlib
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import Dict, Any, Optional, Tuple
import logging

from .fraud_detection import FraudDetectionKMeans
from .personal_finance import PersonalFinanceAnalyzer
from .json_utils import safe_json_serialize

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Class để train và quản lý các models
    """
    
    def __init__(self, models_dir: str = "trained_models"):
        self.models_dir = models_dir
        self.fraud_data_file = "phgl.xlsx"
        self.personal_data_file = "canhan.xlsx"
        
        # Tạo thư mục models nếu chưa có
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Paths cho saved models
        self.fraud_model_path = os.path.join(self.models_dir, "fraud_model.pkl")
        self.personal_model_path = os.path.join(self.models_dir, "personal_model.pkl")
        self.model_metadata_path = os.path.join(self.models_dir, "model_metadata.json")
    
    def get_file_hash(self, file_path: str) -> str:
        """Tính hash của file để detect thay đổi"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except FileNotFoundError:
            return ""
    
    def need_retrain(self) -> bool:
        """Kiểm tra xem có cần retrain models không"""
        # Kiểm tra file metadata có tồn tại không
        if not os.path.exists(self.model_metadata_path):
            return True
        
        try:
            # Load metadata
            import json
            with open(self.model_metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Kiểm tra hash của các file data
            fraud_hash = self.get_file_hash(self.fraud_data_file)
            personal_hash = self.get_file_hash(self.personal_data_file)
            
            if (metadata.get('fraud_data_hash') != fraud_hash or 
                metadata.get('personal_data_hash') != personal_hash):
                return True
            
            # Kiểm tra models có tồn tại không
            if (not os.path.exists(self.fraud_model_path) or 
                not os.path.exists(self.personal_model_path)):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking retrain need: {str(e)}")
            return True
    
    def train_fraud_detection_model(self) -> Dict[str, Any]:
        """Train fraud detection model"""
        try:
            logger.info("Training fraud detection model...")
            
            # Load data
            df = pd.read_excel(self.fraud_data_file, engine='openpyxl')
            logger.info(f"Loaded fraud training data: {df.shape}")
            
            # Initialize fraud detector
            fraud_detector = FraudDetectionKMeans()
            
            # Preprocess data
            df_processed = fraud_detector.preprocess_data(df)
            
            if not fraud_detector.numerical_columns:
                raise ValueError("No numerical columns found for training")
            
            # Prepare features
            X = df_processed[fraud_detector.numerical_columns].copy()
            X = X.fillna(X.median())
            
            # Scale features
            fraud_detector.scaler = StandardScaler()
            X_scaled = fraud_detector.scaler.fit_transform(X)
            
            # Find optimal k using elbow method
            k_values, wcss_values, optimal_k = fraud_detector.elbow_method(X_scaled)
            
            # Train K-Means with optimal k
            fraud_detector.kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            fraud_detector.kmeans.fit(X_scaled)
            
            # Calculate distances for threshold
            cluster_labels = fraud_detector.kmeans.predict(X_scaled)
            distances = np.linalg.norm(
                X_scaled - fraud_detector.kmeans.cluster_centers_[cluster_labels], 
                axis=1
            )
            
            # Set threshold at 95th percentile
            threshold = np.percentile(distances, 95)
            
            # Prepare model data
            model_data = {
                'scaler': fraud_detector.scaler,
                'kmeans': fraud_detector.kmeans,
                'numerical_columns': fraud_detector.numerical_columns,
                'optimal_k': optimal_k,
                'threshold': threshold,
                'elbow_data': {
                    'k_values': k_values,
                    'wcss_values': wcss_values,
                    'optimal_k': optimal_k
                },
                'training_stats': {
                    'total_samples': len(df_processed),
                    'features_count': len(fraud_detector.numerical_columns),
                    'threshold_percentile': 95
                }
            }
            
            # Save model
            with open(self.fraud_model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Fraud detection model saved with k={optimal_k}")
            return model_data['training_stats']
            
        except Exception as e:
            logger.error(f"Error training fraud detection model: {str(e)}")
            raise
    
    def train_personal_finance_model(self) -> Dict[str, Any]:
        """Train personal finance model"""
        try:
            logger.info("Training personal finance model...")
            
            # Load data
            df = pd.read_excel(self.personal_data_file, engine='openpyxl')
            logger.info(f"Loaded personal finance training data: {df.shape}")
            
            # Initialize analyzer
            analyzer = PersonalFinanceAnalyzer()
            
            # Preprocess data
            df_processed = analyzer.preprocess_data(df)
            
            if df_processed.empty:
                raise ValueError("No valid spending data found for training")
            
            # Analyze categories from training data
            category_stats = df_processed.groupby('category')['amount'].agg([
                'count', 'sum', 'mean', 'std'
            ]).fillna(0)
            
            # Calculate category insights
            total_spending = df_processed['amount'].sum()
            category_insights = {}
            
            for category in category_stats.index:
                stats = category_stats.loc[category]
                category_insights[category] = {
                    'avg_amount': float(stats['mean']),
                    'total_amount': float(stats['sum']),
                    'transaction_count': int(stats['count']),
                    'percentage': float(stats['sum'] / total_spending * 100),
                    'variability': float(stats['std'] / stats['mean']) if stats['mean'] > 0 else 0
                }
            
            # Spending patterns
            if 'hour' in df_processed.columns:
                hourly_patterns = df_processed.groupby('hour')['amount'].agg([
                    'count', 'sum', 'mean'
                ]).to_dict('index')
            else:
                hourly_patterns = {}
            
            # Weekend vs weekday patterns
            if 'is_weekend' in df_processed.columns:
                weekend_stats = df_processed.groupby('is_weekend')['amount'].agg([
                    'count', 'sum', 'mean'
                ]).to_dict('index')
            else:
                weekend_stats = {}
            
            # Overall statistics
            overall_stats = {
                'total_transactions': len(df_processed),
                'total_spending': float(total_spending),
                'avg_transaction': float(df_processed['amount'].mean()),
                'median_transaction': float(df_processed['amount'].median()),
                'spending_std': float(df_processed['amount'].std())
            }
            
            # Prepare model data
            model_data = {
                'category_keywords': analyzer.category_keywords,
                'category_insights': category_insights,
                'hourly_patterns': hourly_patterns,
                'weekend_stats': weekend_stats,
                'overall_stats': overall_stats,
                'training_period': {
                    'start_date': str(df_processed['datetime'].min().date()) if 'datetime' in df_processed.columns else None,
                    'end_date': str(df_processed['datetime'].max().date()) if 'datetime' in df_processed.columns else None,
                    'days_covered': int((df_processed['datetime'].max() - df_processed['datetime'].min()).days) if 'datetime' in df_processed.columns else 0
                }
            }
            
            # Save model
            with open(self.personal_model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Personal finance model saved with {len(category_insights)} categories")
            return overall_stats
            
        except Exception as e:
            logger.error(f"Error training personal finance model: {str(e)}")
            raise
    
    def train_all_models(self) -> Dict[str, Any]:
        """Train tất cả models và lưu metadata"""
        try:
            results = {}
            
            # Train models
            fraud_stats = self.train_fraud_detection_model()
            personal_stats = self.train_personal_finance_model()
            
            # Create metadata
            metadata = {
                'training_timestamp': datetime.now().isoformat(),
                'fraud_data_hash': self.get_file_hash(self.fraud_data_file),
                'personal_data_hash': self.get_file_hash(self.personal_data_file),
                'fraud_model_stats': fraud_stats,
                'personal_model_stats': personal_stats,
                'model_version': '1.0'
            }
            
            # Save metadata
            import json
            with open(self.model_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(safe_json_serialize(metadata), f, indent=2, ensure_ascii=False)
            
            results = {
                'success': True,
                'fraud_model': fraud_stats,
                'personal_model': personal_stats,
                'timestamp': metadata['training_timestamp']
            }
            
            logger.info("All models trained successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise
    
    def load_fraud_model(self) -> Optional[Dict[str, Any]]:
        """Load fraud detection model"""
        try:
            if os.path.exists(self.fraud_model_path):
                with open(self.fraud_model_path, 'rb') as f:
                    return pickle.load(f)
            return None
        except Exception as e:
            logger.error(f"Error loading fraud model: {str(e)}")
            return None
    
    def load_personal_model(self) -> Optional[Dict[str, Any]]:
        """Load personal finance model"""
        try:
            if os.path.exists(self.personal_model_path):
                with open(self.personal_model_path, 'rb') as f:
                    return pickle.load(f)
            return None
        except Exception as e:
            logger.error(f"Error loading personal model: {str(e)}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Lấy thông tin về models hiện tại"""
        try:
            if os.path.exists(self.model_metadata_path):
                import json
                with open(self.model_metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {}