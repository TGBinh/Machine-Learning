"""
Fraud Detection Module using K-Means Clustering
Tích hợp logic từ kmean.py để phát hiện gian lận
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import io
import base64
import os
from typing import Dict, Any, List, Tuple, Optional
import logging
from .json_utils import safe_json_serialize

logger = logging.getLogger(__name__)

class FraudDetectionKMeans:
    """
    Class phát hiện gian lận sử dụng K-Means clustering với Elbow method
    Hỗ trợ cả pre-trained models và training on-demand
    """
    
    def __init__(self, use_pretrained: bool = True, models_dir: str = "trained_models"):
        self.scaler = StandardScaler()
        self.kmeans = None
        self.optimal_k = None
        self.numerical_columns = []
        self.elbow_data = None
        self.threshold = None
        self.use_pretrained = use_pretrained
        self.models_dir = models_dir
        
        # Nếu sử dụng pre-trained model, load ngay
        if self.use_pretrained:
            self._load_pretrained_model()
    
    def _load_pretrained_model(self):
        """Load pre-trained fraud detection model"""
        try:
            import pickle
            model_path = os.path.join(self.models_dir, "fraud_model.pkl")
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.scaler = model_data.get('scaler')
                self.kmeans = model_data.get('kmeans')
                self.numerical_columns = model_data.get('numerical_columns', [])
                self.optimal_k = model_data.get('optimal_k')
                self.threshold = model_data.get('threshold')
                self.elbow_data = model_data.get('elbow_data')
                
                logger.info(f"Pre-trained fraud model loaded with k={self.optimal_k}")
                return True
            else:
                logger.warning(f"Pre-trained model not found at {model_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading pre-trained model: {str(e)}")
            return False
    
    def detect_fraud_pretrained(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Phát hiện gian lận sử dụng pre-trained model (nhanh hơn)
        """
        try:
            # Kiểm tra model đã được load chưa
            if not self.kmeans or not self.scaler or not self.numerical_columns:
                logger.warning("Pre-trained model not available, falling back to on-demand training")
                return self.detect_fraud(df, contamination=0.05)
            
            # Preprocess data với cùng format như training
            df_processed = self.preprocess_data(df)
            
            # Kiểm tra columns
            missing_cols = [col for col in self.numerical_columns if col not in df_processed.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Lấy dữ liệu số và xử lý missing values
            X = df_processed[self.numerical_columns].copy()
            X = X.fillna(X.median())
            
            # Transform với pre-trained scaler
            X_scaled = self.scaler.transform(X)
            
            # Predict clusters
            cluster_labels = self.kmeans.predict(X_scaled)
            
            # Tính khoảng cách đến centroids
            distances = np.linalg.norm(
                X_scaled - self.kmeans.cluster_centers_[cluster_labels], 
                axis=1
            )
            
            # Sử dụng pre-trained threshold
            df_processed['cluster'] = cluster_labels
            df_processed['distance_to_centroid'] = distances
            df_processed['is_potential_fraud'] = distances > self.threshold
            df_processed['fraud_score'] = distances / distances.max()
            
            # Tạo risk levels
            df_processed['risk_level'] = df_processed['fraud_score'].apply(
                lambda x: 'Cao' if x > 0.8 else 'Trung bình' if x > 0.5 else 'Thấp'
            )
            
            # Lấy các giao dịch nghi ngờ
            suspicious_transactions = df_processed[df_processed['is_potential_fraud']].copy()
            suspicious_transactions = suspicious_transactions.sort_values('fraud_score', ascending=False)
            
            # Chuẩn bị kết quả
            results = {
                'total_transactions': int(len(df_processed)),
                'suspicious_count': int(len(suspicious_transactions)),
                'fraud_rate': float(round((len(suspicious_transactions) / len(df_processed)) * 100, 2)),
                'total_suspicious_amount': float(suspicious_transactions['transaction_amount'].sum()) if 'transaction_amount' in suspicious_transactions.columns else 0.0,
                'suspicious_transactions': self._format_suspicious_transactions(suspicious_transactions),
                'cluster_info': self._get_cluster_info(df_processed),
                'elbow_data': self.elbow_data,
                'scatter_data': self._create_scatter_data(X_scaled, cluster_labels, df_processed['is_potential_fraud']),
                'model_type': 'pre-trained'
            }
            
            # Convert numpy types to JSON serializable types
            return safe_json_serialize(results)
            
        except Exception as e:
            logger.error(f"Error in pre-trained fraud detection: {str(e)}")
            logger.info("Falling back to on-demand training")
            return self.detect_fraud(df, contamination=0.05)
        
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tiền xử lý dữ liệu cho phát hiện gian lận
        """
        try:
            # Chuẩn hóa tên cột
            df_processed = df.copy()
            
            # Map các tên cột khác nhau về chuẩn
            column_mapping = {
                'TransactionID': 'transaction_id',
                'AccountID': 'account_id', 
                'TransactionAmount': 'transaction_amount',
                'TransactionDate': 'transaction_date',
                'TransactionType': 'transaction_type',
                'Location': 'location',
                'DeviceID': 'device_id',
                'IP Address': 'ip_address',
                'MerchantID': 'merchant_id',
                'Channel': 'channel',
                'CustomerAge': 'customer_age',
                'CustomerOccupation': 'customer_occupation',
                'TransactionDuration': 'transaction_duration',
                'LoginAttempts': 'login_attempts',
                'AccountBalance': 'account_balance',
                'PreviousTransactionDate': 'previous_transaction_date'
            }
            
            # Rename columns if they exist
            for old_col, new_col in column_mapping.items():
                if old_col in df_processed.columns:
                    df_processed = df_processed.rename(columns={old_col: new_col})
            
            # Xử lý datetime
            if 'transaction_date' in df_processed.columns:
                df_processed['transaction_date'] = pd.to_datetime(df_processed['transaction_date'], errors='coerce')
                df_processed['hour'] = df_processed['transaction_date'].dt.hour
                df_processed['day_of_week'] = df_processed['transaction_date'].dt.dayofweek
                df_processed['is_weekend'] = df_processed['day_of_week'].isin([5, 6]).astype(int)
            
            # Lấy các cột số để phân tích
            self.numerical_columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
            
            # Loại bỏ các cột ID không cần thiết cho clustering
            exclude_columns = ['transaction_id', 'account_id', 'device_id', 'merchant_id']
            self.numerical_columns = [col for col in self.numerical_columns if col not in exclude_columns]
            
            logger.info(f"Numerical columns for clustering: {self.numerical_columns}")
            
            return df_processed
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise
    
    def elbow_method(self, X: np.ndarray, max_k: int = 10) -> Tuple[List[int], List[float], int]:
        """
        Thực hiện Elbow method để tìm số cụm tối ưu
        """
        try:
            wcss = []
            k_range = range(1, min(max_k + 1, len(X) + 1))
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X)
                wcss.append(kmeans.inertia_)
            
            # Tìm optimal k using elbow method
            # Tính độ dốc giữa các điểm
            slopes = []
            for i in range(1, len(wcss)):
                slope = wcss[i-1] - wcss[i]
                slopes.append(slope)
            
            # Tìm điểm có độ dốc giảm mạnh nhất (elbow)
            if len(slopes) >= 2:
                slope_changes = []
                for i in range(1, len(slopes)):
                    change = slopes[i-1] - slopes[i]
                    slope_changes.append(change)
                
                # Optimal k là điểm có slope change lớn nhất + 2 (vì index bắt đầu từ 1)
                optimal_k = slope_changes.index(max(slope_changes)) + 3
                optimal_k = min(optimal_k, max_k)  # Không vượt quá max_k
            else:
                optimal_k = 3  # Default fallback
            
            self.optimal_k = optimal_k
            self.elbow_data = {
                'k_values': list(k_range),
                'wcss_values': wcss,
                'optimal_k': optimal_k
            }
            
            logger.info(f"Optimal k found: {optimal_k}")
            return list(k_range), wcss, optimal_k
            
        except Exception as e:
            logger.error(f"Error in elbow method: {str(e)}")
            # Fallback to k=4 if error occurs
            self.optimal_k = 4
            return [1, 2, 3, 4], [1000, 800, 600, 500], 4
    
    def detect_fraud(self, df: pd.DataFrame, contamination: float = 0.05) -> Dict[str, Any]:
        """
        Phát hiện gian lận sử dụng K-Means clustering
        """
        try:
            # Preprocess data
            df_processed = self.preprocess_data(df)
            
            if not self.numerical_columns:
                raise ValueError("Không tìm thấy cột số để phân tích")
            
            # Lấy dữ liệu số và xử lý missing values
            X = df_processed[self.numerical_columns].copy()
            X = X.fillna(X.median())
            
            # Chuẩn hóa dữ liệu
            X_scaled = self.scaler.fit_transform(X)
            
            # Tìm optimal k using elbow method
            k_values, wcss_values, optimal_k = self.elbow_method(X_scaled)
            
            # Thực hiện K-Means clustering với optimal k
            self.kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            cluster_labels = self.kmeans.fit_predict(X_scaled)
            
            # Tính khoảng cách từ mỗi điểm đến centroid của cụm
            distances = np.linalg.norm(
                X_scaled - self.kmeans.cluster_centers_[cluster_labels], 
                axis=1
            )
            
            # Xác định threshold cho outliers (potential frauds)
            threshold_percentile = (1 - contamination) * 100
            threshold = np.percentile(distances, threshold_percentile)
            
            # Đánh dấu potential frauds
            df_processed['cluster'] = cluster_labels
            df_processed['distance_to_centroid'] = distances
            df_processed['is_potential_fraud'] = distances > threshold
            df_processed['fraud_score'] = distances / distances.max()  # Normalize to 0-1
            
            # Tạo risk levels
            df_processed['risk_level'] = df_processed['fraud_score'].apply(
                lambda x: 'Cao' if x > 0.8 else 'Trung bình' if x > 0.5 else 'Thấp'
            )
            
            # Lấy các giao dịch nghi ngờ
            suspicious_transactions = df_processed[df_processed['is_potential_fraud']].copy()
            suspicious_transactions = suspicious_transactions.sort_values('fraud_score', ascending=False)
            
            # Chuẩn bị kết quả
            results = {
                'total_transactions': int(len(df_processed)),
                'suspicious_count': int(len(suspicious_transactions)),
                'fraud_rate': float(round((len(suspicious_transactions) / len(df_processed)) * 100, 2)),
                'total_suspicious_amount': float(suspicious_transactions['transaction_amount'].sum()) if 'transaction_amount' in suspicious_transactions.columns else 0.0,
                'suspicious_transactions': self._format_suspicious_transactions(suspicious_transactions),
                'cluster_info': self._get_cluster_info(df_processed),
                'elbow_data': self.elbow_data,
                'scatter_data': self._create_scatter_data(X_scaled, cluster_labels, df_processed['is_potential_fraud']),
                'model_type': 'on-demand'
            }
            
            # Convert numpy types to JSON serializable types
            return safe_json_serialize(results)
            
        except Exception as e:
            logger.error(f"Error in fraud detection: {str(e)}")
            raise
    
    def _format_suspicious_transactions(self, suspicious_df: pd.DataFrame) -> List[Dict]:
        """
        Format suspicious transactions for display
        """
        transactions = []
        
        for idx, row in suspicious_df.head(20).iterrows():  # Top 20 suspicious
            transaction = {
                'transaction_id': str(row.get('transaction_id', 'N/A')),
                'account_id': str(row.get('account_id', 'N/A')),
                'amount': float(row.get('transaction_amount', 0)),
                'date': str(row.get('transaction_date', 'N/A')),
                'type': str(row.get('transaction_type', 'N/A')),
                'risk_level': str(row.get('risk_level', 'N/A')),
                'fraud_score': float(row.get('fraud_score', 0)),
                'reason': self._generate_fraud_reason(row)
            }
            transactions.append(transaction)
        
        return transactions
    
    def _generate_fraud_reason(self, row: pd.Series) -> str:
        """
        Tạo lý do tại sao giao dịch được coi là nghi ngờ
        """
        reasons = []
        
        try:
            # Check amount anomaly
            if row.get('transaction_amount', 0) > 10000000:  # > 10M VND
                reasons.append("Số tiền giao dịch cao bất thường")
            
            # Check time anomaly
            hour = row.get('hour', -1)
            if 0 <= hour <= 5 or hour >= 23:
                reasons.append("Giao dịch vào thời gian bất thường")
            
            # Check weekend large transactions
            if row.get('is_weekend', False) and row.get('transaction_amount', 0) > 5000000:
                reasons.append("Giao dịch lớn vào cuối tuần")
            
            # Check fraud score
            if row.get('fraud_score', 0) > 0.8:
                reasons.append("Điểm bất thường cao theo mô hình AI")
            
            return "; ".join(reasons) if reasons else "Bất thường theo mô hình machine learning"
            
        except:
            return "Không xác định được lý do cụ thể"
    
    def _get_cluster_info(self, df: pd.DataFrame) -> Dict:
        """
        Lấy thông tin về các cụm
        """
        try:
            cluster_counts = df['cluster'].value_counts().sort_index()
            
            cluster_data = {
                'labels': [f'Cụm {i}' for i in cluster_counts.index],
                'values': [int(v) for v in cluster_counts.tolist()],
                'total_clusters': int(len(cluster_counts))
            }
            
            return cluster_data
            
        except Exception as e:
            logger.error(f"Error getting cluster info: {str(e)}")
            return {'labels': [], 'values': [], 'total_clusters': 0}
    
    def _create_scatter_data(self, X_scaled: np.ndarray, cluster_labels: np.ndarray, 
                           is_fraud: pd.Series) -> Dict:
        """
        Tạo dữ liệu cho scatter plot
        """
        try:
            if X_scaled.shape[1] < 2:
                return {'datasets': []}
            
            datasets = []
            unique_clusters = np.unique(cluster_labels)
            colors = ['rgba(255, 99, 132, 0.6)', 'rgba(54, 162, 235, 0.6)', 
                     'rgba(255, 206, 86, 0.6)', 'rgba(75, 192, 192, 0.6)',
                     'rgba(153, 102, 255, 0.6)']
            
            # Normal clusters
            for i, cluster in enumerate(unique_clusters):
                cluster_mask = cluster_labels == cluster
                normal_mask = cluster_mask & ~is_fraud
                
                if np.any(normal_mask):
                    datasets.append({
                        'label': f'Cụm {cluster}',
                        'data': [
                            {'x': float(X_scaled[j, 0]), 'y': float(X_scaled[j, 1])}
                            for j in range(len(X_scaled)) if normal_mask[j]
                        ],
                        'backgroundColor': colors[i % len(colors)],
                        'pointRadius': 3
                    })
            
            # Fraud points
            fraud_mask = is_fraud.values
            if np.any(fraud_mask):
                datasets.append({
                    'label': 'Giao dịch nghi ngờ',
                    'data': [
                        {'x': float(X_scaled[j, 0]), 'y': float(X_scaled[j, 1])}
                        for j in range(len(X_scaled)) if fraud_mask[j]
                    ],
                    'backgroundColor': 'rgba(255, 0, 0, 0.8)',
                    'pointRadius': 6,
                    'pointStyle': 'cross'
                })
            
            return {'datasets': datasets}
            
        except Exception as e:
            logger.error(f"Error creating scatter data: {str(e)}")
            return {'datasets': []}
    
    def create_elbow_plot(self) -> Optional[str]:
        """
        Tạo biểu đồ elbow method và trả về base64 string
        """
        try:
            if not self.elbow_data:
                return None
            
            plt.figure(figsize=(10, 6))
            plt.plot(self.elbow_data['k_values'], self.elbow_data['wcss_values'], 
                    marker='o', linewidth=2, markersize=8)
            plt.xlabel('Number of clusters (k)', fontsize=12)
            plt.ylabel('WCSS', fontsize=12)
            plt.title('Elbow Method - Xác định số cụm tối ưu', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # Highlight optimal k
            optimal_idx = self.elbow_data['k_values'].index(self.elbow_data['optimal_k'])
            plt.annotate(f'Optimal k = {self.elbow_data["optimal_k"]}',
                        xy=(self.elbow_data['optimal_k'], self.elbow_data['wcss_values'][optimal_idx]),
                        xytext=(self.elbow_data['optimal_k'] + 1, self.elbow_data['wcss_values'][optimal_idx] + 50),
                        arrowprops=dict(arrowstyle='->', color='red'),
                        fontsize=12, color='red')
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plot_data = buffer.getvalue()
            buffer.close()
            plt.close()
            
            return base64.b64encode(plot_data).decode()
            
        except Exception as e:
            logger.error(f"Error creating elbow plot: {str(e)}")
            return None