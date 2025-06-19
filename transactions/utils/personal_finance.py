"""
Personal Finance Analysis Module
Phân tích tài chính cá nhân và đưa ra gợi ý tiết kiệm
"""
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, List, Tuple
import logging
from datetime import datetime, timedelta
import re
from .json_utils import safe_json_serialize

logger = logging.getLogger(__name__)

class PersonalFinanceAnalyzer:
    """
    Class phân tích tài chính cá nhân
    Hỗ trợ cả pre-trained models và analysis on-demand
    """
    
    def __init__(self, use_pretrained: bool = True, models_dir: str = "trained_models"):
        self.use_pretrained = use_pretrained
        self.models_dir = models_dir
        self.pretrained_model = None
        
        # Keywords based on actual transaction data patterns from canhan.xlsx
        self.category_keywords = {
            'Ăn uống': ['tiền ăn', 'tien an', 'an uong', 'ăn uống', 'cafe', 'restaurant', 'food', 'drink', 'beer', 'coffee', 'com', 'pho', 'quan', 'nha hang', 'mc', 'kfc', 'burger', 'pizza', 'lotteria', 'highlands', 'starbucks', 'tra sua', 'banh mi', 'bun', 'mien', 'chao', 'xoi', 'che', 'sua chua', 'kem', 'nuoc', 'coca', 'pepsi', 'tra', 'sua', 'ruou', 'bia'],
            'Mua sắm': ['tiền mua sắm', 'tien mua sam', 'mua sắm', 'mua sam', 'shop', 'store', 'buy', 'purchase', 'lazada', 'shopee', 'tiki', 'sendo', 'hang', 'do dung', 'trang tri', 'my pham', 'mua', 'order', 'dat hang', 'fashion', 'clothing', 'ao', 'quan', 'giay', 'tui', 'dep', 'mu', 'phone', 'laptop', 'computer', 'samsung', 'apple', 'iphone', 'supermarket', 'market', 'grocery', 'coopmart', 'big c', 'lotte mart', 'vinmart'],
            'Giao thông': ['tiền xe', 'tien xe', 'grab', 'taxi', 'uber', 'gojek', 'be', 'mai linh', 'vinasun', 'cuoc xe', 'petrol', 'gas', 'xang', 'dau', 'petrolimex', 'bus', 'train', 'metro', 've xe', 'parking', 'do xe', 'gui xe'],
            'Học tập': ['tiền học', 'tien hoc', 'hoc phi', 'học phí', 'university', 'college', 'truong', 'dai hoc', 'cao dang', 'book', 'sach', 'vo', 'but', 'course', 'dao tao', 'hoc them', 'education', 'study', 'school'],
            'Nhà cửa': ['tiền nhà', 'tien nha', 'house', 'home', 'nha', 'thue nha', 'rent', 'mortgage', 'sua nha', 'noi that', 'furniture', 'repair', 'sua chua', 'bao tri'],
            'Điện nước': ['tiền điện nước', 'tien dien nuoc', 'electric', 'dien', 'evn', 'tien dien', 'hoa don dien', 'water', 'nuoc', 'tien nuoc', 'hoa don nuoc', 'internet', 'wifi', 'phone', 'mobile', 'viettel', 'vinaphone', 'mobifone', 'fpt', 'vnpt', 'gas', 'bill', 'utility', 'hoa don'],
            'Khác': []
        }
        
        # Nếu sử dụng pre-trained model, load ngay
        if self.use_pretrained:
            self._load_pretrained_model()
    
    def _load_pretrained_model(self):
        """Load pre-trained personal finance model"""
        try:
            import pickle
            model_path = os.path.join(self.models_dir, "personal_model.pkl")
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.pretrained_model = pickle.load(f)
                
                # Update category keywords from trained model if available
                if 'category_keywords' in self.pretrained_model:
                    self.category_keywords = self.pretrained_model['category_keywords']
                
                logger.info("Pre-trained personal finance model loaded")
                return True
            else:
                logger.warning(f"Pre-trained personal model not found at {model_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading pre-trained personal model: {str(e)}")
            return False
    
    def analyze_personal_finance_pretrained(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Phân tích tài chính cá nhân sử dụng pre-trained model (nhanh hơn và chính xác hơn)
        """
        try:
            logger.info("Starting personal finance analysis...")
            
            # Kiểm tra model đã được load chưa
            if not self.pretrained_model:
                logger.warning("Pre-trained model not available, falling back to standard analysis")
                return self.analyze_personal_finance(df)
            
            logger.info("Preprocessing data...")
            # Preprocess data
            df_processed = self.preprocess_data(df)
            
            if df_processed.empty:
                raise ValueError("No spending data found for analysis")
            
            logger.info(f"Processed {len(df_processed)} transactions, starting detailed analysis...")
            
            # Sử dụng insights từ pre-trained model
            category_insights = self.pretrained_model.get('category_insights', {})
            overall_stats = self.pretrained_model.get('overall_stats', {})
            
            logger.info("Analyzing overview...")
            # Overview analysis với enhanced insights
            overview = self._analyze_overview_enhanced(df_processed, overall_stats)
            
            logger.info("Analyzing categories...")
            # Category analysis với pre-trained insights
            categories = self._analyze_categories_enhanced(df_processed, category_insights)
            
            logger.info("Analyzing trends...")
            # Trend analysis
            trend = self._analyze_trend(df_processed)
            
            logger.info("Generating savings suggestions...")
            # Enhanced savings suggestions với pre-trained insights
            savings = self._generate_savings_suggestions_enhanced(df_processed, categories, category_insights)
            
            logger.info("Generating detailed report...")
            # Detailed report
            detailed = self._generate_detailed_report(df_processed)
            
            logger.info("Compiling results...")
            results = {
                'overview': overview,
                'categories': categories,
                'trend': trend,
                'savings': savings,
                'detailed': detailed,
                'model_type': 'pre-trained',
                'training_period': self.pretrained_model.get('training_period', {}),
                'benchmark_comparison': self._compare_with_benchmark(df_processed, overall_stats)
            }
            
            logger.info("Converting to JSON-safe format...")
            # Convert numpy types to JSON serializable types
            return safe_json_serialize(results)
            
        except Exception as e:
            logger.error(f"Error in pre-trained personal finance analysis: {str(e)}")
            logger.info("Falling back to standard analysis")
            return self.analyze_personal_finance(df)
    
    def _analyze_overview_enhanced(self, df: pd.DataFrame, benchmark_stats: Dict) -> Dict[str, Any]:
        """Enhanced overview analysis với benchmark comparison"""
        overview = self._analyze_overview(df)
        
        if benchmark_stats:
            # So sánh với benchmark
            benchmark_avg = benchmark_stats.get('avg_transaction', 0)
            user_avg = overview.get('avg_transaction_amount', 0)
            
            if benchmark_avg > 0:
                comparison = ((user_avg - benchmark_avg) / benchmark_avg) * 100
                overview['benchmark_interpretation'] = 'cao hơn trung bình' if comparison > 0 else 'thấp hơn trung bình'
                overview['benchmark_percentage'] = float(comparison)
        
        return overview
    
    def _analyze_categories_enhanced(self, df: pd.DataFrame, category_insights: Dict) -> Dict[str, Any]:
        """Enhanced category analysis với pre-trained insights"""
        categories = self._analyze_categories(df)
        
        # Thêm insights từ training data
        if category_insights:
            for detail in categories['details']:
                category_name = detail['name']
                if category_name in category_insights:
                    insight = category_insights[category_name]
                    detail['benchmark_avg'] = insight.get('avg_amount', 0)
                    detail['benchmark_percentage'] = insight.get('percentage', 0)
                    
                    # So sánh với benchmark
                    if insight.get('avg_amount', 0) > 0:
                        comparison = ((detail['amount'] / len(df) - insight['avg_amount']) / insight['avg_amount']) * 100
                        detail['vs_benchmark'] = float(comparison)
        
        return categories
    
    def _generate_savings_suggestions_enhanced(self, df: pd.DataFrame, categories: Dict, category_insights: Dict) -> Dict[str, Any]:
        """Enhanced savings suggestions với pre-trained insights"""
        savings = self._generate_savings_suggestions(df, categories)
        
        # Thêm suggestions dựa trên benchmark
        if category_insights:
            enhanced_suggestions = []
            
            for suggestion in savings['suggestions']:
                enhanced_suggestions.append(suggestion)
            
            # Thêm suggestions dựa trên so sánh với benchmark
            for detail in categories['details']:
                category_name = detail['name']
                if category_name in category_insights:
                    insight = category_insights[category_name]
                    benchmark_pct = insight.get('percentage', 0)
                    user_pct = detail['percentage']
                    
                    if user_pct > benchmark_pct * 1.2:  # 20% higher than benchmark
                        potential_savings = detail['amount'] * 0.15  # 15% reduction
                        enhanced_suggestions.append({
                            'title': f'Giảm chi tiêu {category_name}',
                            'description': f'Chi tiêu {category_name} của bạn cao hơn {user_pct - benchmark_pct:.1f}% so với trung bình. Hãy cân nhắc giảm bớt.',
                            'category': category_name,
                            'current_spending': detail['amount'],
                            'saving_amount': potential_savings,
                            'saving_percentage': 15
                        })
            
            savings['suggestions'] = enhanced_suggestions[:8]  # Limit to 8 suggestions
            
            # Recalculate potential savings
            total_potential = sum([s.get('saving_amount', 0) for s in savings['suggestions']])
            savings['potential'] = total_potential
        
        return savings
    
    def _compare_with_benchmark(self, df: pd.DataFrame, benchmark_stats: Dict) -> Dict[str, Any]:
        """So sánh với benchmark từ training data"""
        if not benchmark_stats:
            return {}
        
        total_spending = float(df['amount'].sum())
        total_transactions = len(df)
        avg_transaction = total_spending / total_transactions if total_transactions > 0 else 0
        
        benchmark_total = benchmark_stats.get('total_spending', 0)
        benchmark_avg = benchmark_stats.get('avg_transaction', 0)
        benchmark_transactions = benchmark_stats.get('total_transactions', 0)
        
        comparison = {}
        
        if benchmark_avg > 0:
            avg_diff = ((avg_transaction - benchmark_avg) / benchmark_avg) * 100
            comparison['avg_transaction'] = {
                'user': float(avg_transaction),
                'benchmark': float(benchmark_avg),
                'difference_percent': float(avg_diff),
                'interpretation': 'cao hơn' if avg_diff > 0 else 'thấp hơn'
            }
        
        if benchmark_transactions > 0:
            freq_diff = ((total_transactions - benchmark_transactions) / benchmark_transactions) * 100
            comparison['transaction_frequency'] = {
                'user': int(total_transactions),
                'benchmark': int(benchmark_transactions),
                'difference_percent': float(freq_diff),
                'interpretation': 'nhiều hơn' if freq_diff > 0 else 'ít hơn'
            }
        
        return comparison
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tiền xử lý dữ liệu tài chính cá nhân
        """
        try:
            df_processed = df.copy()
            
            # Chuẩn hóa tên cột - updated for actual data structure
            column_mapping = {
                'Mã giao dịch': 'transaction_id',
                'Thời gian': 'datetime',
                'Ngày giờ chuyển': 'datetime',
                'ID người nhận': 'recipient_id',
                'Số tiền': 'amount',
                'Số tiền giao dịch': 'amount',
                'Số dư hiện tại': 'balance',
                'Số dư tài khoản': 'balance',
                'Trạng thái': 'status',
                'Trạng thái giao dịch': 'status',
                'Loại giao dịch': 'transaction_type',
                'Nội dung giao dịch': 'description'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in df_processed.columns:
                    df_processed = df_processed.rename(columns={old_col: new_col})
            
            # Xử lý datetime
            if 'datetime' in df_processed.columns:
                df_processed['datetime'] = pd.to_datetime(df_processed['datetime'], errors='coerce')
                df_processed['date'] = df_processed['datetime'].dt.date
                df_processed['hour'] = df_processed['datetime'].dt.hour
                df_processed['day_of_week'] = df_processed['datetime'].dt.dayofweek
                df_processed['is_weekend'] = df_processed['day_of_week'].isin([5, 6])
                df_processed['week'] = df_processed['datetime'].dt.isocalendar().week
                df_processed['month'] = df_processed['datetime'].dt.month
            
            # Lọc bỏ các giao dịch thu nhập trước khi xử lý amount
            # 1. Lọc theo loại giao dịch nếu có
            if 'transaction_type' in df_processed.columns:
                # Chỉ giữ lại "Chuyển tiền" (chi tiêu), loại bỏ "Nhận tiền" (thu nhập)
                df_processed = df_processed[df_processed['transaction_type'] != 'Nhận tiền'].copy()
            
            # 2. Lọc theo nội dung - loại bỏ thu nhập
            if 'description' in df_processed.columns:
                income_keywords = ['tiền lương', 'tien luong', 'lương', 'tiền thưởng', 'tien thuong', 'thưởng', 'thu nhập', 'thu nhap']
                for keyword in income_keywords:
                    df_processed = df_processed[~df_processed['description'].str.contains(keyword, case=False, na=False)]
            
            # Xử lý amount
            if 'amount' in df_processed.columns:
                df_processed['amount'] = pd.to_numeric(df_processed['amount'], errors='coerce')
                
                # Kiểm tra xem có giao dịch âm không (chi tiêu)
                negative_count = (df_processed['amount'] < 0).sum()
                positive_count = (df_processed['amount'] > 0).sum()
                
                if negative_count > 0:
                    # Có giao dịch âm - lấy giao dịch chi tiêu (âm) và chuyển thành dương
                    df_processed = df_processed[df_processed['amount'] < 0].copy()
                    df_processed['amount'] = df_processed['amount'].abs()
                elif positive_count > 0:
                    # Không có giao dịch âm - giữ lại giao dịch dương (đã lọc thu nhập)
                    df_processed = df_processed[df_processed['amount'] > 0].copy()
                else:
                    # Không có dữ liệu hợp lệ
                    df_processed = df_processed.iloc[0:0]  # Empty dataframe
            
            # Phân loại giao dịch theo category
            df_processed['category'] = df_processed.apply(self._categorize_transaction, axis=1)
            
            # Debug categorization results
            category_counts = df_processed['category'].value_counts()
            logger.info(f"Categorization results:")
            for category, count in category_counts.items():
                percentage = (count / len(df_processed)) * 100
                logger.info(f"  {category}: {count} transactions ({percentage:.1f}%)")
            
            # Log sample transactions for debugging
            logger.info("Sample categorized transactions:")
            for i, row in df_processed.head(10).iterrows():
                logger.info(f"  '{row.get('description', '')}' -> {row['category']}")
            
            logger.info(f"Processed {len(df_processed)} spending transactions")
            return df_processed
            
        except Exception as e:
            logger.error(f"Error in preprocessing personal finance data: {str(e)}")
            raise
    
    def _categorize_transaction(self, row: pd.Series) -> str:
        """
        Phân loại giao dịch dựa trên nội dung và loại giao dịch - Cải tiến với exact match
        """
        try:
            description = str(row.get('description', '')).lower().strip()
            transaction_type = str(row.get('transaction_type', '')).lower().strip()
            
            # Kết hợp description và transaction_type để phân loại
            text_to_analyze = f"{description} {transaction_type}".lower()
            
            # Loại bỏ các ký tự đặc biệt và normalize text
            import re
            text_to_analyze = re.sub(r'[^\w\s]', ' ', text_to_analyze)
            text_to_analyze = ' '.join(text_to_analyze.split())  # Remove extra spaces
            
            # Exact match trước tiên cho các descriptions phổ biến
            exact_matches = {
                'tiền điện nước': 'Điện nước',
                'tien dien nuoc': 'Điện nước',
                'ăn uống': 'Ăn uống',
                'an uong': 'Ăn uống',
                'tiền ăn': 'Ăn uống',
                'tien an': 'Ăn uống',
                'mua sắm': 'Mua sắm', 
                'mua sam': 'Mua sắm',
                'tiền xe': 'Giao thông',
                'tien xe': 'Giao thông',
                'tiền học': 'Học tập',
                'tien hoc': 'Học tập',
                'tiền nhà': 'Nhà cửa',
                'tien nha': 'Nhà cửa'
            }
            
            # Kiểm tra exact match trước
            for phrase, category in exact_matches.items():
                if phrase in text_to_analyze:
                    return category
            
            # Phân loại theo mức độ ưu tiên (categories cụ thể trước)
            category_scores = {}
            
            for category, keywords in self.category_keywords.items():
                if category == 'Khác':
                    continue
                
                score = 0
                matches = []
                
                for keyword in keywords:
                    keyword_clean = keyword.lower().strip()
                    
                    # Simple contains match - avoid complex regex for performance
                    if keyword_clean in text_to_analyze:
                        # Bonus score for exact word match
                        if keyword_clean == text_to_analyze:
                            score += 10
                        else:
                            score += 5
                        matches.append(keyword)
                
                if score > 0:
                    category_scores[category] = {'score': score, 'matches': matches}
            
            # Trả về category có điểm cao nhất
            if category_scores:
                best_category = max(category_scores.items(), key=lambda x: x[1]['score'])
                return best_category[0]
            
            # Fallback: Thử phân loại theo patterns phổ biến
            fallback_category = self._fallback_categorization(text_to_analyze)
            if fallback_category != 'Khác':
                return fallback_category
            
            # Special handling cho các trường hợp đặc biệt
            special_category = self._special_categorization(text_to_analyze)
            if special_category != 'Khác':
                return special_category
            
            return 'Khác'
            
        except Exception as e:
            logger.error(f"Error in categorization: {str(e)}")
            return 'Khác'
    
    def _fallback_categorization(self, text: str) -> str:
        """
        Fallback categorization dựa trên patterns phổ biến
        """
        # Common transaction patterns - simplified to match 7 main categories
        patterns = {
            'Ăn uống': [
                r'\b(com|pho|bun|mien|chao|an|uong|drink|food)\b',
                r'\b(cafe|coffee|tra|nuoc)\b',
                r'\b(quan|nha hang|restaurant)\b'
            ],
            'Mua sắm': [
                r'\b(mua|buy|shop|store|market)\b',
                r'\b(hang|san pham|product)\b',
                r'\b(thanh toan|payment|pay)\b'
            ],
            'Giao thông': [
                r'\b(xe|taxi|grab|uber|cuoc)\b',
                r'\b(di chuyen|transport|vehicle)\b',
                r'\b(xang|petrol|gas)\b'
            ],
            'Điện nước': [
                r'\b(hoa don|bill|dien|nuoc|internet|phi)\b',
                r'\b(dich vu|service|utility)\b'
            ],
            'Học tập': [
                r'\b(hoc|study|education|truong|school)\b',
                r'\b(sach|book|course|khoa hoc)\b'
            ],
            'Nhà cửa': [
                r'\b(nha|house|home|rent|thue)\b',
                r'\b(noi that|furniture|sua chua|repair)\b'
            ]
        }
        
        for category, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, text, re.IGNORECASE):
                    return category
        
        return 'Khác'
    
    def _special_categorization(self, text: str) -> str:
        """
        Xử lý đặc biệt cho các trường hợp phức tạp
        """
        # Xử lý cụm từ có nhiều từ khóa - ưu tiên theo độ cụ thể
        # Case đặc biệt: "tiền ăn chuyển tiền" -> ưu tiên "tiền ăn"
        if ('tien an' in text or 'an tien' in text) and 'chuyen tien' in text:
            return 'Ăn uống'  # Ưu tiên mục đích sử dụng tiền
        elif 'tien an' in text or 'an tien' in text:
            return 'Ăn uống'
        elif 'chuyen tien' in text or 'chuyen khoan' in text:
            return 'Khác'  # Chuyển tiền được phân vào Khác
        if any(word in text for word in ['thanh toan', 'payment', 'pay']) and not any(word in text for word in ['chuyen', 'transfer']):
            return 'Mua sắm'
        if any(word in text for word in ['mua', 'buy', 'order']) and any(word in text for word in ['quan ao', 'giay', 'tui']):
            return 'Mua sắm'
        if any(word in text for word in ['grab', 'taxi', 'uber']) or 'cuoc xe' in text:
            return 'Giao thông'
        if any(word in text for word in ['xang', 'petrol', 'gas']) and 'xe' in text:
            return 'Giao thông'
        if any(word in text for word in ['dien', 'nuoc', 'internet', 'wifi']) and any(word in text for word in ['hoa don', 'bill', 'phi']):
            return 'Điện nước'
        
        return 'Khác'
    
    def analyze_personal_finance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Phân tích tài chính cá nhân toàn diện
        """
        try:
            # Preprocess data
            df_processed = self.preprocess_data(df)
            
            if df_processed.empty:
                raise ValueError("No spending data found for analysis")
            
            # Overview analysis
            overview = self._analyze_overview(df_processed)
            
            # Category analysis
            categories = self._analyze_categories(df_processed)
            
            # Trend analysis
            trend = self._analyze_trend(df_processed)
            
            # Savings suggestions
            savings = self._generate_savings_suggestions(df_processed, categories)
            
            # Detailed report
            detailed = self._generate_detailed_report(df_processed)
            
            results = {
                'overview': overview,
                'categories': categories,
                'trend': trend,
                'savings': savings,
                'detailed': detailed,
                'model_type': 'on-demand'
            }
            
            # Convert numpy types to JSON serializable types
            return safe_json_serialize(results)
            
        except Exception as e:
            logger.error(f"Error in personal finance analysis: {str(e)}")
            raise
    
    def _analyze_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Phân tích tổng quan
        """
        try:
            total_spending = float(df['amount'].sum())
            total_transactions = len(df)
            
            # Tính số ngày phân tích
            if 'datetime' in df.columns and not df['datetime'].isna().all():
                date_range = df['datetime'].max() - df['datetime'].min()
                analysis_days = max(1, date_range.days + 1)
            else:
                analysis_days = 30  # Default
            
            avg_daily_spending = total_spending / analysis_days
            
            return {
                'total_spending': float(total_spending),
                'total_transactions': int(total_transactions),
                'analysis_days': int(analysis_days),
                'avg_daily_spending': float(avg_daily_spending),
                'avg_transaction_amount': float(total_spending / total_transactions) if total_transactions > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error in overview analysis: {str(e)}")
            return {
                'total_spending': 0.0,
                'total_transactions': 0,
                'analysis_days': 0,
                'avg_daily_spending': 0.0,
                'avg_transaction_amount': 0.0
            }
    
    def _analyze_categories(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Phân tích chi tiêu theo danh mục
        """
        try:
            category_spending = df.groupby('category')['amount'].sum().sort_values(ascending=False)
            total_spending = category_spending.sum()
            
            # Tạo dữ liệu cho biểu đồ
            labels = category_spending.index.tolist()
            values = [float(v) for v in category_spending.values.tolist()]
            
            # Tạo chi tiết cho bảng
            details = []
            for category, amount in category_spending.items():
                percentage = (amount / total_spending * 100) if total_spending > 0 else 0
                details.append({
                    'name': category,
                    'amount': float(amount),
                    'percentage': round(percentage, 1)
                })
            
            return {
                'labels': labels,
                'values': values,
                'details': details
            }
            
        except Exception as e:
            logger.error(f"Error in category analysis: {str(e)}")
            return {'labels': [], 'values': [], 'details': []}
    
    def _analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Phân tích xu hướng chi tiêu theo thời gian
        """
        try:
            if 'date' not in df.columns:
                return {'labels': [], 'values': []}
            
            # Nhóm theo ngày
            daily_spending = df.groupby('date')['amount'].sum().sort_index()
            
            # Nếu có quá nhiều ngày, nhóm theo tuần
            if len(daily_spending) > 30:
                weekly_spending = df.groupby(df['datetime'].dt.to_period('W'))['amount'].sum()
                labels = [f"Tuần {i+1}" for i in range(len(weekly_spending))]
                values = [float(v) for v in weekly_spending.values.tolist()]
            else:
                labels = [str(date) for date in daily_spending.index]
                values = [float(v) for v in daily_spending.values.tolist()]
            
            return {
                'labels': labels,
                'values': values
            }
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {str(e)}")
            return {'labels': [], 'values': []}
    
    def _generate_savings_suggestions(self, df: pd.DataFrame, categories: Dict) -> Dict[str, Any]:
        """
        Tạo gợi ý tiết kiệm
        """
        try:
            total_spending = float(df['amount'].sum())
            target_savings_rate = 0.15  # 15% tiết kiệm
            target_monthly_savings = total_spending * target_savings_rate
            
            suggestions = []
            total_potential_savings = 0
            
            # Phân tích từng danh mục để đưa ra gợi ý
            for category_info in categories['details']:
                category = category_info['name']
                amount = category_info['amount']
                percentage = category_info['percentage']
                
                # Gợi ý dựa trên danh mục và mức độ chi tiêu
                suggestion = self._create_category_suggestion(category, amount, percentage, total_spending)
                if suggestion:
                    suggestions.append(suggestion)
                    total_potential_savings += suggestion['saving_amount']
            
            # Gợi ý chung
            general_suggestions = self._create_general_suggestions(df, total_spending)
            suggestions.extend(general_suggestions)
            
            # Tính lại tổng tiết kiệm từ tất cả gợi ý
            total_potential_savings = sum(s['saving_amount'] for s in suggestions)
            
            # Tính khả năng đạt được mục tiêu
            achievable_rate = min(100, (total_potential_savings / target_monthly_savings * 100)) if target_monthly_savings > 0 else 0
            
            # Đảm bảo có ít nhất 5 gợi ý
            if len(suggestions) < 5:
                # Thêm gợi ý mặc định nếu không đủ
                default_suggestions = self._get_default_suggestions(total_spending)
                suggestions.extend(default_suggestions[:(5 - len(suggestions))])
                # Tính lại tổng tiết kiệm
                total_potential_savings = sum(s['saving_amount'] for s in suggestions)
                achievable_rate = min(100, (total_potential_savings / target_monthly_savings * 100)) if target_monthly_savings > 0 else 0
            
            return {
                'target': target_monthly_savings,
                'potential': total_potential_savings,
                'achievable_rate': round(achievable_rate, 1),
                'suggestions': suggestions[:10]  # Tăng lên 10 gợi ý
            }
            
        except Exception as e:
            logger.error(f"Error in savings suggestions: {str(e)}")
            return {'target': 0, 'potential': 0, 'achievable_rate': 0, 'suggestions': []}
    
    def _create_category_suggestion(self, category: str, amount: float, 
                                  percentage: float, total_spending: float) -> Dict[str, Any]:
        """
        Tạo gợi ý tiết kiệm cho từng danh mục
        """
        suggestions_map = {
            'Ăn uống': {
                'title': 'Giảm chi phí ăn uống',
                'description': 'Nấu ăn tại nhà thay vì ăn ngoài, mang theo nước uống để tránh mua đồ uống đắt tiền',
                'saving_rate': 0.25
            },
            'Mua sắm - Thời trang': {
                'title': 'Mua sắm thời trang thông minh',
                'description': 'Chờ sale, mua hàng outlet, mix & match quần áo hiện có thay vì mua mới',
                'saving_rate': 0.30
            },
            'Mua sắm - Điện tử': {
                'title': 'Cân nhắc mua thiết bị điện tử',
                'description': 'Mua hàng refurbished, so sánh giá nhiều nơi, chờ khuyến mãi lớn',
                'saving_rate': 0.20
            },
            'Mua sắm - Thực phẩm': {
                'title': 'Mua thực phẩm hiệu quả',
                'description': 'Lập kế hoạch bữa ăn, mua sỉ, tận dụng khuyến mãi siêu thị',
                'saving_rate': 0.15
            },
            'Mua sắm - Khác': {
                'title': 'Mua sắm thông minh',
                'description': 'Lập danh sách mua sắm, tìm kiếm khuyến mãi, so sánh giá trước khi mua',
                'saving_rate': 0.20
            },
            'Giao thông - Taxi/Grab': {
                'title': 'Giảm chi phí taxi',
                'description': 'Sử dụng xe buýt, metro, đi chung xe hoặc đi xe máy cá nhân',
                'saving_rate': 0.40
            },
            'Giao thông - Xăng dầu': {
                'title': 'Tiết kiệm xăng dầu',
                'description': 'Lái xe êm, bảo dưỡng xe định kỳ, kết hợp nhiều việc trong một chuyến đi',
                'saving_rate': 0.15
            },
            'Giao thông - Phương tiện công cộng': {
                'title': 'Tối ưu vé tháng',
                'description': 'Mua vé tháng thay vì vé lẻ, sử dụng các app có khuyến mãi',
                'saving_rate': 0.10
            },
            'Giải trí - Phim ảnh': {
                'title': 'Tiết kiệm chi phí giải trí',
                'description': 'Xem phim vào suất chiếu sớm, sử dụng thẻ thành viên, xem phim tại nhà',
                'saving_rate': 0.25
            },
            'Giải trí - Game': {
                'title': 'Kiểm soát chi phí game',
                'description': 'Đặt ngân sách cố định cho game, tìm game miễn phí chất lượng',
                'saving_rate': 0.50
            },
            'Giải trí - Thể thao': {
                'title': 'Tập thể thao tiết kiệm',
                'description': 'Tập tại công viên, sử dụng video YouTube, mua gói tập dài hạn',
                'saving_rate': 0.30
            },
            'Y tế - Nhà thuốc': {
                'title': 'Mua thuốc thông minh',
                'description': 'Mua thuốc generic, so sánh giá nhiều nhà thuốc, mua theo đơn',
                'saving_rate': 0.20
            },
            'Hóa đơn - Điện': {
                'title': 'Tiết kiệm điện',
                'description': 'Tắt thiết bị không dùng, sử dụng đèn LED, điều hòa 26°C',
                'saving_rate': 0.15
            },
            'Hóa đơn - Internet/Điện thoại': {
                'title': 'Tối ưu gói cước',
                'description': 'Chọn gói phù hợp nhu cầu, sử dụng wifi thay vì 4G, gộp chung gói gia đình',
                'saving_rate': 0.20
            }
        }
        
        # Tìm suggestion phù hợp với category (exact match hoặc partial match)
        suggestion_info = None
        for key, value in suggestions_map.items():
            if key == category or key in category or category in key:
                suggestion_info = value
                break
        
        if suggestion_info and percentage > 5:  # Giảm threshold xuống 5% để có nhiều gợi ý hơn
            saving_amount = amount * suggestion_info['saving_rate']
            
            return {
                'title': suggestion_info['title'],
                'description': suggestion_info['description'],
                'category': category,
                'current_spending': amount,
                'saving_amount': saving_amount,
                'saving_percentage': suggestion_info['saving_rate'] * 100
            }
        
        return None
    
    def _create_general_suggestions(self, df: pd.DataFrame, total_spending: float) -> List[Dict]:
        """
        Tạo gợi ý chung - đảm bảo có ít nhất 5 gợi ý
        """
        suggestions = []
        
        # Gợi ý về thời gian chi tiêu
        if 'hour' in df.columns:
            late_night_spending = df[df['hour'].between(22, 23)]['amount'].sum()
            if late_night_spending > total_spending * 0.10:  # Giảm threshold
                suggestions.append({
                    'title': 'Hạn chế chi tiêu ban đêm',
                    'description': 'Bạn có xu hướng chi tiêu nhiều vào ban đêm. Hãy cân nhắc kỹ trước khi mua sắm vào thời gian này',
                    'category': 'Thói quen',
                    'current_spending': late_night_spending,
                    'saving_amount': late_night_spending * 0.3,
                    'saving_percentage': 30
                })
                
            # Gợi ý về chi tiêu cuối tuần
            weekend_spending = df[df['is_weekend'] == True]['amount'].sum() if 'is_weekend' in df.columns else 0
            if weekend_spending > total_spending * 0.3:
                suggestions.append({
                    'title': 'Kiểm soát chi tiêu cuối tuần',
                    'description': 'Chi tiêu cuối tuần chiếm tỷ lệ cao. Hãy lập kế hoạch ngân sách cho cuối tuần',
                    'category': 'Quản lý thời gian',
                    'current_spending': weekend_spending,
                    'saving_amount': weekend_spending * 0.15,
                    'saving_percentage': 15
                })
        
        # Gợi ý về giao dịch lớn
        large_transactions = df[df['amount'] > df['amount'].quantile(0.8)]  # Giảm từ 0.9 xuống 0.8
        if len(large_transactions) > 0:
            large_amount = large_transactions['amount'].sum()
            suggestions.append({
                'title': 'Cân nhắc các giao dịch lớn',
                'description': 'Dành thời gian suy nghĩ 24h trước khi thực hiện các giao dịch lớn để tránh mua sắm theo cảm tính',
                'category': 'Quản lý tài chính',
                'current_spending': large_amount,
                'saving_amount': large_amount * 0.1,
                'saving_percentage': 10
            })
        
        # Gợi ý về tần suất giao dịch
        daily_avg_transactions = len(df) / max(1, df['datetime'].dt.date.nunique()) if 'datetime' in df.columns else len(df) / 30
        if daily_avg_transactions > 3:
            frequent_small_amount = df[df['amount'] < df['amount'].median()]['amount'].sum()
            suggestions.append({
                'title': 'Giảm giao dịch nhỏ lẻ',
                'description': f'Bạn có {daily_avg_transactions:.1f} giao dịch/ngày. Hãy gộp các mua sắm nhỏ để tiết kiệm phí và thời gian',
                'category': 'Thói quen chi tiêu',
                'current_spending': frequent_small_amount,
                'saving_amount': frequent_small_amount * 0.05,
                'saving_percentage': 5
            })
        
        # Gợi ý về quản lý ngân sách tổng thể
        monthly_spending = total_spending
        suggestions.append({
            'title': 'Thiết lập ngân sách hàng tháng',
            'description': f'Với chi tiêu hiện tại {monthly_spending:,.0f} VND/tháng, hãy áp dụng quy tắc 50-30-20: 50% nhu cầu thiết yếu, 30% giải trí, 20% tiết kiệm',
            'category': 'Quản lý ngân sách',
            'current_spending': monthly_spending,
            'saving_amount': monthly_spending * 0.20,
            'saving_percentage': 20
        })
        
        # Gợi ý về theo dõi chi tiêu
        suggestions.append({
            'title': 'Ghi chép chi tiêu hàng ngày',
            'description': 'Việc ghi chép chi tiêu giúp bạn nhận thức rõ hơn về thói quen tiêu dùng và có thể tiết kiệm 10-15%',
            'category': 'Công cụ quản lý',
            'current_spending': monthly_spending,
            'saving_amount': monthly_spending * 0.125,
            'saving_percentage': 12.5
        })
        
        # Gợi ý về mua sắm theo kế hoạch
        suggestions.append({
            'title': 'Lập danh sách mua sắm',
            'description': 'Lập danh sách và ngân sách trước khi đi mua sắm để tránh mua những thứ không cần thiết',
            'category': 'Kỹ năng tiết kiệm',
            'current_spending': monthly_spending * 0.3,  # Ước tính 30% chi tiêu có thể kiểm soát được
            'saving_amount': monthly_spending * 0.1,
            'saving_percentage': 10
        })
        
        # Gợi ý về tận dụng khuyến mãi
        suggestions.append({
            'title': 'Tận dụng khuyến mãi và cashback',
            'description': 'Sử dụng app ngân hàng, thẻ tín dụng có cashback, và theo dõi khuyến mãi để tiết kiệm 5-10%',
            'category': 'Cơ hội tiết kiệm',
            'current_spending': monthly_spending,
            'saving_amount': monthly_spending * 0.075,
            'saving_percentage': 7.5
        })
        
        return suggestions
    
    def _get_default_suggestions(self, total_spending: float) -> List[Dict]:
        """
        Tạo các gợi ý mặc định khi không đủ gợi ý từ dữ liệu
        """
        return [
            {
                'title': 'Áp dụng quy tắc 24h',
                'description': 'Trước khi mua bất cứ thứ gì không cần thiết, hãy chờ 24h để suy nghĩ lại',
                'category': 'Thói quen tài chính',
                'current_spending': total_spending,
                'saving_amount': total_spending * 0.08,
                'saving_percentage': 8
            },
            {
                'title': 'So sánh giá trước khi mua',
                'description': 'Luôn so sánh giá ít nhất 3 nơi trước khi quyết định mua sắm',
                'category': 'Kỹ năng mua sắm',
                'current_spending': total_spending * 0.4,
                'saving_amount': total_spending * 0.06,
                'saving_percentage': 6
            },
            {
                'title': 'Thiết lập quỹ khẩn cấp',
                'description': 'Dành 10% thu nhập hàng tháng cho quỹ khẩn cấp để tránh vay nợ',
                'category': 'An toàn tài chính',
                'current_spending': total_spending,
                'saving_amount': total_spending * 0.1,
                'saving_percentage': 10
            },
            {
                'title': 'Mua hàng cũ chất lượng tốt',
                'description': 'Xem xét mua đồ cũ cho các mặt hàng ít sử dụng như sách, đồ trang trí',
                'category': 'Lựa chọn thông minh',
                'current_spending': total_spending * 0.2,
                'saving_amount': total_spending * 0.04,
                'saving_percentage': 4
            },
            {
                'title': 'Tận dụng chương trình loyalty',
                'description': 'Tham gia chương trình khách hàng thân thiết của các cửa hàng thường xuyên',
                'category': 'Tối ưu hóa chi phí',
                'current_spending': total_spending,
                'saving_amount': total_spending * 0.03,
                'saving_percentage': 3
            }
        ]
    
    def _generate_detailed_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Tạo báo cáo chi tiết với các tiêu chí phù hợp
        """
        try:
            # Chi tiêu lớn nhất (loại bỏ các giao dịch có thể là lương/thu nhập)
            # Chỉ lấy giao dịch < median * 5 để tránh thu nhập
            median_amount = df['amount'].median()
            expense_df = df[df['amount'] < median_amount * 5]  # Lọc bỏ các giao dịch quá lớn (có thể là lương)
            
            top_expenses = expense_df.nlargest(10, 'amount')[['amount', 'description', 'datetime']].copy()
            top_expenses_list = []
            
            for _, row in top_expenses.iterrows():
                description = str(row['description']).strip()
                # Làm sạch description
                if description.lower() in ['nan', 'none', '']:
                    description = 'Giao dịch không rõ mục đích'
                elif len(description) > 50:
                    description = description[:50] + '...'
                    
                top_expenses_list.append({
                    'amount': float(row['amount']),
                    'description': description,
                    'date': str(row['datetime'].date()) if pd.notna(row['datetime']) else 'N/A'
                })
            
            # Chi tiêu theo giờ trong ngày (sửa lỗi chỉ hiển thị 0:00)
            if 'hour' in df.columns and not df['hour'].isna().all():
                hourly_spending = df.groupby('hour')['amount'].sum().sort_index()
                # Đảm bảo có đủ 24 giờ
                all_hours = range(24)
                hourly_data = {}
                for hour in all_hours:
                    hourly_data[hour] = hourly_spending.get(hour, 0)
                
                hourly_labels = [f"{hour:02d}:00" for hour in all_hours]
                hourly_values = [float(hourly_data[hour]) for hour in all_hours]
            else:
                # Fallback: tạo dữ liệu giả theo thời gian trong ngày
                if 'datetime' in df.columns:
                    df['hour'] = df['datetime'].dt.hour
                    hourly_spending = df.groupby('hour')['amount'].sum().sort_index()
                    hourly_labels = [f"{int(hour):02d}:00" for hour in hourly_spending.index]
                    hourly_values = [float(v) for v in hourly_spending.values]
                else:
                    # Nếu không có dữ liệu thời gian, tạo phân bố ngẫu nhiên
                    import random
                    hourly_labels = [f"{hour:02d}:00" for hour in range(24)]
                    total_amount = df['amount'].sum()
                    # Phân bố theo pattern thực tế: sáng ít, trưa + chiều nhiều, tối vừa phải
                    weights = [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.05, 0.08, 0.08, 0.06, 
                              0.09, 0.12, 0.08, 0.06, 0.05, 0.04, 0.05, 0.08, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01]
                    hourly_values = [float(total_amount * w) for w in weights]
            
            # Chi tiêu theo ngày trong tuần
            if 'datetime' in df.columns:
                daily_spending = df.groupby(df['datetime'].dt.day_name())['amount'].sum()
                days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                daily_labels_vn = ['Thứ 2', 'Thứ 3', 'Thứ 4', 'Thứ 5', 'Thứ 6', 'Thứ 7', 'Chủ nhật']
                daily_values = [float(daily_spending.get(day, 0)) for day in days_order]
            else:
                daily_labels_vn = ['Thứ 2', 'Thứ 3', 'Thứ 4', 'Thứ 5', 'Thứ 6', 'Thứ 7', 'Chủ nhật']
                daily_values = [0] * 7
            
            # Phân tích thói quen chi tiêu
            spending_insights = []
            
            # Insight về tần suất
            total_days = (df['datetime'].max() - df['datetime'].min()).days + 1 if 'datetime' in df.columns else 30
            avg_transactions_per_day = len(df) / max(1, total_days)
            spending_insights.append(f"Trung bình {avg_transactions_per_day:.1f} giao dịch/ngày")
            
            # Insight về số tiền trung bình
            avg_amount = df['amount'].mean()
            median_amount = df['amount'].median()
            spending_insights.append(f"Chi tiêu trung bình: {avg_amount:,.0f} VND/giao dịch")
            spending_insights.append(f"Chi tiêu phổ biến: {median_amount:,.0f} VND/giao dịch")
            
            # Insight về thời gian
            if 'hour' in df.columns:
                peak_hour = df.groupby('hour')['amount'].sum().idxmax()
                spending_insights.append(f"Khung giờ chi tiêu nhiều nhất: {int(peak_hour):02d}:00")
            
            return {
                'top_expenses': top_expenses_list,
                'hourly_spending': {
                    'labels': hourly_labels,
                    'values': hourly_values
                },
                'daily_spending': {
                    'labels': daily_labels_vn,
                    'values': daily_values
                },
                'spending_insights': spending_insights
            }
            
        except Exception as e:
            logger.error(f"Error in detailed report: {str(e)}")
            return {
                'top_expenses': [],
                'hourly_spending': {'labels': [], 'values': []},
                'daily_spending': {'labels': [], 'values': []},
                'spending_insights': []
            }