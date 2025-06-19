"""
Django management command để train models
Usage: python manage.py train_models [--force]
"""
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from transactions.utils.model_trainer import ModelTrainer

class Command(BaseCommand):
    help = 'Train machine learning models for fraud detection and personal finance analysis'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force retrain even if models are up to date'
        )
        
        parser.add_argument(
            '--fraud-only',
            action='store_true',
            help='Train only fraud detection model'
        )
        
        parser.add_argument(
            '--personal-only',
            action='store_true',
            help='Train only personal finance model'
        )
    
    def handle(self, *args, **options):
        """Main command handler"""
        try:
            self.stdout.write("=" * 60)
            self.stdout.write(self.style.SUCCESS("MODEL TRAINING STARTED"))
            self.stdout.write("=" * 60)
            
            # Initialize trainer
            trainer = ModelTrainer()
            
            # Check if training is needed
            force_retrain = options.get('force', False)
            fraud_only = options.get('fraud_only', False)
            personal_only = options.get('personal_only', False)
            
            if not force_retrain and not trainer.need_retrain():
                self.stdout.write(
                    self.style.WARNING("Models are up to date. Use --force to retrain anyway.")
                )
                
                # Show current model info
                model_info = trainer.get_model_info()
                if model_info:
                    self.stdout.write("\nCurrent model information:")
                    self.stdout.write(f"  Training timestamp: {model_info.get('training_timestamp', 'Unknown')}")
                    self.stdout.write(f"  Model version: {model_info.get('model_version', 'Unknown')}")
                    
                    fraud_stats = model_info.get('fraud_model_stats', {})
                    personal_stats = model_info.get('personal_model_stats', {})
                    
                    if fraud_stats:
                        self.stdout.write(f"  Fraud model: {fraud_stats.get('total_samples', 0)} samples, {fraud_stats.get('features_count', 0)} features")
                    
                    if personal_stats:
                        self.stdout.write(f"  Personal model: {personal_stats.get('total_transactions', 0)} transactions")
                
                return
            
            # Check data files
            if not fraud_only:
                if not os.path.exists(trainer.personal_data_file):
                    raise CommandError(f"Personal finance data file not found: {trainer.personal_data_file}")
            
            if not personal_only:
                if not os.path.exists(trainer.fraud_data_file):
                    raise CommandError(f"Fraud detection data file not found: {trainer.fraud_data_file}")
            
            # Train models
            if fraud_only:
                self.stdout.write("Training fraud detection model only...")
                fraud_stats = trainer.train_fraud_detection_model()
                self.stdout.write(
                    self.style.SUCCESS(f"[OK] Fraud model trained: {fraud_stats.get('total_samples', 0)} samples")
                )
                
            elif personal_only:
                self.stdout.write("Training personal finance model only...")
                personal_stats = trainer.train_personal_finance_model()
                self.stdout.write(
                    self.style.SUCCESS(f"[OK] Personal model trained: {personal_stats.get('total_transactions', 0)} transactions")
                )
                
            else:
                self.stdout.write("Training all models...")
                results = trainer.train_all_models()
                
                self.stdout.write(
                    self.style.SUCCESS("[OK] All models trained successfully!")
                )
                
                # Display results
                fraud_stats = results.get('fraud_model', {})
                personal_stats = results.get('personal_model', {})
                
                self.stdout.write("\nTraining Results:")
                self.stdout.write(f"  Timestamp: {results.get('timestamp', 'Unknown')}")
                
                if fraud_stats:
                    self.stdout.write(f"  Fraud Detection:")
                    self.stdout.write(f"    - Samples: {fraud_stats.get('total_samples', 0)}")
                    self.stdout.write(f"    - Features: {fraud_stats.get('features_count', 0)}")
                    self.stdout.write(f"    - Threshold: {fraud_stats.get('threshold_percentile', 95)}th percentile")
                
                if personal_stats:
                    self.stdout.write(f"  Personal Finance:")
                    self.stdout.write(f"    - Transactions: {personal_stats.get('total_transactions', 0)}")
                    self.stdout.write(f"    - Total spending: {personal_stats.get('total_spending', 0):,.2f}")
                    self.stdout.write(f"    - Avg transaction: {personal_stats.get('avg_transaction', 0):,.2f}")
            
            self.stdout.write("\n" + "=" * 60)
            self.stdout.write(self.style.SUCCESS("MODEL TRAINING COMPLETED"))
            self.stdout.write("=" * 60)
            
        except FileNotFoundError as e:
            raise CommandError(f"Required file not found: {str(e)}")
        except Exception as e:
            raise CommandError(f"Training failed: {str(e)}")
    
    def check_data_files(self):
        """Check if required data files exist"""
        files = ['phgl.xlsx', 'canhan.xlsx']
        missing = []
        
        for file in files:
            if not os.path.exists(file):
                missing.append(file)
        
        if missing:
            raise CommandError(f"Missing data files: {', '.join(missing)}")
        
        return True