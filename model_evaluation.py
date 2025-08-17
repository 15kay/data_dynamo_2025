import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, validation_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self, model_path='best_model.pkl', data_path='combined_data.pkl'):
        """
        Initialize the model evaluator
        """
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✓ Model loaded from {model_path}")
        except FileNotFoundError:
            print(f"⚠ Model file {model_path} not found. Will use default model.")
            self.model = None
            
        try:
            with open(data_path, 'rb') as f:
                self.data = pickle.load(f)
            print(f"✓ Data loaded from {data_path}")
        except FileNotFoundError:
            print(f"⚠ Data file {data_path} not found. Will generate sample data.")
            self.data = self._generate_sample_data()
    
    def _generate_sample_data(self):
        """
        Generate sample data for evaluation if files are not found
        """
        np.random.seed(42)
        n_samples = 100
        
        # Generate synthetic features
        features = {
            'Service_Gap': np.random.uniform(0, 50, n_samples),
            'Refuse_Service_Pct': np.random.uniform(60, 95, n_samples),
            'Service_Delivery_Index': np.random.uniform(0.3, 0.9, n_samples),
            'Water_Access_Pct': np.random.uniform(70, 98, n_samples),
            'Electricity_Access_Pct': np.random.uniform(75, 99, n_samples),
            'Sanitation_Access_Pct': np.random.uniform(65, 95, n_samples),
            'Education_Access_Pct': np.random.uniform(80, 98, n_samples),
            'Population_Density': np.random.uniform(10, 1000, n_samples),
            'Year_Trend': np.random.uniform(0, 5, n_samples)
        }
        
        df = pd.DataFrame(features)
        
        # Generate target variable based on features
        df['Protest_Risk'] = (
            df['Service_Gap'] * 0.3 + 
            (100 - df['Refuse_Service_Pct']) * 0.2 +
            (1 - df['Service_Delivery_Index']) * 20 +
            np.random.normal(0, 2, n_samples)
        )
        
        return df
    
    def prepare_data(self):
        """
        Prepare features and target for evaluation
        """
        feature_columns = [
            'Service_Gap', 'Refuse_Service_Pct', 'Service_Delivery_Index',
            'Water_Access_Pct', 'Electricity_Access_Pct', 'Sanitation_Access_Pct',
            'Education_Access_Pct', 'Population_Density', 'Year_Trend'
        ]
        
        # Select available features
        available_features = [col for col in feature_columns if col in self.data.columns]
        
        if not available_features:
            print("⚠ No standard features found. Using all numeric columns.")
            available_features = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if 'Protest_Risk' in available_features:
                available_features.remove('Protest_Risk')
        
        X = self.data[available_features]
        y = self.data.get('Protest_Risk', self.data.iloc[:, -1])  # Use last column if Protest_Risk not found
        
        print(f"Features used: {available_features}")
        print(f"Data shape: {X.shape}")
        
        return X, y, available_features
    
    def cross_validation_analysis(self, X, y, cv=5):
        """
        Perform cross-validation analysis
        """
        if self.model is None:
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='r2')
        cv_rmse = np.sqrt(-cross_val_score(self.model, X, y, cv=cv, scoring='neg_mean_squared_error'))
        cv_mae = -cross_val_score(self.model, X, y, cv=cv, scoring='neg_mean_absolute_error')
        
        results = {
            'R² Scores': cv_scores,
            'RMSE Scores': cv_rmse,
            'MAE Scores': cv_mae,
            'Mean R²': cv_scores.mean(),
            'Std R²': cv_scores.std(),
            'Mean RMSE': cv_rmse.mean(),
            'Std RMSE': cv_rmse.std(),
            'Mean MAE': cv_mae.mean(),
            'Std MAE': cv_mae.std()
        }
        
        return results
    
    def learning_curve_analysis(self, X, y):
        """
        Analyze learning curves to detect overfitting/underfitting
        """
        if self.model is None:
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_scores = []
        val_scores = []
        
        for train_size in train_sizes:
            n_samples = int(train_size * len(X))
            if n_samples < 10:  # Minimum samples for training
                continue
                
            X_subset = X.iloc[:n_samples]
            y_subset = y.iloc[:n_samples]
            
            # Cross-validation on subset
            cv_scores = cross_val_score(self.model, X_subset, y_subset, cv=3, scoring='r2')
            train_scores.append(cv_scores.mean())
            
            # Validation on remaining data
            if n_samples < len(X):
                self.model.fit(X_subset, y_subset)
                val_pred = self.model.predict(X.iloc[n_samples:])
                val_score = r2_score(y.iloc[n_samples:], val_pred)
                val_scores.append(val_score)
            else:
                val_scores.append(cv_scores.mean())
        
        return train_sizes[:len(train_scores)], train_scores, val_scores
    
    def feature_importance_analysis(self, X, y, feature_names):
        """
        Analyze feature importance and stability
        """
        if self.model is None:
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Fit model to get feature importance
        self.model.fit(X, y)
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            return importance_df
        else:
            print("⚠ Model does not support feature importance analysis")
            return None
    
    def residual_analysis(self, X, y):
        """
        Perform residual analysis
        """
        if self.model is None:
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Fit and predict
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        
        # Calculate residuals
        residuals = y - y_pred
        
        # Calculate metrics
        metrics = {
            'R²': r2_score(y, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
            'MAE': mean_absolute_error(y, y_pred),
            'Mean Residual': residuals.mean(),
            'Std Residual': residuals.std(),
            'Min Residual': residuals.min(),
            'Max Residual': residuals.max()
        }
        
        return y_pred, residuals, metrics
    
    def create_evaluation_plots(self, X, y, feature_names):
        """
        Create comprehensive evaluation plots
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Evaluation Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Cross-validation scores
        cv_results = self.cross_validation_analysis(X, y)
        axes[0, 0].bar(['R²', 'RMSE', 'MAE'], 
                      [cv_results['Mean R²'], cv_results['Mean RMSE'], cv_results['Mean MAE']])
        axes[0, 0].set_title('Cross-Validation Metrics')
        axes[0, 0].set_ylabel('Score')
        
        # 2. Learning curves
        train_sizes, train_scores, val_scores = self.learning_curve_analysis(X, y)
        axes[0, 1].plot(train_sizes, train_scores, 'o-', label='Training Score', color='blue')
        axes[0, 1].plot(train_sizes, val_scores, 'o-', label='Validation Score', color='red')
        axes[0, 1].set_title('Learning Curves')
        axes[0, 1].set_xlabel('Training Set Size')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Feature importance
        importance_df = self.feature_importance_analysis(X, y, feature_names)
        if importance_df is not None:
            top_features = importance_df.head(8)
            axes[0, 2].barh(top_features['Feature'], top_features['Importance'])
            axes[0, 2].set_title('Feature Importance')
            axes[0, 2].set_xlabel('Importance')
        
        # 4. Residual analysis
        y_pred, residuals, metrics = self.residual_analysis(X, y)
        axes[1, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[1, 0].axhline(y=0, color='red', linestyle='--')
        axes[1, 0].set_title('Residuals vs Predicted')
        axes[1, 0].set_xlabel('Predicted Values')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Actual vs Predicted
        axes[1, 1].scatter(y, y_pred, alpha=0.6)
        min_val, max_val = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'red', linestyle='--')
        axes[1, 1].set_title(f'Actual vs Predicted (R² = {metrics["R²"]:.3f})')
        axes[1, 1].set_xlabel('Actual Values')
        axes[1, 1].set_ylabel('Predicted Values')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Residual distribution
        axes[1, 2].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 2].axvline(x=0, color='red', linestyle='--')
        axes[1, 2].set_title('Residual Distribution')
        axes[1, 2].set_xlabel('Residuals')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_evaluation_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return metrics, cv_results
    
    def generate_evaluation_report(self):
        """
        Generate comprehensive evaluation report
        """
        print("\n" + "="*60)
        print("         SERVICE DELIVERY PROTEST MODEL EVALUATION")
        print("="*60)
        
        # Prepare data
        X, y, feature_names = self.prepare_data()
        
        # Create evaluation plots and get metrics
        metrics, cv_results = self.create_evaluation_plots(X, y, feature_names)
        
        # Print detailed report
        print(f"\n📊 DATASET OVERVIEW")
        print(f"   • Sample Size: {len(X):,} observations")
        print(f"   • Features: {len(feature_names)} variables")
        print(f"   • Target Range: {y.min():.2f} to {y.max():.2f}")
        print(f"   • Target Mean: {y.mean():.2f} ± {y.std():.2f}")
        
        print(f"\n🎯 MODEL PERFORMANCE METRICS")
        print(f"   • R² Score: {metrics['R²']:.3f}")
        print(f"   • RMSE: {metrics['RMSE']:.3f}")
        print(f"   • MAE: {metrics['MAE']:.3f}")
        print(f"   • Mean Residual: {metrics['Mean Residual']:.3f}")
        
        print(f"\n🔄 CROSS-VALIDATION RESULTS")
        print(f"   • Mean R²: {cv_results['Mean R²']:.3f} ± {cv_results['Std R²']:.3f}")
        print(f"   • Mean RMSE: {cv_results['Mean RMSE']:.3f} ± {cv_results['Std RMSE']:.3f}")
        print(f"   • Mean MAE: {cv_results['Mean MAE']:.3f} ± {cv_results['Std MAE']:.3f}")
        
        # Feature importance
        importance_df = self.feature_importance_analysis(X, y, feature_names)
        if importance_df is not None:
            print(f"\n🔍 TOP FEATURE IMPORTANCE")
            for idx, row in importance_df.head(5).iterrows():
                print(f"   • {row['Feature']}: {row['Importance']:.3f}")
        
        # Model interpretation
        print(f"\n📈 MODEL INTERPRETATION")
        if metrics['R²'] > 0.7:
            print(f"   • Excellent model performance (R² > 0.7)")
        elif metrics['R²'] > 0.5:
            print(f"   • Good model performance (R² > 0.5)")
        elif metrics['R²'] > 0.3:
            print(f"   • Moderate model performance (R² > 0.3)")
        else:
            print(f"   • Poor model performance (R² < 0.3)")
        
        if abs(metrics['Mean Residual']) < 0.1:
            print(f"   • Model is well-calibrated (low bias)")
        else:
            print(f"   • Model shows some bias (mean residual: {metrics['Mean Residual']:.3f})")
        
        print(f"\n💡 RECOMMENDATIONS")
        if cv_results['Std R²'] > 0.1:
            print(f"   • High variance in CV scores - consider more data or regularization")
        if metrics['RMSE'] > y.std():
            print(f"   • RMSE higher than target std - model may need improvement")
        if importance_df is not None and len(importance_df) > 0:
            top_feature = importance_df.iloc[0]['Feature']
            print(f"   • Focus on '{top_feature}' as the most predictive feature")
        
        print(f"\n📁 OUTPUT FILES")
        print(f"   • Evaluation dashboard: model_evaluation_dashboard.png")
        print(f"   • Detailed metrics saved for further analysis")
        
        print("\n" + "="*60)
        
        return {
            'metrics': metrics,
            'cv_results': cv_results,
            'feature_importance': importance_df,
            'data_info': {
                'n_samples': len(X),
                'n_features': len(feature_names),
                'target_range': (y.min(), y.max()),
                'target_stats': (y.mean(), y.std())
            }
        }

def main():
    """
    Main execution function
    """
    print("🚀 Starting Model Evaluation...")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Generate comprehensive evaluation report
    results = evaluator.generate_evaluation_report()
    
    print("\n✅ Model evaluation completed successfully!")
    return results

if __name__ == "__main__":
    results = main()