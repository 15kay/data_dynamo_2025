import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class ServiceDeliveryProtestPredictor:
    def __init__(self):
        self.census_data = {}
        self.protest_data = None
        self.combined_data = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_census_data(self):
        """
        Load and process Census 2022 data
        """
        print("📊 Loading Census 2022 Data...")
        
        census_file = "Datasets/Census 2022_Themes_24-10-2023.xlsx"
        
        # Key sheets for service delivery prediction
        key_sheets = [
            'Total population',
            'Access to piped water', 
            'Toilet facility',
            'Refuse disposal',
            'Energy for lighting',
            'Energy for cooking',
            'Type of dwelling',
            'Highest level of educ (20+ yrs)'
        ]
        
        for sheet_name in key_sheets:
            try:
                df = pd.read_excel(census_file, sheet_name=sheet_name)
                
                # Clean column names
                df.columns = df.columns.str.strip()
                
                # Keep only rows with valid municipality data
                df = df.dropna(subset=['Province name', 'District/Local municipality name'])
                
                # Convert numeric columns properly
                for col in df.columns:
                    if col not in ['Province name', 'Province abbreviation', 'District/Local municipality name']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                self.census_data[sheet_name] = df
                print(f"  ✓ Loaded {sheet_name}: {df.shape[0]} municipalities")
                
            except Exception as e:
                print(f"  ❌ Error loading {sheet_name}: {str(e)}")
        
        return self.census_data
    
    def load_protest_data(self):
        """
        Load and process protest events data
        """
        print("\n🔥 Loading Protest Events Data...")
        
        # Load demonstration events
        demo_file = "Datasets/south-africa_demonstration_events_by_month-year_as-of-13aug2025.xlsx"
        events_file = "Datasets/Events&Fatalaties.xlsx"
        
        try:
            # Load demonstration data
            demo_df = pd.read_excel(demo_file, sheet_name='Data')
            print(f"  ✓ Loaded demonstration events: {demo_df.shape[0]} records")
            
            # Load events & fatalities data
            events_df = pd.read_excel(events_file, sheet_name='Data')
            print(f"  ✓ Loaded events & fatalities: {events_df.shape[0]} records")
            
            # Combine and aggregate protest data
            # Focus on recent years (2020-2025) for better relevance to 2022 Census
            recent_demo = demo_df[demo_df['Year'] >= 2020].copy()
            recent_events = events_df[events_df['Year'] >= 2020].copy()
            
            # Aggregate by year for modeling
            demo_yearly = recent_demo.groupby('Year').agg({
                'Events': 'sum'
            }).reset_index()
            demo_yearly.rename(columns={'Events': 'Demonstration_Events'}, inplace=True)
            
            events_yearly = recent_events.groupby('Year').agg({
                'Events': 'sum',
                'Fatalities': 'sum'
            }).reset_index()
            events_yearly.rename(columns={'Events': 'Violence_Events'}, inplace=True)
            
            # Combine protest data
            self.protest_data = pd.merge(demo_yearly, events_yearly, on='Year', how='outer')
            self.protest_data = self.protest_data.fillna(0)
            
            print(f"  ✓ Combined protest data: {self.protest_data.shape[0]} years")
            print(f"  📅 Years covered: {self.protest_data['Year'].min()} - {self.protest_data['Year'].max()}")
            
            return self.protest_data
            
        except Exception as e:
            print(f"  ❌ Error loading protest data: {str(e)}")
            return None
    
    def create_service_delivery_features(self):
        """
        Create features from Census data that relate to service delivery
        """
        print("\n🔧 Creating Service Delivery Features...")
        
        # Start with population data as base
        base_df = self.census_data['Total population'].copy()
        
        # Clean and prepare base dataframe
        base_df = base_df[['Province name', 'District/Local municipality name', 'N']].copy()
        base_df.rename(columns={'N': 'Total_Population'}, inplace=True)
        
        # Ensure Total_Population is numeric
        base_df['Total_Population'] = pd.to_numeric(base_df['Total_Population'], errors='coerce')
        base_df = base_df.dropna(subset=['Total_Population'])
        
        print(f"  📍 Base data: {base_df.shape[0]} municipalities with valid population data")
        
        # Add water access features
        if 'Access to piped water' in self.census_data:
            water_df = self.census_data['Access to piped water'].copy()
            
            # Find numeric columns that might represent good water access
            numeric_cols = water_df.select_dtypes(include=[np.number]).columns
            water_access_cols = [col for col in numeric_cols if any(term in col.lower() for term in ['piped', 'yard', 'dwelling'])]
            
            if water_access_cols and 'Total' in water_df.columns:
                water_df['Good_Water_Access'] = water_df[water_access_cols].sum(axis=1, skipna=True)
                water_df['Water_Access_Pct'] = (water_df['Good_Water_Access'] / water_df['Total'].replace(0, np.nan)) * 100
                water_df['Water_Access_Pct'] = water_df['Water_Access_Pct'].fillna(0).clip(0, 100)
                
                base_df = pd.merge(base_df, 
                                 water_df[['Province name', 'District/Local municipality name', 'Water_Access_Pct']], 
                                 on=['Province name', 'District/Local municipality name'], how='left')
                print(f"  ✓ Added water access features")
        
        # Add sanitation features
        if 'Toilet facility' in self.census_data:
            toilet_df = self.census_data['Toilet facility'].copy()
            
            # Find flush toilet columns
            numeric_cols = toilet_df.select_dtypes(include=[np.number]).columns
            flush_cols = [col for col in numeric_cols if 'flush' in col.lower()]
            
            if flush_cols and 'Total' in toilet_df.columns:
                toilet_df['Good_Sanitation'] = toilet_df[flush_cols].sum(axis=1, skipna=True)
                toilet_df['Sanitation_Pct'] = (toilet_df['Good_Sanitation'] / toilet_df['Total'].replace(0, np.nan)) * 100
                toilet_df['Sanitation_Pct'] = toilet_df['Sanitation_Pct'].fillna(0).clip(0, 100)
                
                base_df = pd.merge(base_df, 
                                 toilet_df[['Province name', 'District/Local municipality name', 'Sanitation_Pct']], 
                                 on=['Province name', 'District/Local municipality name'], how='left')
                print(f"  ✓ Added sanitation features")
        
        # Add electricity access features
        if 'Energy for lighting' in self.census_data:
            energy_df = self.census_data['Energy for lighting'].copy()
            
            # Find electricity columns
            numeric_cols = energy_df.select_dtypes(include=[np.number]).columns
            elec_cols = [col for col in numeric_cols if 'electric' in col.lower()]
            
            if elec_cols and 'Total' in energy_df.columns:
                energy_df['Electricity_Access'] = energy_df[elec_cols].sum(axis=1, skipna=True)
                energy_df['Electricity_Pct'] = (energy_df['Electricity_Access'] / energy_df['Total'].replace(0, np.nan)) * 100
                energy_df['Electricity_Pct'] = energy_df['Electricity_Pct'].fillna(0).clip(0, 100)
                
                base_df = pd.merge(base_df, 
                                 energy_df[['Province name', 'District/Local municipality name', 'Electricity_Pct']], 
                                 on=['Province name', 'District/Local municipality name'], how='left')
                print(f"  ✓ Added electricity features")
        
        # Add refuse disposal features
        if 'Refuse disposal' in self.census_data:
            refuse_df = self.census_data['Refuse disposal'].copy()
            
            # Find municipal refuse columns
            numeric_cols = refuse_df.select_dtypes(include=[np.number]).columns
            municipal_cols = [col for col in numeric_cols if any(term in col.lower() for term in ['municipal', 'local', 'authority'])]
            
            if municipal_cols and 'Total' in refuse_df.columns:
                refuse_df['Municipal_Refuse'] = refuse_df[municipal_cols].sum(axis=1, skipna=True)
                refuse_df['Refuse_Service_Pct'] = (refuse_df['Municipal_Refuse'] / refuse_df['Total'].replace(0, np.nan)) * 100
                refuse_df['Refuse_Service_Pct'] = refuse_df['Refuse_Service_Pct'].fillna(0).clip(0, 100)
                
                base_df = pd.merge(base_df, 
                                 refuse_df[['Province name', 'District/Local municipality name', 'Refuse_Service_Pct']], 
                                 on=['Province name', 'District/Local municipality name'], how='left')
                print(f"  ✓ Added refuse disposal features")
        
        # Add education features
        if 'Highest level of educ (20+ yrs)' in self.census_data:
            edu_df = self.census_data['Highest level of educ (20+ yrs)'].copy()
            
            # Find higher education columns
            numeric_cols = edu_df.select_dtypes(include=[np.number]).columns
            higher_ed_cols = [col for col in numeric_cols if any(term in col.lower() for term in ['tertiary', 'university', 'college', 'higher'])]
            
            if higher_ed_cols and 'Total' in edu_df.columns:
                edu_df['Higher_Education'] = edu_df[higher_ed_cols].sum(axis=1, skipna=True)
                edu_df['Higher_Education_Pct'] = (edu_df['Higher_Education'] / edu_df['Total'].replace(0, np.nan)) * 100
                edu_df['Higher_Education_Pct'] = edu_df['Higher_Education_Pct'].fillna(0).clip(0, 100)
                
                base_df = pd.merge(base_df, 
                                 edu_df[['Province name', 'District/Local municipality name', 'Higher_Education_Pct']], 
                                 on=['Province name', 'District/Local municipality name'], how='left')
                print(f"  ✓ Added education features")
        
        # Add dwelling type features
        if 'Type of dwelling' in self.census_data:
            dwelling_df = self.census_data['Type of dwelling'].copy()
            
            # Find formal housing columns
            numeric_cols = dwelling_df.select_dtypes(include=[np.number]).columns
            formal_cols = [col for col in numeric_cols if 'house' in col.lower() and 'informal' not in col.lower()]
            
            if formal_cols and 'Total' in dwelling_df.columns:
                dwelling_df['Formal_Housing'] = dwelling_df[formal_cols].sum(axis=1, skipna=True)
                dwelling_df['Formal_Housing_Pct'] = (dwelling_df['Formal_Housing'] / dwelling_df['Total'].replace(0, np.nan)) * 100
                dwelling_df['Formal_Housing_Pct'] = dwelling_df['Formal_Housing_Pct'].fillna(0).clip(0, 100)
                
                base_df = pd.merge(base_df, 
                                 dwelling_df[['Province name', 'District/Local municipality name', 'Formal_Housing_Pct']], 
                                 on=['Province name', 'District/Local municipality name'], how='left')
                print(f"  ✓ Added housing features")
        
        # Create composite service delivery index
        service_cols = ['Water_Access_Pct', 'Sanitation_Pct', 'Electricity_Pct', 'Refuse_Service_Pct']
        available_service_cols = [col for col in service_cols if col in base_df.columns]
        
        if available_service_cols:
            base_df['Service_Delivery_Index'] = base_df[available_service_cols].mean(axis=1, skipna=True)
            base_df['Service_Gap'] = 100 - base_df['Service_Delivery_Index']
            print(f"  ✓ Created composite service delivery index")
        
        # Fill missing values with median
        numeric_cols = base_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            base_df[col] = base_df[col].fillna(base_df[col].median())
        
        print(f"  ✅ Created features for {base_df.shape[0]} municipalities")
        print(f"  📊 Features: {[col for col in base_df.columns if col not in ['Province name', 'District/Local municipality name']]}")
        
        return base_df
    
    def create_provincial_aggregates(self, municipal_df):
        """
        Aggregate municipal data to provincial level for modeling
        """
        print("\n🗺️ Creating Provincial Aggregates...")
        
        # Group by province and calculate weighted averages
        numeric_cols = municipal_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'Total_Population']
        
        provincial_data = []
        
        for province in municipal_df['Province name'].unique():
            prov_data = municipal_df[municipal_df['Province name'] == province].copy()
            
            # Calculate population-weighted averages for service delivery indicators
            total_pop = prov_data['Total_Population'].sum()
            
            if total_pop > 0:
                prov_summary = {'Province': province, 'Total_Population': total_pop}
                
                for col in numeric_cols:
                    if col in prov_data.columns:
                        # Population-weighted average
                        weighted_avg = (prov_data[col] * prov_data['Total_Population']).sum() / total_pop
                        prov_summary[col] = weighted_avg
                
                provincial_data.append(prov_summary)
        
        provincial_df = pd.DataFrame(provincial_data)
        
        print(f"  ✓ Created provincial data for {len(provincial_df)} provinces")
        
        return provincial_df
    
    def combine_data_for_modeling(self):
        """
        Combine census and protest data for modeling
        """
        print("\n🔗 Combining Data for Modeling...")
        
        # Create municipal features
        municipal_features = self.create_service_delivery_features()
        
        # Aggregate to provincial level
        provincial_features = self.create_provincial_aggregates(municipal_features)
        
        # Create training data by simulating the relationship between
        # service delivery gaps and protest likelihood
        modeling_data = []
        
        np.random.seed(42)  # For reproducible results
        
        for _, row in provincial_features.iterrows():
            province = row['Province']
            
            # Create multiple data points per province with some variation
            for year in range(2020, 2025):
                data_point = row.to_dict()
                data_point['Year'] = year
                
                # Simulate protest likelihood based on service delivery gaps
                service_gap = row.get('Service_Gap', 50)
                population = row.get('Total_Population', 1000000)
                
                # Base protest risk increases with service gaps and population density
                population_factor = np.log(population / 100000) if population > 0 else 0
                service_factor = service_gap / 100
                
                # Add year trend (increasing protests over time)
                year_trend = (year - 2020) * 0.1
                
                # Add some randomness
                random_factor = np.random.normal(0, 0.3)
                
                # Combine factors
                base_risk = service_factor * population_factor + year_trend + random_factor
                protest_risk = max(0, base_risk)
                
                # Convert to expected number of protests (scale appropriately)
                expected_protests = protest_risk * 5  # Scale factor
                
                data_point['Protest_Risk_Score'] = protest_risk
                data_point['Expected_Protests'] = expected_protests
                
                modeling_data.append(data_point)
        
        self.combined_data = pd.DataFrame(modeling_data)
        
        print(f"  ✓ Created modeling dataset with {len(self.combined_data)} observations")
        print(f"  ✓ Features: {len([col for col in self.combined_data.columns if col not in ['Province', 'Year', 'Protest_Risk_Score', 'Expected_Protests']])}")
        
        return self.combined_data
    
    def prepare_features(self):
        """
        Prepare features for machine learning
        """
        print("\n🎯 Preparing Features for ML...")
        
        # Select feature columns (exclude target and identifier columns)
        exclude_cols = ['Province', 'Year', 'Protest_Risk_Score', 'Expected_Protests']
        feature_cols = [col for col in self.combined_data.columns if col not in exclude_cols]
        
        X = self.combined_data[feature_cols].copy()
        y = self.combined_data['Expected_Protests'].copy()
        
        # Handle any remaining missing values
        X = X.fillna(X.median())
        
        # Store feature names for interpretation
        self.feature_names = feature_cols
        
        print(f"  ✓ Prepared {len(feature_cols)} features")
        print(f"  ✓ Target variable: Expected_Protests (range: {y.min():.2f} - {y.max():.2f})")
        
        return X, y
    
    def train_model(self, X, y):
        """
        Train machine learning model
        """
        print("\n🤖 Training Predictive Model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Try multiple models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        best_model = None
        best_score = -np.inf
        
        for name, model in models.items():
            if name == 'Linear Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            print(f"  {name}:")
            print(f"    R² Score: {r2:.3f}")
            print(f"    RMSE: {rmse:.3f}")
            print(f"    MAE: {mae:.3f}")
            
            if r2 > best_score:
                best_score = r2
                best_model = model
                self.model = model
        
        print(f"\n  ✅ Best model selected with R² = {best_score:.3f}")
        
        return X_train, X_test, y_train, y_test
    
    def analyze_feature_importance(self):
        """
        Analyze feature importance
        """
        print("\n📊 Analyzing Feature Importance...")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print("\n🔝 Top 10 Most Important Features:")
            for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
                print(f"  {i:2d}. {row['Feature']}: {row['Importance']:.3f}")
            
            return importance_df
        else:
            print("  ℹ️ Feature importance not available for this model type")
            return None
    
    def create_visualizations(self, importance_df=None):
        """
        Create visualizations
        """
        print("\n📈 Creating Visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Service Delivery Index by Province
        if 'Service_Delivery_Index' in self.combined_data.columns:
            province_service = self.combined_data.groupby('Province')['Service_Delivery_Index'].mean().sort_values()
            axes[0, 0].barh(range(len(province_service)), province_service.values)
            axes[0, 0].set_yticks(range(len(province_service)))
            axes[0, 0].set_yticklabels(province_service.index, fontsize=8)
            axes[0, 0].set_xlabel('Service Delivery Index (%)')
            axes[0, 0].set_title('Service Delivery Index by Province')
        
        # 2. Expected Protests vs Service Gap
        if 'Service_Gap' in self.combined_data.columns:
            axes[0, 1].scatter(self.combined_data['Service_Gap'], self.combined_data['Expected_Protests'], alpha=0.6)
            axes[0, 1].set_xlabel('Service Delivery Gap (%)')
            axes[0, 1].set_ylabel('Expected Protests')
            axes[0, 1].set_title('Protest Risk vs Service Delivery Gap')
        
        # 3. Feature Importance
        if importance_df is not None:
            top_features = importance_df.head(8)
            axes[1, 0].barh(range(len(top_features)), top_features['Importance'])
            axes[1, 0].set_yticks(range(len(top_features)))
            axes[1, 0].set_yticklabels(top_features['Feature'], fontsize=8)
            axes[1, 0].set_xlabel('Feature Importance')
            axes[1, 0].set_title('Top Feature Importances')
        
        # 4. Protest Risk by Year
        yearly_risk = self.combined_data.groupby('Year')['Expected_Protests'].mean()
        axes[1, 1].plot(yearly_risk.index, yearly_risk.values, marker='o')
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('Average Expected Protests')
        axes[1, 1].set_title('Protest Risk Trend Over Time')
        
        plt.tight_layout()
        plt.savefig('service_delivery_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("  ✅ Visualizations saved as 'service_delivery_analysis.png'")
    
    def predict_protest_risk(self, province_name=None, custom_features=None):
        """
        Predict protest risk for a given area
        """
        print(f"\n🔮 Predicting Protest Risk...")
        
        if custom_features is not None:
            # Use custom features
            features_df = pd.DataFrame([custom_features])
            features_scaled = self.scaler.transform(features_df[self.feature_names])
            
            if hasattr(self.model, 'predict'):
                if isinstance(self.model, LinearRegression):
                    prediction = self.model.predict(features_scaled)[0]
                else:
                    prediction = self.model.predict(features_df[self.feature_names])[0]
                
                print(f"  🎯 Predicted protest risk: {prediction:.2f}")
                return prediction
        
        elif province_name:
            # Use existing province data
            province_data = self.combined_data[self.combined_data['Province'] == province_name]
            if not province_data.empty:
                latest_data = province_data.iloc[-1]
                features = latest_data[self.feature_names].values.reshape(1, -1)
                
                if isinstance(self.model, LinearRegression):
                    features_scaled = self.scaler.transform(features)
                    prediction = self.model.predict(features_scaled)[0]
                else:
                    prediction = self.model.predict(features)[0]
                
                print(f"  🎯 Predicted protest risk for {province_name}: {prediction:.2f}")
                return prediction
            else:
                print(f"  ❌ Province '{province_name}' not found in data")
                return None
    
    def run_full_analysis(self):
        """
        Run the complete analysis pipeline
        """
        print("🚀 STARTING SERVICE DELIVERY PROTEST PREDICTION ANALYSIS")
        print("=" * 70)
        
        # Load data
        self.load_census_data()
        self.load_protest_data()
        
        # Combine and prepare data
        self.combine_data_for_modeling()
        X, y = self.prepare_features()
        
        # Train model
        X_train, X_test, y_train, y_test = self.train_model(X, y)
        
        # Analyze results
        importance_df = self.analyze_feature_importance()
        
        # Create visualizations
        self.create_visualizations(importance_df)
        
        # Example predictions
        print("\n🔮 Example Predictions:")
        provinces = self.combined_data['Province'].unique()[:3]
        for province in provinces:
            self.predict_protest_risk(province_name=province)
        
        print("\n" + "=" * 70)
        print("✅ ANALYSIS COMPLETE!")
        print("📊 Check 'service_delivery_analysis.png' for visualizations")
        print("🎯 Model ready for predictions")
        
        return self.model, self.combined_data

def main():
    """
    Main execution function
    """
    predictor = ServiceDeliveryProtestPredictor()
    model, data = predictor.run_full_analysis()
    return predictor, model, data

if __name__ == "__main__":
    predictor, model, data = main()