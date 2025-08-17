import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def explore_census_data():
    """
    Explore the Census 2022 data to understand structure and variables
    """
    print("=" * 60)
    print("CENSUS 2022 DATA EXPLORATION")
    print("=" * 60)
    
    # Load Census 2022 data
    census_file = "Datasets/Census 2022_Themes_24-10-2023.xlsx"
    
    try:
        # Get all sheet names first
        excel_file = pd.ExcelFile(census_file)
        sheet_names = excel_file.sheet_names
        
        print(f"\n📊 Found {len(sheet_names)} sheets in Census 2022 data:")
        for i, sheet in enumerate(sheet_names, 1):
            print(f"  {i}. {sheet}")
        
        # Explore each sheet
        census_data = {}
        for sheet_name in sheet_names:
            print(f"\n{'='*50}")
            print(f"EXPLORING SHEET: {sheet_name}")
            print(f"{'='*50}")
            
            try:
                df = pd.read_excel(census_file, sheet_name=sheet_name)
                census_data[sheet_name] = df
                
                print(f"\n📋 Sheet Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
                
                # Display column names
                print(f"\n📝 Column Names ({len(df.columns)} total):")
                for i, col in enumerate(df.columns, 1):
                    print(f"  {i:2d}. {col}")
                
                # Display first few rows
                print(f"\n🔍 First 3 rows:")
                print(df.head(3).to_string())
                
                # Basic statistics for numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    print(f"\n📊 Numeric columns summary ({len(numeric_cols)} columns):")
                    print(df[numeric_cols].describe())
                
                # Check for missing values
                missing_data = df.isnull().sum()
                if missing_data.sum() > 0:
                    print(f"\n⚠️  Missing values:")
                    missing_cols = missing_data[missing_data > 0]
                    for col, count in missing_cols.items():
                        print(f"  {col}: {count} missing ({count/len(df)*100:.1f}%)")
                else:
                    print(f"\n✅ No missing values found")
                
                # Data types
                print(f"\n🏷️  Data types:")
                dtype_counts = df.dtypes.value_counts()
                for dtype, count in dtype_counts.items():
                    print(f"  {dtype}: {count} columns")
                    
            except Exception as e:
                print(f"❌ Error reading sheet '{sheet_name}': {str(e)}")
                continue
        
        return census_data
        
    except Exception as e:
        print(f"❌ Error loading Census data: {str(e)}")
        return None

def explore_demonstration_data():
    """
    Explore demonstration events data to understand service delivery protests
    """
    print("\n" + "=" * 60)
    print("DEMONSTRATION EVENTS DATA EXPLORATION")
    print("=" * 60)
    
    demo_file = "Datasets/south-africa_demonstration_events_by_month-year_as-of-13aug2025.xlsx"
    
    try:
        # Load demonstration data
        df_demo = pd.read_excel(demo_file)
        
        print(f"\n📊 Demonstration Data Dimensions: {df_demo.shape[0]} rows × {df_demo.shape[1]} columns")
        
        # Display column names
        print(f"\n📝 Column Names ({len(df_demo.columns)} total):")
        for i, col in enumerate(df_demo.columns, 1):
            print(f"  {i:2d}. {col}")
        
        # Display first few rows
        print(f"\n🔍 First 5 rows:")
        print(df_demo.head().to_string())
        
        # Look for service delivery related protests
        print(f"\n🔍 Searching for service delivery related events...")
        
        # Check if there are columns that might indicate event type or description
        text_columns = df_demo.select_dtypes(include=['object']).columns
        service_delivery_keywords = ['service', 'delivery', 'water', 'electricity', 'housing', 
                                   'sanitation', 'municipal', 'council', 'infrastructure']
        
        for col in text_columns:
            print(f"\n📋 Unique values in '{col}' (first 10):")
            unique_vals = df_demo[col].dropna().unique()[:10]
            for val in unique_vals:
                print(f"  - {val}")
            
            # Search for service delivery related terms
            if df_demo[col].dtype == 'object':
                service_related = df_demo[col].str.contains('|'.join(service_delivery_keywords), 
                                                         case=False, na=False)
                if service_related.any():
                    count = service_related.sum()
                    print(f"\n🎯 Found {count} potential service delivery events in '{col}'")
        
        return df_demo
        
    except Exception as e:
        print(f"❌ Error loading demonstration data: {str(e)}")
        return None

def main():
    """
    Main exploration function
    """
    print("🚀 Starting Data Exploration for Service Delivery Protest Prediction")
    print("📍 Focus: Census 2022 demographics + Demonstration events")
    
    # Explore Census data
    census_data = explore_census_data()
    
    # Explore demonstration data
    demo_data = explore_demonstration_data()
    
    print("\n" + "=" * 60)
    print("EXPLORATION SUMMARY")
    print("=" * 60)
    
    if census_data:
        print(f"\n✅ Census 2022 data loaded successfully")
        print(f"   - {len(census_data)} sheets available")
        for sheet_name, df in census_data.items():
            print(f"   - {sheet_name}: {df.shape[0]} rows × {df.shape[1]} columns")
    
    if demo_data is not None:
        print(f"\n✅ Demonstration data loaded successfully")
        print(f"   - {demo_data.shape[0]} events × {demo_data.shape[1]} variables")
    
    print("\n🎯 Next Steps:")
    print("   1. Identify key demographic indicators from Census data")
    print("   2. Filter demonstration events for service delivery protests")
    print("   3. Create geographic/temporal linkages between datasets")
    print("   4. Build predictive features")
    print("   5. Train machine learning model")
    
    return census_data, demo_data

if __name__ == "__main__":
    census_data, demo_data = main()