import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def explore_events_fatalities():
    """
    Explore Events&Fatalities.xlsx for protest data
    """
    print("=" * 60)
    print("EVENTS & FATALITIES DATA EXPLORATION")
    print("=" * 60)
    
    events_file = "Datasets/Events&Fatalaties.xlsx"
    
    try:
        # Get all sheet names first
        excel_file = pd.ExcelFile(events_file)
        sheet_names = excel_file.sheet_names
        
        print(f"\n📊 Found {len(sheet_names)} sheets in Events & Fatalities data:")
        for i, sheet in enumerate(sheet_names, 1):
            print(f"  {i}. {sheet}")
        
        # Explore each sheet
        events_data = {}
        for sheet_name in sheet_names:
            print(f"\n{'='*50}")
            print(f"EXPLORING SHEET: {sheet_name}")
            print(f"{'='*50}")
            
            try:
                df = pd.read_excel(events_file, sheet_name=sheet_name)
                events_data[sheet_name] = df
                
                print(f"\n📋 Sheet Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
                
                # Display column names
                print(f"\n📝 Column Names ({len(df.columns)} total):")
                for i, col in enumerate(df.columns, 1):
                    print(f"  {i:2d}. {col}")
                
                # Display first few rows
                print(f"\n🔍 First 3 rows:")
                print(df.head(3).to_string())
                
                # Look for service delivery related content
                text_columns = df.select_dtypes(include=['object']).columns
                service_keywords = ['service', 'delivery', 'water', 'electricity', 'housing', 
                                  'sanitation', 'municipal', 'council', 'infrastructure',
                                  'protest', 'demonstration', 'riot']
                
                for col in text_columns:
                    if df[col].dtype == 'object':
                        # Check for service delivery related terms
                        for keyword in service_keywords:
                            matches = df[col].str.contains(keyword, case=False, na=False)
                            if matches.any():
                                count = matches.sum()
                                print(f"\n🎯 Found {count} events with '{keyword}' in '{col}'")
                                # Show some examples
                                examples = df[matches][col].head(3).tolist()
                                for example in examples:
                                    print(f"   - {str(example)[:100]}...")
                
            except Exception as e:
                print(f"❌ Error reading sheet '{sheet_name}': {str(e)}")
                continue
        
        return events_data
        
    except Exception as e:
        print(f"❌ Error loading Events & Fatalities data: {str(e)}")
        return None

def explore_political_violence():
    """
    Explore political violence events data
    """
    print("\n" + "=" * 60)
    print("POLITICAL VIOLENCE EVENTS DATA EXPLORATION")
    print("=" * 60)
    
    violence_file = "Datasets/additional - south-africa_political_violence_events_and_fatalities_by_month-year_as-of-13aug2025.xlsx"
    
    try:
        # Load political violence data
        df_violence = pd.read_excel(violence_file)
        
        print(f"\n📊 Political Violence Data Dimensions: {df_violence.shape[0]} rows × {df_violence.shape[1]} columns")
        
        # Display column names
        print(f"\n📝 Column Names ({len(df_violence.columns)} total):")
        for i, col in enumerate(df_violence.columns, 1):
            print(f"  {i:2d}. {col}")
        
        # Display first few rows
        print(f"\n🔍 First 5 rows:")
        print(df_violence.head().to_string())
        
        # Look for service delivery related events
        print(f"\n🔍 Searching for service delivery related events...")
        
        text_columns = df_violence.select_dtypes(include=['object']).columns
        service_keywords = ['service', 'delivery', 'water', 'electricity', 'housing', 
                          'sanitation', 'municipal', 'council', 'infrastructure']
        
        for col in text_columns:
            if df_violence[col].dtype == 'object':
                print(f"\n📋 Sample values in '{col}':")
                unique_vals = df_violence[col].dropna().unique()[:5]
                for val in unique_vals:
                    print(f"  - {str(val)[:100]}")
                
                # Search for service delivery related terms
                for keyword in service_keywords:
                    matches = df_violence[col].str.contains(keyword, case=False, na=False)
                    if matches.any():
                        count = matches.sum()
                        print(f"\n🎯 Found {count} events with '{keyword}' in '{col}'")
        
        return df_violence
        
    except Exception as e:
        print(f"❌ Error loading political violence data: {str(e)}")
        return None

def explore_demonstration_events():
    """
    Re-explore demonstration events with different approach
    """
    print("\n" + "=" * 60)
    print("DEMONSTRATION EVENTS DATA RE-EXPLORATION")
    print("=" * 60)
    
    demo_file = "Datasets/south-africa_demonstration_events_by_month-year_as-of-13aug2025.xlsx"
    
    try:
        # Try different sheet names or parameters
        excel_file = pd.ExcelFile(demo_file)
        sheet_names = excel_file.sheet_names
        
        print(f"\n📊 Found {len(sheet_names)} sheets in Demonstration data:")
        for i, sheet in enumerate(sheet_names, 1):
            print(f"  {i}. {sheet}")
        
        # Try loading each sheet
        for sheet_name in sheet_names:
            print(f"\n{'='*40}")
            print(f"SHEET: {sheet_name}")
            print(f"{'='*40}")
            
            try:
                df = pd.read_excel(demo_file, sheet_name=sheet_name)
                print(f"Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
                
                if df.shape[0] > 0 and df.shape[1] > 1:
                    print(f"\nColumns: {list(df.columns)}")
                    print(f"\nFirst few rows:")
                    print(df.head().to_string())
                    return df
                    
            except Exception as e:
                print(f"Error reading sheet: {e}")
        
        # If no good sheets found, try reading without sheet specification
        print("\n🔄 Trying to read file without sheet specification...")
        df = pd.read_excel(demo_file)
        print(f"Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading demonstration data: {str(e)}")
        return None

def main():
    """
    Main exploration function for protest data
    """
    print("🚀 Starting Protest Data Exploration")
    print("📍 Focus: Finding service delivery protest events")
    
    # Explore Events & Fatalities
    events_data = explore_events_fatalities()
    
    # Explore Political Violence
    violence_data = explore_political_violence()
    
    # Re-explore Demonstration Events
    demo_data = explore_demonstration_events()
    
    print("\n" + "=" * 60)
    print("PROTEST DATA EXPLORATION SUMMARY")
    print("=" * 60)
    
    datasets_found = []
    
    if events_data:
        print(f"\n✅ Events & Fatalities data loaded")
        for sheet_name, df in events_data.items():
            print(f"   - {sheet_name}: {df.shape[0]} rows × {df.shape[1]} columns")
            datasets_found.append(('Events & Fatalities', sheet_name, df))
    
    if violence_data is not None:
        print(f"\n✅ Political Violence data loaded")
        print(f"   - {violence_data.shape[0]} events × {violence_data.shape[1]} variables")
        datasets_found.append(('Political Violence', 'main', violence_data))
    
    if demo_data is not None:
        print(f"\n✅ Demonstration data loaded")
        print(f"   - {demo_data.shape[0]} events × {demo_data.shape[1]} variables")
        datasets_found.append(('Demonstrations', 'main', demo_data))
    
    print(f"\n📊 Total datasets found: {len(datasets_found)}")
    
    return datasets_found

if __name__ == "__main__":
    protest_datasets = main()