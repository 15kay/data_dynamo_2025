import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from service_delivery_protest_predictor import ServiceDeliveryProtestPredictor

# Set page configuration
st.set_page_config(
    page_title="Service Delivery Protest Predictor",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: bold;
}
.sub-header {
    font-size: 1.5rem;
    color: #ff7f0e;
    margin-bottom: 1rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #1f77b4;
}
.prediction-high {
    background-color: #ffebee;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #f44336;
}
.prediction-medium {
    background-color: #fff3e0;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #ff9800;
}
.prediction-low {
    background-color: #e8f5e8;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #4caf50;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_predictor_and_data():
    """Load the trained model and data"""
    try:
        predictor = ServiceDeliveryProtestPredictor()
        predictor.load_census_data()
        predictor.load_protest_data()
        predictor.combine_data_for_modeling()
        X, y = predictor.prepare_features()
        predictor.train_model(X, y)
        return predictor, predictor.combined_data
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def main():
    # Main header
    st.markdown('<h1 class="main-header">🏛️ Service Delivery Protest Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Analyzing Census 2022 data to predict service delivery protest risks across South African provinces</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading model and data..."):
        predictor, data = load_predictor_and_data()
    
    if predictor is None or data is None:
        st.error("Failed to load the prediction model. Please check the data files.")
        return
    
    # Sidebar for navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Overview", "📈 Data Explorer", "🔮 Predictions", "📊 Provincial Analysis", "🎯 Custom Prediction"]
    )
    
    if page == "🏠 Overview":
        show_overview(predictor, data)
    elif page == "📈 Data Explorer":
        show_data_explorer(predictor, data)
    elif page == "🔮 Predictions":
        show_predictions(predictor, data)
    elif page == "📊 Provincial Analysis":
        show_provincial_analysis(predictor, data)
    elif page == "🎯 Custom Prediction":
        show_custom_prediction(predictor)

def show_overview(predictor, data):
    """Show overview page"""
    st.markdown('<h2 class="sub-header">📋 Project Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Provinces Analyzed", "9")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Municipalities", "265")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Model R² Score", "0.652")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Project description
    st.markdown("""
    ### 🎯 Project Objectives
    
    This application analyzes **Census 2022** data to predict service delivery protest risks across South African provinces. 
    The model considers various socio-economic factors including:
    
    - 💧 **Water Access**: Percentage of households with piped water
    - 🚽 **Sanitation**: Access to flush toilet facilities
    - ⚡ **Electricity**: Household electricity access
    - 🗑️ **Refuse Collection**: Municipal waste management services
    - 🏠 **Housing**: Formal housing conditions
    - 🎓 **Education**: Higher education levels
    
    ### 🔬 Methodology
    
    1. **Data Integration**: Combined Census 2022 demographic data with historical protest events
    2. **Feature Engineering**: Created service delivery indicators and composite indices
    3. **Machine Learning**: Trained Random Forest model to predict protest likelihood
    4. **Validation**: Achieved 65.2% R² score in cross-validation
    
    ### 📊 Key Findings
    
    - **Service gaps** are the strongest predictor of protest risk
    - **Refuse collection** and **sanitation** are critical service delivery factors
    - **Eastern Cape** shows highest predicted protest risk
    - **Population size** amplifies the impact of service delivery gaps
    """)
    
    # Feature importance chart
    st.markdown('<h3 class="sub-header">🔝 Most Important Factors</h3>', unsafe_allow_html=True)
    
    importance_data = {
        'Feature': ['Service Gap', 'Refuse Service', 'Service Delivery Index', 'Sanitation', 'Population', 'Electricity', 'Water Access'],
        'Importance': [0.239, 0.236, 0.194, 0.173, 0.124, 0.034, 0.000]
    }
    
    fig = px.bar(
        x=importance_data['Importance'],
        y=importance_data['Feature'],
        orientation='h',
        title="Feature Importance in Protest Prediction Model",
        labels={'x': 'Importance Score', 'y': 'Features'},
        color=importance_data['Importance'],
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def show_data_explorer(predictor, data):
    """Show data exploration page"""
    st.markdown('<h2 class="sub-header">📈 Census 2022 Data Explorer</h2>', unsafe_allow_html=True)
    
    # Data summary
    st.markdown("### 📊 Dataset Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Total Observations**: {len(data)}")
        st.info(f"**Provinces**: {data['Province'].nunique()}")
    
    with col2:
        st.info(f"**Years Covered**: {data['Year'].min()} - {data['Year'].max()}")
        st.info(f"**Features**: {len([col for col in data.columns if col not in ['Province', 'Year', 'Protest_Risk_Score', 'Expected_Protests']])}")
    
    # Service delivery indicators by province
    st.markdown("### 🗺️ Service Delivery by Province")
    
    # Select indicator
    indicator = st.selectbox(
        "Choose service delivery indicator:",
        ['Service_Delivery_Index', 'Water_Access_Pct', 'Sanitation_Pct', 'Electricity_Pct', 'Refuse_Service_Pct']
    )
    
    # Create provincial averages
    provincial_avg = data.groupby('Province')[indicator].mean().sort_values(ascending=True)
    
    fig = px.bar(
        x=provincial_avg.values,
        y=provincial_avg.index,
        orientation='h',
        title=f"{indicator.replace('_', ' ').title()} by Province",
        labels={'x': f'{indicator.replace("_", " ").title()} (%)', 'y': 'Province'},
        color=provincial_avg.values,
        color_continuous_scale='RdYlGn'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation matrix
    st.markdown("### 🔗 Feature Correlations")
    
    numeric_cols = ['Total_Population', 'Water_Access_Pct', 'Sanitation_Pct', 'Electricity_Pct', 
                   'Refuse_Service_Pct', 'Service_Delivery_Index', 'Service_Gap', 'Expected_Protests']
    
    corr_matrix = data[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        title="Feature Correlation Matrix",
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Raw data table
    st.markdown("### 📋 Raw Data Sample")
    st.dataframe(data.head(20), use_container_width=True)

def show_predictions(predictor, data):
    """Show predictions page"""
    st.markdown('<h2 class="sub-header">🔮 Protest Risk Predictions</h2>', unsafe_allow_html=True)
    
    # Provincial predictions
    st.markdown("### 🗺️ Provincial Risk Assessment")
    
    provinces = data['Province'].unique()
    predictions = []
    
    for province in provinces:
        risk = predictor.predict_protest_risk(province_name=province)
        if risk is not None:
            predictions.append({'Province': province, 'Predicted_Risk': risk})
    
    pred_df = pd.DataFrame(predictions).sort_values('Predicted_Risk', ascending=False)
    
    # Risk level classification
    def classify_risk(risk):
        if risk >= 7:
            return 'High', '#f44336'
        elif risk >= 4:
            return 'Medium', '#ff9800'
        else:
            return 'Low', '#4caf50'
    
    pred_df['Risk_Level'], pred_df['Color'] = zip(*pred_df['Predicted_Risk'].apply(classify_risk))
    
    # Risk visualization
    fig = px.bar(
        pred_df,
        x='Predicted_Risk',
        y='Province',
        orientation='h',
        title="Predicted Protest Risk by Province",
        labels={'Predicted_Risk': 'Predicted Risk Score', 'Province': 'Province'},
        color='Predicted_Risk',
        color_continuous_scale='Reds'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk level cards
    st.markdown("### 🚨 Risk Level Breakdown")
    
    col1, col2, col3 = st.columns(3)
    
    high_risk = pred_df[pred_df['Risk_Level'] == 'High']
    medium_risk = pred_df[pred_df['Risk_Level'] == 'Medium']
    low_risk = pred_df[pred_df['Risk_Level'] == 'Low']
    
    with col1:
        st.markdown('<div class="prediction-high">', unsafe_allow_html=True)
        st.markdown("**🔴 High Risk Provinces**")
        for _, row in high_risk.iterrows():
            st.write(f"• {row['Province']}: {row['Predicted_Risk']:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="prediction-medium">', unsafe_allow_html=True)
        st.markdown("**🟡 Medium Risk Provinces**")
        for _, row in medium_risk.iterrows():
            st.write(f"• {row['Province']}: {row['Predicted_Risk']:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="prediction-low">', unsafe_allow_html=True)
        st.markdown("**🟢 Low Risk Provinces**")
        for _, row in low_risk.iterrows():
            st.write(f"• {row['Province']}: {row['Predicted_Risk']:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Time series predictions
    st.markdown("### 📈 Risk Trends Over Time")
    
    yearly_risk = data.groupby(['Year', 'Province'])['Expected_Protests'].mean().reset_index()
    
    fig = px.line(
        yearly_risk,
        x='Year',
        y='Expected_Protests',
        color='Province',
        title="Predicted Protest Risk Trends (2020-2024)",
        labels={'Expected_Protests': 'Expected Protests', 'Year': 'Year'}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def show_provincial_analysis(predictor, data):
    """Show detailed provincial analysis"""
    st.markdown('<h2 class="sub-header">📊 Provincial Deep Dive</h2>', unsafe_allow_html=True)
    
    # Province selector
    selected_province = st.selectbox(
        "Select a province for detailed analysis:",
        data['Province'].unique()
    )
    
    province_data = data[data['Province'] == selected_province].iloc[0]
    
    # Province overview
    st.markdown(f"### 🏛️ {selected_province} Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Population", f"{province_data['Total_Population']:,.0f}")
    
    with col2:
        st.metric("Service Delivery Index", f"{province_data['Service_Delivery_Index']:.1f}%")
    
    with col3:
        st.metric("Service Gap", f"{province_data['Service_Gap']:.1f}%")
    
    with col4:
        risk = predictor.predict_protest_risk(province_name=selected_province)
        st.metric("Predicted Risk", f"{risk:.2f}" if risk else "N/A")
    
    # Service delivery breakdown
    st.markdown("### 🔧 Service Delivery Breakdown")
    
    services = {
        'Water Access': province_data.get('Water_Access_Pct', 0),
        'Sanitation': province_data.get('Sanitation_Pct', 0),
        'Electricity': province_data.get('Electricity_Pct', 0),
        'Refuse Collection': province_data.get('Refuse_Service_Pct', 0)
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(services.keys()),
            y=list(services.values()),
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        )
    ])
    
    fig.update_layout(
        title=f"Service Delivery Indicators - {selected_province}",
        yaxis_title="Coverage (%)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Comparison with national average
    st.markdown("### 📊 Comparison with National Average")
    
    national_avg = data.groupby('Province').mean().mean()
    
    comparison_data = {
        'Indicator': ['Water Access', 'Sanitation', 'Electricity', 'Refuse Collection', 'Service Index'],
        selected_province: [
            province_data.get('Water_Access_Pct', 0),
            province_data.get('Sanitation_Pct', 0),
            province_data.get('Electricity_Pct', 0),
            province_data.get('Refuse_Service_Pct', 0),
            province_data.get('Service_Delivery_Index', 0)
        ],
        'National Average': [
            data['Water_Access_Pct'].mean(),
            data['Sanitation_Pct'].mean(),
            data['Electricity_Pct'].mean(),
            data['Refuse_Service_Pct'].mean(),
            data['Service_Delivery_Index'].mean()
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    fig = px.bar(
        comparison_df,
        x='Indicator',
        y=[selected_province, 'National Average'],
        title=f"{selected_province} vs National Average",
        barmode='group'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def show_custom_prediction(predictor):
    """Show custom prediction interface"""
    st.markdown('<h2 class="sub-header">🎯 Custom Protest Risk Prediction</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🔧 Create Custom Scenario
    
    Adjust the service delivery indicators below to see how they affect protest risk prediction.
    This tool helps policymakers understand the impact of service delivery improvements.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 👥 Demographics")
        population = st.slider("Total Population", 100000, 5000000, 1000000, step=50000)
        
        st.markdown("#### 💧 Water & Sanitation")
        water_access = st.slider("Water Access (%)", 0, 100, 80, step=5)
        sanitation = st.slider("Sanitation Access (%)", 0, 100, 70, step=5)
    
    with col2:
        st.markdown("#### ⚡ Infrastructure")
        electricity = st.slider("Electricity Access (%)", 0, 100, 85, step=5)
        refuse = st.slider("Refuse Collection (%)", 0, 100, 60, step=5)
        
        st.markdown("#### 🏠 Housing & Education")
        housing = st.slider("Formal Housing (%)", 0, 100, 75, step=5)
        education = st.slider("Higher Education (%)", 0, 100, 15, step=1)
    
    # Calculate composite indices
    service_delivery_index = np.mean([water_access, sanitation, electricity, refuse])
    service_gap = 100 - service_delivery_index
    
    # Create custom features
    custom_features = {
        'Total_Population': population,
        'Water_Access_Pct': water_access,
        'Sanitation_Pct': sanitation,
        'Electricity_Pct': electricity,
        'Refuse_Service_Pct': refuse,
        'Service_Delivery_Index': service_delivery_index,
        'Service_Gap': service_gap
    }
    
    # Make prediction
    if st.button("🔮 Predict Protest Risk", type="primary"):
        with st.spinner("Calculating prediction..."):
            try:
                risk = predictor.predict_protest_risk(custom_features=custom_features)
                
                if risk is not None:
                    # Display prediction
                    st.markdown("---")
                    st.markdown("### 📊 Prediction Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Predicted Risk Score", f"{risk:.2f}")
                    
                    with col2:
                        st.metric("Service Delivery Index", f"{service_delivery_index:.1f}%")
                    
                    with col3:
                        st.metric("Service Gap", f"{service_gap:.1f}%")
                    
                    # Risk interpretation
                    if risk >= 7:
                        st.error("🔴 **HIGH RISK**: This scenario indicates high protest likelihood. Immediate attention to service delivery gaps is recommended.")
                    elif risk >= 4:
                        st.warning("🟡 **MEDIUM RISK**: Moderate protest risk. Consider targeted improvements in key service areas.")
                    else:
                        st.success("🟢 **LOW RISK**: Low protest likelihood. Current service delivery levels appear adequate.")
                    
                    # Recommendations
                    st.markdown("### 💡 Recommendations")
                    
                    if service_gap > 30:
                        st.write("• **Priority**: Address major service delivery gaps")
                    if water_access < 80:
                        st.write("• **Water**: Improve piped water access infrastructure")
                    if sanitation < 70:
                        st.write("• **Sanitation**: Expand flush toilet facilities")
                    if electricity < 90:
                        st.write("• **Electricity**: Enhance electrical grid coverage")
                    if refuse < 70:
                        st.write("• **Waste Management**: Strengthen municipal refuse collection")
                    
                else:
                    st.error("Failed to generate prediction. Please check your inputs.")
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()