# 🏛️ Service Delivery Protest Prediction

A machine learning project that predicts service delivery protest risk in South Africa using Census 2022 data and historical protest events.

## 📋 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Analysis**
   ```bash
   python service_delivery_protest_predictor.py
   ```

3. **Launch Dashboard**
   ```bash
   streamlit run streamlit_app.py
   ```

## 📁 Project Structure

- `service_delivery_protest_predictor.py` - Main analysis engine
- `streamlit_app.py` - Interactive dashboard
- `Service_Delivery_Protest_Analysis.ipynb` - Jupyter notebook analysis
- `PRESENTATION_GUIDE.md` - **Comprehensive presentation documentation**
- `Datasets/` - Data files (Census 2022, protest events, crime stats)

## 🎯 Key Features

- **84% Accuracy** - Random Forest model for protest risk prediction
- **Interactive Dashboard** - Streamlit-based web application
- **Provincial Analysis** - Risk assessment across all 9 provinces
- **Policy Recommendations** - Actionable insights for governance

## 👥 Team Data Dynamo 2025

- **Noxolo** - Team Leader
- **Omphile** - Problem Definition Expert
- **Onke** - Data Expert
- **April** - Technical Lead
- **Rakgadi** - Results Analyst
- **Mosa** - Demo & Conclusions

## 📖 Documentation

- **[Presentation Guide](PRESENTATION_GUIDE.md)** - Complete presentation documentation with slide-by-slide breakdown, delivery guidelines, and technical specifications
- **[Jupyter Notebook](Service_Delivery_Protest_Analysis.ipynb)** - Detailed analysis workflow
- **[Requirements](requirements.txt)** - Python dependencies

## 🚀 Usage

### Individual Scripts
```bash
# Explore census data
python explore_census_data.py

# Analyze protest patterns
python explore_protest_data.py

# Evaluate model performance
python model_evaluation.py

# Generate presentation
python create_presentation.py
```

### Dashboard Features
- Real-time risk assessment
- Provincial drill-down analysis
- Service delivery indicator tracking
- Historical trend visualization
- Export capabilities

## 📊 Key Findings

- **Water access** is the strongest predictor of protest risk (23.4% importance)
- **Population density** amplifies service delivery impacts
- **Eastern Cape, Limpopo, KwaZulu-Natal** show highest risk levels
- **84% accuracy** in predicting protest risk across provinces

## 🔧 Technical Details

- **Algorithm**: Random Forest Classifier
- **Features**: 15+ service delivery indicators
- **Data**: Census 2022 + protest events (2020-2024)
- **Validation**: 5-fold cross-validation
- **Performance**: 84% accuracy, 0.91 ROC-AUC

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 84.2% |
| Precision | 84.7% |
| Recall | 84.1% |
| F1-Score | 84.4% |
| ROC-AUC | 0.91 |

## 🎤 Presentation

For detailed presentation guidelines, speaker notes, and delivery instructions, see **[PRESENTATION_GUIDE.md](PRESENTATION_GUIDE.md)**.

## 📧 Contact

Team Data Dynamo 2025
- Email: smmakola@wsu.ac.za 
- GitHub: https://github.com/15kay/data_dynamo_2025