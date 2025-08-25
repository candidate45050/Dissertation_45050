# Cancer Drug Price Comparison Dashboard 💊

A comprehensive Streamlit-based dashboard for comparing cancer drug prices between Medicare Part D and Cost Plus Drugs, designed for dissertation research on pharmaceutical pricing analysis.

## 🎯 Overview

This interactive dashboard provides detailed analysis and visualization of price differences between Medicare Part D prescription drug prices and Cost Plus Drugs pricing. The tool helps identify potential cost savings opportunities and analyzes spending patterns across different cancer medications.

## Online Streamlit Dashboard
https://dissertation45050.streamlit.app/

## ✨ Key Features

### 📊 Data Analysis
- **Comprehensive Drug Matching**: Advanced algorithms match drugs between Medicare and Cost Plus datasets using multiple criteria
- **Price Comparison**: Side-by-side comparison of unit prices and total spending
- **Savings Analysis**: Calculate potential savings with both percentage and dollar amount metrics
- **Trend Analysis**: Medicare price trends from 2019-2023 including CAGR and year-over-year changes

### 📈 Interactive Visualizations
- **Total Spending Comparison**: Bar charts showing Medicare vs Cost Plus total spending
- **Unit Price Analysis**: Per-dose price comparisons across medications
- **Savings Opportunities**: Ranked charts showing highest savings potential
- **Logarithmic Scaling**: Toggle between linear and logarithmic scales for better visualization

### 🎛️ User Interface
- **Drug Selection**: Multi-select dropdown to focus analysis on specific medications
- **Tabbed Interface**: Separate views for total cost and unit price analysis
- **Expandable Details**: Detailed breakdowns available on demand
- **Data Export**: Download analysis results as CSV files

## 📋 Prerequisites

### Required Data Files
Place these files in the project root directory:

1. **`DSD_PTD_RY25_P04_V10_DY23_BGM.csv`**
   - Medicare Part D Drug Spending Dashboard data
   - Contains spending and pricing information for 2023

2. **`price_calculator_data.xlsx`**
   - Cost Plus Drugs pricing data
   - Should include columns: drug_name, generic_for, strength, quantity, price

### Python Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- `streamlit` - Web application framework
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `plotly` - Interactive visualizations
- `openpyxl` - Excel file reading

## 🚀 Installation & Setup

1. **Clone or download the project files**
   ```bash
   git clone <repository-url>
   cd cancer-drug-dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install streamlit pandas numpy plotly openpyxl
   ```

3. **Prepare data files**
   - Place `DSD_PTD_RY25_P04_V10_DY23_BGM.csv` in the project root
   - Place `price_calculator_data.xlsx` in the project root

4. **Run the dashboard**
   ```bash
   streamlit run cancer_drug_dashboard.py
   ```

5. **Access the application**
   - Open your browser to `http://localhost:8501`

## 📖 Usage Guide

### Getting Started
1. Launch the dashboard using the command above
2. Wait for data loading (progress indicator will show)
3. Review the "Data Sources & Methodology" section for context

### Navigation
- **Drug Selection**: Use the multiselect dropdown to focus on specific drugs
- **Total Cost Tab**: Analyze overall spending and savings by total expenditure
- **Unit Price Tab**: Compare per-dose costs and examine price trends
- **Detailed Breakdowns**: Expand sections for comprehensive data tables

### Key Metrics
- **Total Medicare Spending**: Aggregate spending for matched drugs (2025 adjusted)
- **Cost Plus Alternative**: Potential spending using Cost Plus pricing
- **Savings Analysis**: Both percentage and dollar amount savings calculations
- **Price Trends**: Historical Medicare price changes and growth rates

## 🔬 Methodology

### Data Processing
1. **Medicare Data**: 
   - Filters for "Overall" manufacturer data
   - Adjusts 2023 prices to mid-2025 using inflation multiplier (1.0586)
   - Focuses on drugs where brand name matches generic name

2. **Cost Plus Data**:
   - Calculates unit prices including $5 service fee and 7.52% markup
   - Extracts numeric values from strength and quantity fields

3. **Drug Matching Algorithm**:
   - **Exact matches**: Direct name comparisons
   - **Substring matching**: One name contained in another
   - **Word-level matching**: Jaccard similarity for compound names
   - **Cross-matching**: Drug names vs generic names
   - **Fuzzy matching**: String similarity as fallback

### Savings Calculations
- **Conservative Approach**: Uses maximum Cost Plus prices for minimum savings estimates
- **Comprehensive Analysis**: Provides both optimistic (min Cost Plus) and conservative (max Cost Plus) scenarios
- **Inflation Adjustment**: Medicare prices adjusted from 2023 to mid-2025

## 📁 File Structure

```
project/
├── cancer_drug_dashboard.py     # Main dashboard application
├── DSD_PTD_RY25_P04_V10_DY23_BGM.csv  # Medicare Part D data
├── price_calculator_data.xlsx   # Cost Plus pricing data
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## 🎨 Dashboard Components

### Header Section
- Professional styling with gradient backgrounds
- Key metrics summary cards
- Data source information panel

### Analysis Tabs
1. **Total Cost Comparison**
   - Aggregate spending analysis
   - Savings opportunities by total dollar amount
   - Logarithmic scaling options

2. **Unit Dosage Cost**
   - Per-pill/dose price comparisons
   - Medicare price trend analysis
   - Detailed formulation breakdowns

### Interactive Features
- Drug filtering and selection
- Chart scaling options (linear/logarithmic)
- Expandable detailed data tables
- CSV data export functionality

## ⚠️ Important Notes

### Data Limitations
- **Matching Accuracy**: Automated drug matching may have false positives/negatives
- **Price Variations**: Cost Plus prices vary by formulation and quantity
- **Temporal Differences**: Medicare data from 2023, Cost Plus data may be more recent

### Interpretation Guidelines
- **Conservative Estimates**: Savings calculations use maximum Cost Plus prices
- **Negative Values**: Some drugs may show Cost Plus as more expensive
- **Context Required**: Results should be interpreted within broader healthcare policy context

## 🛠️ Technical Details

### Performance Optimizations
- `@st.cache_data` decorators for data loading functions
- Efficient pandas operations for large datasets
- Streamlit's built-in caching for improved response times

### Visualization Libraries
- **Plotly**: Interactive charts with hover information
- **Streamlit**: Native components for tables and metrics
- **Custom CSS**: Professional styling and responsive design

## 🔧 Customization

### Modifying Matching Threshold
```python
matching_threshold = 0.6  # Line 414 - adjust as needed
```

### Inflation Adjustment
```python
medicare_df['medicare_unit_price'] = medicare_df['medicare_unit_price'] * 1.0586  # Line 81
```

### Cost Plus Markup
```python
costplus_df['unit_price'] = (costplus_df['price'] + 5) * 1.0752 / costplus_df['quantity_numeric']  # Line 152
```

## 📊 Data Sources

1. **Medicare Part D Drug Spending Dashboard**
   - Official CMS data for 2023
   - Weighted average spending per dosage unit
   - Brand and generic name information

2. **Cost Plus Drugs**
   - Transparent pricing from Cost Plus company
   - Includes service fees and markup calculations
   - Various formulations and quantities

## 🎓 Academic Use

This dashboard was developed for dissertation research on pharmaceutical pricing. It provides:
- Empirical analysis of drug pricing differences
- Quantitative savings potential estimates  
- Visual evidence for academic presentations
- Exportable data for further statistical analysis

## 📞 Support

For technical issues or questions about the dashboard:
1. Check that all required data files are present
2. Verify Python package versions match requirements
3. Review error messages in the Streamlit interface
4. Ensure data file formats match expected schemas

---


**Disclaimer**: This tool is for research and educational purposes. Results should be verified and interpreted within appropriate healthcare policy and economic contexts.

