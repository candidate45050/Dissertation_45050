import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from difflib import SequenceMatcher
import re
from typing import Dict, List, Tuple

# Page configuration
st.set_page_config(
    page_title="Cancer Drug Price Comparison Dashboard",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f4e79;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff 0%, #e6f3ff 100%);
        border-radius: 10px;
        border: 2px solid #1f4e79;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .savings-positive {
        color: #28a745;
        font-weight: bold;
    }
    
    .savings-negative {
        color: #dc3545;
        font-weight: bold;
    }
    
    .data-source {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_medicare_data():
    """Load and process Medicare Part D data"""
    try:
        df = pd.read_csv('DSD_PTD_RY25_P04_V10_DY23_BGM.csv')
        # Filter for Overall manufacturer data
        medicare_df = df[df['Mftr_Name'] == 'Overall'].copy()
        
        # Clean and prepare data
        medicare_df = medicare_df[['Brnd_Name', 'Gnrc_Name', 'Avg_Spnd_Per_Dsg_Unt_Wghtd_2023', 'Tot_Dsg_Unts_2023', 'Tot_Spndng_2023', 'Chg_Avg_Spnd_Per_Dsg_Unt_22_23', 'CAGR_Avg_Spnd_Per_Dsg_Unt_19_23']].copy()
        medicare_df.columns = ['brand_name', 'generic_name', 'medicare_unit_price', 'total_dosage_units', 'total_spending_2023', 'price_change_22_23', 'cagr_19_23']
        
        # Remove rows with missing price data or dosage unit data
        medicare_df = medicare_df.dropna(subset=['medicare_unit_price', 'total_dosage_units'])
        medicare_df = medicare_df[medicare_df['medicare_unit_price'] > 0]
        medicare_df = medicare_df[medicare_df['total_dosage_units'] > 0]
        
        # Adjust Medicare prices for inflation from 2023 to mid 2025
        medicare_df['medicare_unit_price'] = medicare_df['medicare_unit_price'] * 1.0586
        # Also adjust total spending for inflation where available
        medicare_df['total_spending_2025'] = medicare_df['total_spending_2023'] * 1.0586
        
        # Additional filter: Brand name equals or contains first part of generic name
        def brand_matches_generic(row):
            if pd.isna(row['brand_name']) or pd.isna(row['generic_name']):
                return False
            
            brand_clean = clean_drug_name(row['brand_name'])
            generic_clean = clean_drug_name(row['generic_name'])
            
            if not brand_clean or not generic_clean:
                return False
            
            # Check if brand name equals generic name
            if brand_clean == generic_clean:
                return True
            
            # Check if brand name contains first word of generic name (and vice versa)
            brand_words = brand_clean.split()
            generic_words = generic_clean.split()
            
            if brand_words and generic_words:
                first_brand_word = brand_words[0]
                first_generic_word = generic_words[0]
                
                # Skip very short words
                if len(first_brand_word) < 4 or len(first_generic_word) < 4:
                    return False
                
                # Check if first words match or one contains the other
                if (first_brand_word == first_generic_word or 
                    first_brand_word in first_generic_word or 
                    first_generic_word in first_brand_word):
                    return True
                
                # Check if full brand name is in generic name or vice versa
                if brand_clean in generic_clean or generic_clean in brand_clean:
                    return True
                    
            return False
        
        # Apply the filtering
        original_count = len(medicare_df)
        medicare_df = medicare_df[medicare_df.apply(brand_matches_generic, axis=1)].copy()
        filtered_count = len(medicare_df)
        
        print(f"Medicare data filtered: {original_count} → {filtered_count} drugs (kept drugs where brand matches generic)")
        
        # Clean brand names for better matching
        medicare_df['brand_name_clean'] = medicare_df['brand_name'].str.upper().str.strip()
        
        return medicare_df
    except Exception as e:
        st.error(f"Error loading Medicare data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_costplus_data():
    """Load and process Cost Plus data"""
    try:
        df = pd.read_excel('price_calculator_data.xlsx')
        
        # Clean and prepare data
        costplus_df = df.copy()
        costplus_df['drug_name_clean'] = costplus_df['drug_name'].str.upper().str.strip()
        
        # Extract numeric values from strength and quantity
        costplus_df['strength_numeric'] = costplus_df['strength'].str.extract(r'(\d+(?:\.\d+)?)').astype(float)
        costplus_df['quantity_numeric'] = costplus_df['quantity'].str.extract(r'(\d+)').astype(int)
        
        # Calculate unit price
        costplus_df['unit_price'] = (costplus_df['price'] + 5) * 1.0752 / costplus_df['quantity_numeric']
        
        return costplus_df
    except Exception as e:
        st.error(f"Error loading Cost Plus data: {e}")
        return pd.DataFrame()

def clean_drug_name(name: str) -> str:
    """Clean drug name for better matching"""
    if pd.isna(name):
        return ""
    
    # Convert to uppercase and strip
    clean = str(name).upper().strip()
    
    # Remove common suffixes and characters that might interfere
    clean = clean.replace('*', '').replace('®', '').replace('™', '')
    clean = clean.replace(' HCL', ' HCl').replace(' HCLO', ' HCl')  # Standardize HCl
    
    # Remove extra spaces
    clean = ' '.join(clean.split())
    
    return clean

def calculate_match_score(costplus_name: str, medicare_name: str) -> tuple:
    """Calculate comprehensive match score between drug names"""
    cp_clean = clean_drug_name(costplus_name)
    med_clean = clean_drug_name(medicare_name)
    
    if not cp_clean or not med_clean:
        return 0.0, "NO_NAME"
    
    # Exact match (highest priority)
    if cp_clean == med_clean:
        return 1.0, "EXACT"
    
    # Substring matches (high priority)
    if cp_clean in med_clean:
        return 0.95, "CP_IN_MEDICARE"
    if med_clean in cp_clean:
        return 0.9, "MEDICARE_IN_CP"
    
    # Word-by-word matching for compound names
    cp_words = set(cp_clean.split())
    med_words = set(med_clean.split())
    
    # Remove common short words that aren't meaningful
    common_words = {'THE', 'AND', 'OR', 'OF', 'FOR', 'WITH', 'IN', 'ON', 'AT', 'BY'}
    cp_words = {w for w in cp_words if len(w) > 2 and w not in common_words}
    med_words = {w for w in med_words if len(w) > 2 and w not in common_words}
    
    if cp_words and med_words:
        # Calculate Jaccard similarity for word sets
        intersection = cp_words.intersection(med_words)
        union = cp_words.union(med_words)
        
        if union:
            jaccard_score = len(intersection) / len(union)
            if jaccard_score >= 0.8:
                return jaccard_score, "HIGH_WORD_MATCH"
            elif jaccard_score >= 0.5:
                return jaccard_score, "MEDIUM_WORD_MATCH"
    
    # Check if main drug name (first word) matches
    cp_first = cp_clean.split()[0] if cp_clean.split() else ""
    med_first = med_clean.split()[0] if med_clean.split() else ""
    
    if cp_first and med_first and len(cp_first) > 3:
        if cp_first == med_first:
            return 0.7, "FIRST_WORD_MATCH"
        elif cp_first in med_first or med_first in cp_first:
            return 0.6, "FIRST_WORD_PARTIAL"
    
    # Fuzzy string matching as fallback
    fuzzy_score = SequenceMatcher(None, cp_clean, med_clean).ratio()
    if fuzzy_score >= 0.8:
        return fuzzy_score, "HIGH_FUZZY"
    elif fuzzy_score >= 0.6:
        return fuzzy_score, "MEDIUM_FUZZY"
    
    return fuzzy_score, "LOW_FUZZY"

def find_best_matches(medicare_df: pd.DataFrame, costplus_df: pd.DataFrame, threshold: float = 0.6) -> pd.DataFrame:
    """Find best matches starting from Cost Plus drugs and searching in Medicare"""
    matches = []
    
    # Get unique Cost Plus drugs (our primary source)
    unique_costplus_drugs = costplus_df['drug_name'].unique()
    
    print(f"Searching for matches for {len(unique_costplus_drugs)} Cost Plus drugs...")
    
    for cp_drug in unique_costplus_drugs:
        # Get Cost Plus drug info
        cp_rows = costplus_df[costplus_df['drug_name'] == cp_drug]
        cp_generic = cp_rows['generic_for'].iloc[0] if not cp_rows['generic_for'].isna().all() else None
        
        best_match = None
        best_score = 0
        best_match_type = ""
        
        # Search through all Medicare drugs
        for _, medicare_row in medicare_df.iterrows():
            medicare_brand = medicare_row['brand_name']
            medicare_generic = medicare_row['generic_name']
            
            # Try matching Cost Plus drug_name with Medicare brand_name
            score1, match_type1 = calculate_match_score(cp_drug, medicare_brand)
            
            # Try matching Cost Plus drug_name with Medicare generic_name
            score2, match_type2 = calculate_match_score(cp_drug, medicare_generic)
            
            # Try matching Cost Plus generic_for with Medicare brand_name
            score3, match_type3 = 0, ""
            if cp_generic:
                score3, match_type3 = calculate_match_score(cp_generic, medicare_brand)
            
            # Try matching Cost Plus generic_for with Medicare generic_name  
            score4, match_type4 = 0, ""
            if cp_generic:
                score4, match_type4 = calculate_match_score(cp_generic, medicare_generic)
            
            # Find the best score among all attempts
            scores = [(score1, match_type1, 'CP_drug->MED_brand'), 
                     (score2, match_type2, 'CP_drug->MED_generic'),
                     (score3, match_type3, 'CP_generic->MED_brand'), 
                     (score4, match_type4, 'CP_generic->MED_generic')]
            
            current_best_score, current_match_type, match_path = max(scores, key=lambda x: x[0])
            
            if current_best_score > best_score and current_best_score >= threshold:
                best_score = current_best_score
                best_match = medicare_row
                best_match_type = f"{current_match_type}({match_path})"
        
        if best_match is not None:
            matches.append({
                'costplus_drug': cp_drug,
                'costplus_generic': cp_generic,
                'medicare_brand': best_match['brand_name'],
                'medicare_generic': best_match['generic_name'],
                'medicare_unit_price': best_match['medicare_unit_price'],
                'total_dosage_units': best_match['total_dosage_units'],
                'total_spending_2025': best_match['total_spending_2025'],
                'price_change_22_23': best_match['price_change_22_23'],
                'cagr_19_23': best_match['cagr_19_23'],
                'match_score': best_score,
                'match_type': best_match_type
            })
            print(f"✓ Matched: '{cp_drug}' -> '{best_match['brand_name']}' (score: {best_score:.3f}, type: {best_match_type})")
        else:
            print(f"✗ No match found for: '{cp_drug}'")
    
    return pd.DataFrame(matches)

def calculate_costplus_price_range(costplus_df: pd.DataFrame, drug_name: str) -> Dict:
    """Calculate min and max unit prices for a Cost Plus drug"""
    drug_data = costplus_df[costplus_df['drug_name_clean'] == drug_name.upper().strip()]
    
    if drug_data.empty:
        return {'min_price': None, 'max_price': None, 'min_details': None, 'max_details': None}
    
    # Simplified approach: find actual min and max unit prices
    min_price_idx = drug_data['unit_price'].idxmin()
    max_price_idx = drug_data['unit_price'].idxmax()
    
    min_cost_row = drug_data.loc[min_price_idx]
    max_cost_row = drug_data.loc[max_price_idx]
    
    return {
        'min_price': min_cost_row['unit_price'],
        'max_price': max_cost_row['unit_price'],
        'min_details': f"{min_cost_row['strength']} - {min_cost_row['quantity']}",
        'max_details': f"{max_cost_row['strength']} - {max_cost_row['quantity']}"
    }

def create_comparison_dataframe(matches_df: pd.DataFrame, costplus_df: pd.DataFrame) -> pd.DataFrame:
    """Create comprehensive comparison dataframe"""
    comparison_data = []
    
    for _, row in matches_df.iterrows():
        price_range = calculate_costplus_price_range(costplus_df, row['costplus_drug'])
        
        if price_range['min_price'] is not None and price_range['max_price'] is not None:
            medicare_price = row['medicare_unit_price']
            costplus_min = price_range['min_price']
            costplus_max = price_range['max_price']
            total_units = row['total_dosage_units']
            
            # Calculate total spending
            calculated_medicare_total_spending = medicare_price * total_units
            
            # Compare with actual total spending from Medicare data (if available)
            actual_medicare_total_spending = row.get('total_spending_2025')
            used_actual_spending = False
            
            if pd.notna(actual_medicare_total_spending) and actual_medicare_total_spending < calculated_medicare_total_spending:
                # Calculate the spending reduction percentage
                spending_reduction_pct = ((calculated_medicare_total_spending - actual_medicare_total_spending) / calculated_medicare_total_spending) * 100
                
                # Only use actual spending if reduction is greater than 1%
                if spending_reduction_pct > 1.0:
                    medicare_total_spending = actual_medicare_total_spending
                    used_actual_spending = True
                else:
                    medicare_total_spending = calculated_medicare_total_spending
            else:
                medicare_total_spending = calculated_medicare_total_spending
            
            costplus_min_total_spending = costplus_min * total_units
            costplus_max_total_spending = costplus_max * total_units
            
            # Calculate potential savings using total spending amounts
            # This ensures asterisk drugs use the substituted Medicare total spending for percentage calculations
            savings_vs_min = ((medicare_total_spending - costplus_max_total_spending) / medicare_total_spending) * 100 if medicare_total_spending > 0 else 0
            savings_vs_max = ((medicare_total_spending - costplus_min_total_spending) / medicare_total_spending) * 100 if medicare_total_spending > 0 else 0
            
            # Add asterisk to drug name if actual spending was used
            medicare_generic_display = row['medicare_generic'] if pd.notna(row['medicare_generic']) else 'N/A'
            if used_actual_spending:
                medicare_generic_display += '*'
            
            # Calculate percentage difference between calculated and actual spending (for asterisk drugs)
            spending_difference_pct = "None"
            if used_actual_spending:
                spending_difference_pct = ((calculated_medicare_total_spending - actual_medicare_total_spending) / calculated_medicare_total_spending) * 100
            
            comparison_data.append({
                'Medicare Generic Name': medicare_generic_display,
                'Cost Plus Drug': row['costplus_drug'],
                'Cost Plus Generic For': row['costplus_generic'] if pd.notna(row['costplus_generic']) else 'N/A',
                'Medicare Unit Price (2025$)': medicare_price,
                'Cost Plus Min Price ($)': costplus_min,
                'Cost Plus Max Price ($)': costplus_max,
                'Total Dosage Units': total_units,
                'Medicare Total Spending (2025$)': medicare_total_spending,
                'Cost Plus Min Total Spending ($)': costplus_min_total_spending,
                'Cost Plus Max Total Spending ($)': costplus_max_total_spending,
                'Spending Reduction (%)': spending_difference_pct,
                'Min Price Details': price_range['min_details'],
                'Max Price Details': price_range['max_details'],
                'Price Change 2022-2023 (%)': row['price_change_22_23'] if pd.notna(row['price_change_22_23']) else 'N/A',
                'CAGR 2019-2023 (%)': row['cagr_19_23'] if pd.notna(row['cagr_19_23']) else 'N/A',
                'Potential Savings (Min) %': savings_vs_min,
                'Potential Savings (Max) %': savings_vs_max
            })
    
    return pd.DataFrame(comparison_data)

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<div class="main-header">💊 Cancer Drug Price Comparison Dashboard<br><small>Medicare Part D (2023 adjusted for inflation to mid-2025) vs Cost Plus Drugs</small></div>', unsafe_allow_html=True)
    
    # Data sources info
    with st.expander("📊 Data Sources & Methodology", expanded=False):
        st.markdown("""
        <div class="data-source">
        <h4>Data Sources:</h4>
        <ul>
            <li><strong>Medicare Part D:</strong> Drug Spending Dashboard (DSD_PTD_RY25_P04_V10_DY23_BGM.csv)</li>
            <li><strong>Cost Plus Drugs:</strong> Price calculator data (price_calculator_data.xlsx)</li>
        </ul>
        
        <h4>Methodology:</h4>
        <ul>
            <li><strong>Medicare Data:</strong> Using Avg_Spnd_Per_Dsg_Unt_Wghtd_2023 for Overall manufacturers</li>
            <li><strong>Medicare Price Adjustment:</strong> 2023 Medicare prices and total spending are adjusted for inflation to mid-2025 using multiplier of 1.0586</li>
            <li><strong>Medicare Total Cost Logic:</strong> <strong>Uses actual Tot_Spndng_2023 only when it is lower than calculated cost AND the reduction is greater than 1%</strong> to provide more accurate cost comparisons. <strong>Drugs marked with asterisk (*) use actual Medicare spending data with >1% reduction from calculated values.</strong></li>
            <li><strong>Medicare Filtering:</strong> Only include drugs where Brnd_Name equals or contains first part of Gnrc_Name (focuses on generic drugs)</li>
            <li><strong>Cost Plus Comparison:</strong> <strong>All savings calculations use the maximum Cost Plus price</strong> across all available formulations to provide conservative (minimum) savings estimates</li>
            <li><strong>Negative Values:</strong> Charts may show negative savings where Cost Plus prices exceed Medicare prices</li>
            <li><strong>Matching Strategy:</strong> Start with Cost Plus drugs, search Medicare using multiple algorithms:</li>
            <ul>
                <li>Exact name matches</li>
                <li>Substring matches (drug name contains the other)</li>
                <li>Word-level matching for compound drug names</li>
                <li>Cross-matching drug names with generic names</li>
                <li>Fuzzy string matching as fallback</li>
            </ul>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        medicare_df = load_medicare_data()
        costplus_df = load_costplus_data()
    
    if medicare_df.empty or costplus_df.empty:
        st.error("Failed to load data. Please check that the data files exist and are accessible.")
        return
    
    # Use fixed matching threshold
    matching_threshold = 0.6
    
    # Find matches
    with st.spinner("Matching drugs between datasets..."):
        matches_df = find_best_matches(medicare_df, costplus_df, matching_threshold)
        comparison_df = create_comparison_dataframe(matches_df, costplus_df)
    
    if comparison_df.empty:
        st.warning("No drug matches found. Try lowering the matching threshold.")
        return
    
    # Drug selection (common for both tabs)
    st.header("💰 Drug Analysis")
    
    all_drugs = comparison_df['Cost Plus Drug'].unique().tolist()
    selected_drugs = st.multiselect(
        "Select drugs to analyze (leave empty to show all):",
        options=all_drugs,
        default=[],
        help="Choose specific drugs to focus your analysis on, or leave empty to see all matched drugs"
    )
    
    # Filter data based on selection
    if selected_drugs:
        filtered_df = comparison_df[comparison_df['Cost Plus Drug'].isin(selected_drugs)].copy()
    else:
        filtered_df = comparison_df.copy()
    
    # Sort data by potential savings (descending) - using Max Cost Plus for conservative estimates
    filtered_df = filtered_df.sort_values('Potential Savings (Min) %', ascending=True)
    
    # Show filtering info
    if selected_drugs:
        st.info(f"Showing {len(filtered_df)} selected drugs out of {len(comparison_df)} total matches")
    else:
        st.info(f"Showing all {len(filtered_df)} matched drugs")
    
    # Key metrics (after filtered_df is defined)
    st.header("📈 Key Insights")
    
    st.info("📊 **Note:** Medicare Part D prices (2023) have been adjusted for inflation to mid-2025 using a multiplier of 1.0586 for accurate comparison. **All savings calculations use the maximum Cost Plus price to show minimum (conservative) savings estimates**. Values may be negative where Cost Plus is more expensive. **Drugs marked with asterisk (*) use actual Medicare spending data when the reduction from calculated cost exceeds 1%.**")
    
    # Add key metrics summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_medicare_spending = round(filtered_df['Medicare Total Spending (2025$)'].sum() / 1_000_000, 2)
        st.metric(
            "Total Medicare Spending",
            f"${total_medicare_spending:.2f}M",
            help="Total Medicare spending for all matched drugs (2025$)"
        )
    
    with col2:
        total_costplus_spending = round(filtered_df['Cost Plus Max Total Spending ($)'].sum() / 1_000_000, 2)
        st.metric(
            "Total Cost Plus Spending (Max)",
            f"${total_costplus_spending:.2f}M",
            help="Total spending if using Cost Plus max prices (conservative estimate)"
        )
    
    with col3:
        total_potential_savings = round(total_medicare_spending - total_costplus_spending, 2)
        savings_percentage = round((total_potential_savings / total_medicare_spending) * 100, 2) if total_medicare_spending > 0 else 0
        st.metric(
            "Potential Total Savings",
            f"${total_potential_savings:.2f}M",
            f"{savings_percentage:.2f}%",
            help="Minimum potential savings using Cost Plus max prices"
        )
    
    with col4:
        avg_savings_percentage = round(filtered_df['Potential Savings (Min) %'].mean(), 2)
        st.metric(
            "Average Drug Savings",
            f"{avg_savings_percentage:.2f}%",
            help="Average percentage savings per drug (conservative estimate)"
        )
    
    # Create tabs
    tab1, tab2 = st.tabs(["📊 Total Cost Comparison", "💊 Unit Dosage Cost (Price per Pill)"])
    
    # Sort data alphabetically by Medicare Generic Name for charts (preserving asterisk)
    chart_df = filtered_df.sort_values('Medicare Generic Name')
    
    # Color-code the savings columns function
    def color_savings(val):
        if pd.isna(val) or val == 'N/A':
            return ''  # No color for missing values
        try:
            if float(val) > 0:
                return 'color: #28a745; font-weight: bold'  # Green for savings
            else:
                return 'color: #dc3545; font-weight: bold'  # Red for higher costs
        except (ValueError, TypeError):
            return ''  # No color for non-numeric values
    
    # Color-code spending reduction column function
    def color_spending_reduction(val):
        if pd.isna(val) or val == "None":
            return ''  # No color for None/NaN values (drugs without asterisk)
        try:
            if float(val) > 0:
                return 'color: #007bff; font-weight: bold'  # Blue for spending reduction
            else:
                return ''
        except (ValueError, TypeError):
            return ''
    
    # Color-code price change and CAGR columns function  
    def color_price_changes(val):
        if pd.isna(val) or val == 'N/A':
            return ''  # No color for N/A values
        try:
            if float(val) > 0:
                return 'color: #dc3545; font-weight: bold'  # Red for price increases
            else:
                return 'color: #28a745; font-weight: bold'  # Green for price decreases
        except (ValueError, TypeError):
            return ''  # No color for non-numeric values
    
    with tab1:
        st.subheader("Total Spending Analysis")
        
        # Create total spending focused dataframe with millions conversion
        total_spending_df = filtered_df[['Medicare Generic Name', 'Cost Plus Generic For',
                                       'Total Dosage Units', 'Medicare Total Spending (2025$)', 
                                       'Cost Plus Min Total Spending ($)', 'Cost Plus Max Total Spending ($)',
                                       'Spending Reduction (%)', 'Potential Savings (Min) %', 'Potential Savings (Max) %']].copy()
        
        # Convert spending columns to millions
        total_spending_df['Medicare Total Spending (Millions)'] = (total_spending_df['Medicare Total Spending (2025$)'] / 1_000_000).round(2)
        total_spending_df['Cost Plus Min Total Spending (Millions)'] = (total_spending_df['Cost Plus Min Total Spending ($)'] / 1_000_000).round(2)
        total_spending_df['Cost Plus Max Total Spending (Millions)'] = (total_spending_df['Cost Plus Max Total Spending ($)'] / 1_000_000).round(2)
        
        # Add total dollar savings columns
        # Min savings using maximum Cost Plus spending (conservative estimate)
        total_spending_df['Min Total Dollar Savings (Millions)'] = (total_spending_df['Medicare Total Spending (Millions)'] - total_spending_df['Cost Plus Max Total Spending (Millions)']).round(2)
        # Max savings using minimum Cost Plus spending (optimistic estimate)
        total_spending_df['Max Total Dollar Savings (Millions)'] = (total_spending_df['Medicare Total Spending (Millions)'] - total_spending_df['Cost Plus Min Total Spending (Millions)']).round(2)
        
        # Create summary table with key metrics only
        st.write("**📊 Summary Table - Key Metrics**")
        st.info("💡 **Note:** Drugs marked with asterisk (*) use actual Medicare spending data from Tot_Spndng_2023 instead of calculated values (unit_price × total_units) when the spending reduction exceeds 1%. **Min Savings** uses maximum Cost Plus prices (conservative estimates), **Max Savings** uses minimum Cost Plus prices (optimistic estimates). The 'Spending Reduction (%)' column shows the percentage reduction from calculated to actual spending - only displayed for asterisk drugs, shows 'None' for all other drugs.")
        
        summary_df = total_spending_df[['Medicare Generic Name', 'Medicare Total Spending (Millions)', 
                                      'Cost Plus Min Total Spending (Millions)', 'Cost Plus Max Total Spending (Millions)', 
                                      'Min Total Dollar Savings (Millions)', 'Max Total Dollar Savings (Millions)', 'Spending Reduction (%)', 
                                      'Potential Savings (Min) %', 'Potential Savings (Max) %']].copy()
        
        summary_df.columns = ['Drug Name', 'Medicare Total ($M)', 'Cost Plus Min ($M)', 'Cost Plus Max ($M)', 
                             'Min Savings ($M)', 'Max Savings ($M)', 'Spending Reduction (%)', 'Min Savings (%)', 'Max Savings (%)']
        
        # Custom formatter for spending reduction column
        def format_spending_reduction(val):
            if val == "None" or pd.isna(val):
                return "None"
            try:
                formatted = f"{float(val):.2f}".rstrip('0').rstrip('.')
                return f"{formatted}%"
            except (ValueError, TypeError):
                return str(val)
        
        # Custom formatter for dollar amounts to remove trailing zeros
        def format_dollars(val):
            try:
                formatted = f"{float(val):.2f}".rstrip('0').rstrip('.')
                return f"${formatted}M"
            except (ValueError, TypeError):
                return str(val)
        
        # Custom formatter for percentages to remove trailing zeros
        def format_percentage(val):
            try:
                formatted = f"{float(val):.2f}".rstrip('0').rstrip('.')
                return f"{formatted}%"
            except (ValueError, TypeError):
                return str(val)
        
        styled_summary_df = summary_df.style.format({
            'Medicare Total ($M)': format_dollars,
            'Cost Plus Min ($M)': format_dollars,
            'Cost Plus Max ($M)': format_dollars,
            'Min Savings ($M)': format_dollars,
            'Max Savings ($M)': format_dollars,
            'Min Savings (%)': format_percentage,
            'Max Savings (%)': format_percentage,
            'Spending Reduction (%)': format_spending_reduction
        }).map(color_savings, subset=['Min Savings ($M)', 'Max Savings ($M)', 'Min Savings (%)', 'Max Savings (%)']).map(color_spending_reduction, subset=['Spending Reduction (%)'])
        
        st.dataframe(styled_summary_df, use_container_width=True, hide_index=True)
        
        # Detailed breakdown in expandable section
        with st.expander("📋 Detailed Breakdown - Full Data", expanded=False):
            # Split into two smaller tables
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Spending Amounts**")
                spending_detail_df = total_spending_df[['Medicare Generic Name', 'Total Dosage Units',
                                                      'Medicare Total Spending (Millions)', 
                                                      'Cost Plus Min Total Spending (Millions)', 
                                                      'Cost Plus Max Total Spending (Millions)']].copy()
                
                spending_detail_df.columns = ['Drug Name', 'Total Units', 'Medicare ($M)', 'CP Min ($M)', 'CP Max ($M)']
                
                # Custom formatter for units to remove trailing zeros
                def format_units(val):
                    try:
                        formatted = f"{float(val):,.2f}".rstrip('0').rstrip('.')
                        return formatted
                    except (ValueError, TypeError):
                        return str(val)
                
                styled_spending_detail = spending_detail_df.style.format({
                    'Total Units': format_units,
                    'Medicare ($M)': format_dollars,
                    'CP Min ($M)': format_dollars,
                    'CP Max ($M)': format_dollars
                })
                
                st.dataframe(styled_spending_detail, use_container_width=True, hide_index=True)
            
            with col2:
                st.write("**Savings Analysis**")
                savings_detail_df = total_spending_df[['Medicare Generic Name', 'Cost Plus Generic For',
                                                     'Min Total Dollar Savings (Millions)', 'Max Total Dollar Savings (Millions)', 'Spending Reduction (%)',
                                                     'Potential Savings (Min) %', 'Potential Savings (Max) %']].copy()
                
                savings_detail_df.columns = ['Drug Name', 'CP Generic Name', 'Min Savings ($M)', 'Max Savings ($M)', 'Spending Reduction (%)', 'Min Savings (%)', 'Max Savings (%)']
                
                styled_savings_detail = savings_detail_df.style.format({
                    'Min Savings ($M)': format_dollars,
                    'Max Savings ($M)': format_dollars,
                    'Min Savings (%)': format_percentage,
                    'Max Savings (%)': format_percentage,
                    'Spending Reduction (%)': format_spending_reduction
                }).map(color_savings, subset=['Min Savings ($M)', 'Max Savings ($M)', 'Min Savings (%)', 'Max Savings (%)']).map(color_spending_reduction, subset=['Spending Reduction (%)'])
                
                st.dataframe(styled_savings_detail, use_container_width=True, hide_index=True)
        
        st.subheader("Total Spending Visualization")
        
        # Y-axis scale toggle for total spending
        use_log_scale_total = st.checkbox(
            "Use logarithmic scale for Total Spending (y-axis)",
            value=True,
            help="Logarithmic scale helps visualize large spending differences more clearly",
            key="total_log_scale"
        )
        
        # Total spending comparison chart
        fig_spending = go.Figure()
        
        # Convert to millions for display
        chart_df_millions = chart_df.copy()
        chart_df_millions['Medicare Total Spending (Millions)'] = (chart_df['Medicare Total Spending (2025$)'] / 1_000_000).round(2)
        chart_df_millions['Cost Plus Min Total Spending (Millions)'] = (chart_df['Cost Plus Min Total Spending ($)'] / 1_000_000).round(2)
        chart_df_millions['Cost Plus Max Total Spending (Millions)'] = (chart_df['Cost Plus Max Total Spending ($)'] / 1_000_000).round(2)
        
        # Medicare total spending
        fig_spending.add_trace(go.Bar(
            x=chart_df_millions['Medicare Generic Name'],
            y=chart_df_millions['Medicare Total Spending (Millions)'],
            name='Medicare Part D Total (2025$)',
            marker=dict(color='red', opacity=0.7),
            hovertemplate='<b>%{x}</b><br>Medicare Total: $%{y:.0f}M<extra></extra>'
        ))
        
        # Cost Plus total spending ranges
        fig_spending.add_trace(go.Bar(
            x=chart_df_millions['Medicare Generic Name'],
            y=chart_df_millions['Cost Plus Min Total Spending (Millions)'],
            name='Cost Plus Total (Min)',
            marker=dict(color='blue', opacity=0.7),
            hovertemplate='<b>%{x}</b><br>Cost Plus Min Total: $%{y:.0f}M<extra></extra>'
        ))
        
        fig_spending.add_trace(go.Bar(
            x=chart_df_millions['Medicare Generic Name'],
            y=chart_df_millions['Cost Plus Max Total Spending (Millions)'],
            name='Cost Plus Total (Max)',
            marker=dict(color='lightblue', opacity=0.7),
            hovertemplate='<b>%{x}</b><br>Cost Plus Max Total: $%{y:.0f}M<extra></extra>'
        ))
        
        fig_spending.update_layout(
            title=dict(
                text="Drug Total Spending Comparison: Medicare Part D vs Cost Plus",
                font=dict(size=20, family="Arial Black")
            ),
            xaxis_title="Medicare Generic Name",
            yaxis_title="Total Spending ($ Millions)",
            yaxis_type="log" if use_log_scale_total else "linear",
            height=500,
            barmode='group',
            hovermode='x unified',
            xaxis=dict(
                tickangle=45,
                title=dict(font=dict(size=16)),
                tickfont=dict(size=16)
            ),
            yaxis=dict(
                title=dict(font=dict(size=16)),
                tickfont=dict(size=16),
                tickformat='.0f',
                ticksuffix="M"
            ),
            legend=dict(
                font=dict(size=14)
            ),
            font=dict(size=14)
        )
        
        st.plotly_chart(fig_spending, use_container_width=True, key="total_spending_main")
        
        # Top savings opportunities for total spending
        st.subheader("Total Spending Savings Opportunities")
        # Sort by potential savings (min) which uses maximum Cost Plus prices - show max savings at top
        top_savings_total = filtered_df.sort_values('Potential Savings (Min) %', ascending=True)
        
        # Calculate total dollar savings using maximum Cost Plus spending (minimum savings)
        top_savings_total = top_savings_total.copy()
        top_savings_total['Total_Dollar_Savings'] = top_savings_total['Medicare Total Spending (2025$)'] - top_savings_total['Cost Plus Max Total Spending ($)']
        top_savings_total['Total_Dollar_Savings_Millions'] = (top_savings_total['Total_Dollar_Savings'] / 1_000_000).round(2)
        
        # Create two columns for side-by-side charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Add spacing to align with col2's toggle
            st.write("")  # Empty space to align charts
            st.markdown('<div style="height: 1.5rem;"></div>', unsafe_allow_html=True)  # Custom spacing for alignment
            
            # Percentage savings chart
            fig_total_pct = px.bar(
                top_savings_total,
                x='Potential Savings (Min) %',
                y='Cost Plus Drug',
                orientation='h',
                title="Savings by Percentage (%) - Max Cost Plus (Min Savings)",
                color='Potential Savings (Min) %',
                color_continuous_scale='RdYlGn'  # Standard scale: min=red, max=green
            )
            
            fig_total_pct.update_layout(
                height=600,
                title=dict(font=dict(size=18, family="Arial Black")),
                xaxis=dict(
                    title=dict(text="Savings (%) vs Max Cost Plus (Min Savings)", font=dict(size=16)),
                    tickfont=dict(size=14),
                    zeroline=True,
                    zerolinecolor='black',
                    zerolinewidth=2
                ),
                yaxis=dict(
                    title=dict(text="", font=dict(size=16)),
                    tickfont=dict(size=14)
                ),
                font=dict(size=14),
                coloraxis_colorbar=dict(
                    tickfont=dict(size=14),
                    tickformat='.0f'
                )
            )
            st.plotly_chart(fig_total_pct, use_container_width=True, key="total_pct_savings")
        
        with col2:
            # X-axis scale toggle for total dollar savings
            use_log_scale_total_dollar = st.checkbox(
                "Use logarithmic scale for Savings by Dollar Amount (x-axis)",
                value=True,
                help="Logarithmic scale helps visualize large savings differences more clearly",
                key="total_dollar_log_scale"
            )
            
            # Dollar savings chart
            fig_total_dollar = px.bar(
                top_savings_total,
                x='Total_Dollar_Savings_Millions',
                y='Cost Plus Drug',
                orientation='h',
                title="Savings by Dollar Amount ($ Millions) - Max Cost Plus (Min Savings)",
                color='Total_Dollar_Savings_Millions',
                color_continuous_scale=[[0, '#90EE90'], [1, '#006400']]  # Light green to dark green
            )
            
            fig_total_dollar.update_layout(
                height=600,
                title=dict(font=dict(size=18, family="Arial Black")),
                xaxis=dict(
                    title=dict(text="Total Dollar Savings ($ Millions) vs Max Cost Plus", font=dict(size=16)),
                    tickfont=dict(size=14),
                    tickformat='.0f',
                    ticksuffix="M",
                    zeroline=True,
                    zerolinecolor='black',
                    zerolinewidth=2,
                    type="log" if use_log_scale_total_dollar else "linear"
                ),
                yaxis=dict(
                    title=dict(text="", font=dict(size=16)),
                    tickfont=dict(size=14)
                ),
                font=dict(size=14),
                coloraxis_colorbar=dict(
                    tickfont=dict(size=14),
                    tickformat='.0f'
                )
            )
            
            # Update hover template to show millions format
            fig_total_dollar.update_traces(
                hovertemplate='<b>%{y}</b><br>Total Dollar Savings: $%{x:.0f}M<extra></extra>'
            )
            st.plotly_chart(fig_total_dollar, use_container_width=True, key="total_dollar_savings")
    
    with tab2:
        st.subheader("Unit Dosage Cost Analysis")
        
        # Create unit price focused dataframe
        unit_price_df = filtered_df[['Medicare Generic Name', 'Cost Plus Generic For',
                                   'Medicare Unit Price (2025$)', 'Cost Plus Min Price ($)', 'Cost Plus Max Price ($)',
                                   'Min Price Details', 'Max Price Details', 'Price Change 2022-2023 (%)', 'CAGR 2019-2023 (%)',
                                   'Potential Savings (Min) %', 'Potential Savings (Max) %']].copy()
        
        # Add dollar savings column using maximum Cost Plus price (minimum savings)
        unit_price_df['Min Dollar Savings per Unit ($)'] = unit_price_df['Medicare Unit Price (2025$)'] - unit_price_df['Cost Plus Max Price ($)']
        
        # Convert price change and CAGR to percentages (multiply by 100) and handle N/A values
        unit_price_df['Price Change 2022-2023 (%)'] = unit_price_df['Price Change 2022-2023 (%)']
        unit_price_df['CAGR 2019-2023 (%)'] = unit_price_df['CAGR 2019-2023 (%)']
        
        # Create summary table with key metrics only
        st.write("**💊 Summary Table - Key Unit Prices & Savings**")
        unit_summary_df = unit_price_df[['Medicare Generic Name', 'Medicare Unit Price (2025$)', 
                                       'Cost Plus Max Price ($)', 'Min Dollar Savings per Unit ($)', 
                                       'Potential Savings (Min) %']].copy()
        
        unit_summary_df.columns = ['Drug Name', 'Medicare Price', 'Cost Plus Max', 'Min Savings ($)', 'Max Savings (%)']
        
        styled_unit_summary = unit_summary_df.style.format({
            'Medicare Price': '${:.2f}',
            'Cost Plus Max': '${:.2f}',
            'Min Savings ($)': '${:.2f}',
            'Max Savings (%)': '{:.1f}%'
        }).map(color_savings, subset=['Min Savings ($)', 'Max Savings (%)'])
        
        st.dataframe(styled_unit_summary, use_container_width=True, hide_index=True)
        
        # Price trends in a focused table
        st.write("**📈 Medicare Price Trends (2019-2023)**")
        trends_df = unit_price_df[['Medicare Generic Name', 'Price Change 2022-2023 (%)', 'CAGR 2019-2023 (%)']].copy()
        trends_df.columns = ['Drug Name', 'Price Change 2022-23 (%)', 'CAGR 2019-23 (%)']
        
        styled_trends_df = trends_df.style.format({
            'Price Change 2022-23 (%)': '{:.3f}%',
            'CAGR 2019-23 (%)': '{:.3f}%'
        }).map(color_price_changes, subset=['Price Change 2022-23 (%)', 'CAGR 2019-23 (%)'])
        
        st.dataframe(styled_trends_df, use_container_width=True, hide_index=True)
        
        # Detailed breakdown in expandable section
        with st.expander("📋 Detailed Breakdown - Full Unit Price Data", expanded=False):
            # Split into two smaller tables
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Price Comparison Details**")
                price_detail_df = unit_price_df[['Medicare Generic Name', 'Medicare Unit Price (2025$)',
                                               'Cost Plus Min Price ($)', 'Cost Plus Max Price ($)',
                                               'Min Price Details', 'Max Price Details']].copy()
                
                price_detail_df.columns = ['Drug Name', 'Medicare Price', 'CP Min Price', 'CP Max Price', 'Min Details', 'Max Details']
                
                styled_price_detail = price_detail_df.style.format({
                    'Medicare Price': '${:.2f}',
                    'CP Min Price': '${:.2f}',
                    'CP Max Price': '${:.2f}'
                })
                
                st.dataframe(styled_price_detail, use_container_width=True, hide_index=True)
            
            with col2:
                st.write("**Comprehensive Savings Analysis**")
                unit_savings_detail_df = unit_price_df[['Medicare Generic Name', 'Cost Plus Generic For',
                                                      'Min Dollar Savings per Unit ($)',
                                                      'Potential Savings (Min) %', 'Potential Savings (Max) %']].copy()
                
                unit_savings_detail_df.columns = ['Drug Name', 'CP Generic Name', 'Min Savings ($)', 'Min Savings (%)', 'Max Savings (%)']
                
                styled_unit_savings_detail = unit_savings_detail_df.style.format({
                    'Min Savings ($)': '${:.2f}',
                    'Min Savings (%)': '{:.1f}%',
                    'Max Savings (%)': '{:.1f}%'
                }).map(color_savings, subset=['Min Savings ($)', 'Min Savings (%)', 'Max Savings (%)'])
                
                st.dataframe(styled_unit_savings_detail, use_container_width=True, hide_index=True)
    
        st.subheader("Unit Price Visualization")
    
        # Y-axis scale toggle for unit prices
        use_log_scale_unit = st.checkbox(
        "Use logarithmic scale for Unit Price (y-axis)",
        value=True,
            help="Logarithmic scale helps visualize large price differences more clearly",
            key="unit_log_scale"
        )
        
        # Unit price comparison chart
        fig_unit_comparison = go.Figure()
        
        # Medicare unit prices
        fig_unit_comparison.add_trace(go.Bar(
        x=chart_df['Medicare Generic Name'],
        y=chart_df['Medicare Unit Price (2025$)'],
        name='Medicare Part D (2025$)',
        marker=dict(color='red', opacity=0.7),
        hovertemplate='<b>%{x}</b><br>Medicare (2025$): $%{y:.0f}<extra></extra>'
    ))
    
        # Cost Plus unit price ranges
        fig_unit_comparison.add_trace(go.Bar(
        x=chart_df['Medicare Generic Name'],
        y=chart_df['Cost Plus Min Price ($)'],
        name='Cost Plus (Min)',
        marker=dict(color='blue', opacity=0.7),
        hovertemplate='<b>%{x}</b><br>Cost Plus Min: $%{y:.0f}<extra></extra>'
    ))
    
        fig_unit_comparison.add_trace(go.Bar(
        x=chart_df['Medicare Generic Name'],
        y=chart_df['Cost Plus Max Price ($)'],
        name='Cost Plus (Max)',
        marker=dict(color='lightblue', opacity=0.7),
        hovertemplate='<b>%{x}</b><br>Cost Plus Max: $%{y:.0f}<extra></extra>'
    ))
    
        fig_unit_comparison.update_layout(
        title=dict(
                text="Drug Unit Price Comparison: Medicare Part D vs Cost Plus",
            font=dict(size=20, family="Arial Black")
        ),
        xaxis_title="Medicare Generic Name",
        yaxis_title="Unit Price ($)",
            yaxis_type="log" if use_log_scale_unit else "linear",
        height=500,
            barmode='group',
        hovermode='x unified',
        xaxis=dict(
            tickangle=45,
            title=dict(font=dict(size=16)),
            tickfont=dict(size=16)
        ),
        yaxis=dict(
            title=dict(font=dict(size=16)),
            tickfont=dict(size=16)
        ),
        legend=dict(
            font=dict(size=14)
        ),
            font=dict(size=14)
        )
        
        st.plotly_chart(fig_unit_comparison, use_container_width=True, key="unit_comparison_main")
        
        # Top savings opportunities for unit prices
        st.subheader("Unit Price Savings Opportunities")
        # Sort by potential savings (min) which uses maximum Cost Plus prices - allow negative values
        top_savings_unit = filtered_df.sort_values('Potential Savings (Min) %', ascending=True)
        
        # Calculate dollar savings per unit using maximum Cost Plus price (minimum savings)
        top_savings_unit = top_savings_unit.copy()
        top_savings_unit['Dollar_Savings_Per_Unit'] = top_savings_unit['Medicare Unit Price (2025$)'] - top_savings_unit['Cost Plus Max Price ($)']
        
        # Create two columns for side-by-side charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Add spacing to align with col2's toggle
            st.write("")  # Empty space to align charts
            st.markdown('<div style="height: 1.5rem;"></div>', unsafe_allow_html=True)  # Custom spacing for alignment
            
            # Percentage savings chart
            fig_unit_pct = px.bar(
                top_savings_unit,
                x='Potential Savings (Min) %',
                y='Cost Plus Drug',
                orientation='h',
                title="Savings by Percentage (%) - Max Cost Plus (Min Savings)",
                color='Potential Savings (Min) %',
                color_continuous_scale='RdYlGn'  # Standard scale: min=red, max=green
            )
            
            fig_unit_pct.update_layout(
                height=600,
                title=dict(font=dict(size=18, family="Arial Black")),
                xaxis=dict(
                    title=dict(text="Savings (%) vs Max Cost Plus (Min Savings)", font=dict(size=16)),
                    tickfont=dict(size=14),
                    zeroline=True,
                    zerolinecolor='black',
                    zerolinewidth=2
                ),
                yaxis=dict(
                    title=dict(text="", font=dict(size=16)),
                    tickfont=dict(size=14)
                ),
                font=dict(size=14),
                coloraxis_colorbar=dict(
                    tickfont=dict(size=14),
                    tickformat='.0f'
                )
            )
            st.plotly_chart(fig_unit_pct, use_container_width=True, key="unit_pct_savings")
        
        with col2:
            # X-axis scale toggle for unit dollar savings
            use_log_scale_unit_dollar = st.checkbox(
                "Use logarithmic scale for Savings by Dollar Amount (x-axis)",
                value=True,
                help="Logarithmic scale helps visualize large savings differences more clearly",
                key="unit_dollar_log_scale"
            )
            
            # Dollar savings chart
            fig_unit_dollar = px.bar(
                top_savings_unit,
                x='Dollar_Savings_Per_Unit',
                y='Cost Plus Drug',
                orientation='h',
                title="Savings by Dollar Amount ($) - Max Cost Plus (Min Savings)",
                color='Dollar_Savings_Per_Unit',
                color_continuous_scale=[[0, '#90EE90'], [1, '#006400']]  # Light green to dark green
            )
            
            fig_unit_dollar.update_layout(
                height=600,
                title=dict(font=dict(size=18, family="Arial Black")),
                xaxis=dict(
                    title=dict(text="Dollar Savings per Unit ($) vs Max Cost Plus", font=dict(size=16)),
                    tickfont=dict(size=14),
                    tickformat='$.0f',
                    zeroline=True,
                    zerolinecolor='black',
                    zerolinewidth=2,
                    type="log" if use_log_scale_unit_dollar else "linear"
                ),
                yaxis=dict(
                    title=dict(text="", font=dict(size=16)),
                    tickfont=dict(size=14)
                ),
                font=dict(size=14),
                coloraxis_colorbar=dict(
                    tickfont=dict(size=14),
                    tickformat='.0f'
                )
            )
            st.plotly_chart(fig_unit_dollar, use_container_width=True, key="unit_dollar_savings")
    
    # Download options
    st.header("📥 Download Data")
    
    col1, col2 = st.columns(2)
    with col1:
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Comparison Data (CSV)",
            data=csv_data,
            file_name="cancer_drug_price_comparison.csv",
            mime="text/csv"
        )
    
    with col2:
        st.info(f"Dataset contains {len(filtered_df)} drug comparisons")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        <p>Cancer Drug Price Comparison Dashboard | Data sources: Medicare Part D Drug Spending Dashboard & Cost Plus Drugs</p>
        <p>For dissertation research purposes | Generated with Streamlit & Plotly</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 