import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# Set page configuration
st.set_page_config(page_title="UK GHG Emissions Dashboard", layout="wide")

# --- 1. Data Loading Functions ---
@st.cache_data
def load_data(file_path, sheet_name):
    """
    Loads and cleans the specified sheet.
    Includes specific logic for Table 1.1 (Gas) vs Table 1.2 (Sectors).
    """
    try:
        # Load data with header at row 5
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=5)
        
        # --- LOGIC FOR TABLE 1.1 (GASES) ---
        if sheet_name == '1.1' or sheet_name == 'Table 1.1':
            # Rename first column to 'Gas' and set as index
            df.rename(columns={df.columns[0]: 'Gas'}, inplace=True)
            df.set_index('Gas', inplace=True)
            
            # Fix Year Columns
            df.columns = pd.to_numeric(df.columns, errors='coerce')
            df = df.dropna(axis=1, how='all')
            df.columns = df.columns.astype(int)
            df.dropna(how='all', inplace=True)
            return df

        # --- LOGIC FOR TABLE 1.2 (SECTORS) ---
        elif sheet_name == '1.2' or sheet_name == 'Table 1.2':
            # Clean column names (remove whitespace)
            df.columns = [c if isinstance(c, int) else str(c).strip() for c in df.columns]
            
            # Filter for Main Sectors (rows ending in ' total' but not 'Grand total')
            if 'TES sector' in df.columns:
                mask = (df['TES sector'].str.endswith(' total', na=False)) & \
                       (df['TES sector'] != 'Grand total')
                df = df[mask].copy()
                
                # Clean Index Names (remove ' total')
                df['TES sector'] = df['TES sector'].str.replace(' total', '')
                df.set_index('TES sector', inplace=True)
                
                # Fix Year Columns (Keep only numeric)
                year_cols = [c for c in df.columns if str(c).replace('.','').isdigit()]
                df = df[year_cols]
                
                # Force headers to int and values to numeric
                df.columns = pd.to_numeric(df.columns).astype(int)
                df = df.apply(pd.to_numeric, errors='coerce')
                return df
            else:
                st.error("Column 'TES sector' not found in Table 1.2")
                return None
                
        return df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def load_model():
    """Loads the pre-trained ARIMA model."""
    try:
        model = joblib.load('arima_ghg_model.pkl')
        return model
    except FileNotFoundError:
        st.warning("⚠️ Model file 'arima_ghg_model.pkl' not found. Please run the training notebook first.")
        return None

# --- 2. Main Dashboard Layout ---
st.title("🇬🇧 UK Greenhouse Gas Emissions Dashboard")
st.markdown("Interactive analysis and forecasting of UK territorial emissions (1990-2023).")

# Sidebar: Configuration
st.sidebar.header("Configuration")
file_path = 'final-greenhouse-gas-emissions-tables-2023.xlsx' 

# Dropdown Options
dataset_options = [
    "Territorial greenhouse gas emissions by gas",
    "Territorial greenhouse gas emissions by source category"
]
selected_dataset = st.sidebar.selectbox("Select Dataset to Analyze", dataset_options)

# ==============================================================================
# OPTION 1: TABLE 1.1 (GASES + FORECASTING)
# ==============================================================================
if selected_dataset == "Territorial greenhouse gas emissions by gas":
    
    # Load Data
    df = load_data(file_path, '1.1')
    
    if df is not None:
        # Separate Total from Components
        total_row = df.loc['Total greenhouse gas emissions']
        components = df.drop('Total greenhouse gas emissions')
        
        # Transpose for Time Series Analysis
        df_ts = components.T
        df_total_ts = total_row.T
        
        # --- SIDEBAR CONTROLS ---
        st.sidebar.subheader("Filter Options (Descriptive)")
        
        # 1. Year Range Slider
        min_year = int(df_ts.index.min())
        max_year = int(df_ts.index.max())
        start_year, end_year = st.sidebar.slider(
            "Select Historical Year Range", 
            min_value=min_year, 
            max_value=max_year, 
            value=(min_year, max_year)
        )
        
        # 2. Gas Selector
        all_gases = df_ts.columns.tolist()
        selected_gases = st.sidebar.multiselect(
            "Select Gases to Display", 
            all_gases, 
            default=all_gases
        )

        # 3. Forecast Slider
        st.sidebar.markdown("---")
        st.sidebar.subheader("Forecast Settings")
        forecast_years = st.sidebar.slider("Forecast Horizon (Years)", 1, 30, 10)
        
        # Apply Filters
        df_ts_filtered = df_ts.loc[start_year:end_year, selected_gases]
        df_total_filtered = df_total_ts.loc[start_year:end_year]
        
        # --- PART A: DESCRIPTIVE ANALYSIS ---
        st.header("Descriptive Analysis")
        
        # Row 1: Headline Trend
        st.subheader(f"1. Total Emissions Trend ({start_year}-{end_year})")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(df_total_filtered.index, df_total_filtered.values, color='#1f77b4', linewidth=3, label='Total Emissions')
        ax1.plot(df_total_filtered.index, df_total_filtered.rolling(window=5).mean(), color='orange', linestyle='--', label='5-Year Moving Avg')
        ax1.set_ylabel('Emissions (MtCO2e)')
        ax1.set_xlabel('Year')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

        # Row 2: Composition and Normalized Trends
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("2. Composition by Gas (Stacked)")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.stackplot(df_ts_filtered.index, df_ts_filtered.T.values, labels=df_ts_filtered.columns, alpha=0.8)
            ax2.set_ylabel('Emissions (MtCO2e)')
            ax2.legend(loc='upper left', fontsize='small')
            st.pyplot(fig2)
            
        with col2:
            st.subheader("3. Relative Change (Index 1990=100)")
            if not df_ts_filtered.empty:
                df_normalized = df_ts_filtered.div(df_ts_filtered.iloc[0]) * 100
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                for column in df_normalized.columns:
                    ax3.plot(df_normalized.index, df_normalized[column], linewidth=2, label=column)
                ax3.axhline(100, color='black', linestyle='--', linewidth=1, alpha=0.5)
                ax3.set_ylabel('Percent of Start Year')
                st.pyplot(fig3)
            else:
                st.warning("No data to display for normalized chart.")

        # Row 3: Snapshot Comparison (Donut Charts)
        st.subheader(f"4. Gas Share Comparison: {start_year} vs {end_year}")
        fig4, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Start Year Donut
        if start_year in components.columns:
            axes[0].pie(components.loc[selected_gases, start_year], labels=None, startangle=90, autopct='%1.1f%%', pctdistance=0.85)
            axes[0].set_title(f'{start_year} Share')
            axes[0].add_artist(plt.Circle((0,0),0.70,fc='white'))
        
        # End Year Donut
        if end_year in components.columns:
            axes[1].pie(components.loc[selected_gases, end_year], labels=None, startangle=90, autopct='%1.1f%%', pctdistance=0.85)
            axes[1].set_title(f'{end_year} Share')
            axes[1].add_artist(plt.Circle((0,0),0.70,fc='white'))
        
        # Common Legend
        axes[1].legend(selected_gases, loc='center left', bbox_to_anchor=(1, 0.5))
        st.pyplot(fig4)

        # Row 4: Waterfall/Bar of Absolute Change
        st.subheader(f"5. Absolute Reduction by Gas ({start_year} to {end_year})")
        if start_year in components.columns and end_year in components.columns:
            change = components.loc[selected_gases, end_year] - components.loc[selected_gases, start_year]
            colors = ['green' if x < 0 else 'red' for x in change]
            
            fig5, ax5 = plt.subplots(figsize=(10, 5))
            change.sort_values().plot(kind='bar', color=colors, ax=ax5)
            ax5.axhline(0, color='black', linewidth=1)
            ax5.set_ylabel('Change (MtCO2e)')
            ax5.grid(axis='y', linestyle='--', alpha=0.5)
            st.pyplot(fig5)

        # --- PART B: PREDICTIVE FORECASTING ---
        st.markdown("---")
        st.header("Predictive Forecasting (ARIMA)")
        st.info(f"Forecasting Total GHG Emissions for the next {forecast_years} years based on historical trends.")

        model = load_model()
        
        if model is not None:
            # Prepare Data
            last_hist_year = df_total_ts.index.max()
            future_year_end = last_hist_year + forecast_years
            
            # Forecast
            forecast_result = model.get_forecast(steps=forecast_years)
            forecast_values = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int()
            
            # Future Index
            future_index = range(last_hist_year + 1, future_year_end + 1)
            forecast_df = pd.DataFrame({
                'Year': future_index,
                'Forecast': forecast_values.values,
                'Lower CI': conf_int.iloc[:, 0].values,
                'Upper CI': conf_int.iloc[:, 1].values
            }).set_index('Year')
            
            # Visualization
            fig_pred, ax_pred = plt.subplots(figsize=(12, 6))
            ax_pred.plot(df_total_ts.index, df_total_ts.values, label='Historical Data', color='black', linewidth=2)
            ax_pred.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='red', linestyle='--', linewidth=2)
            ax_pred.fill_between(forecast_df.index, forecast_df['Lower CI'], forecast_df['Upper CI'], color='pink', alpha=0.4, label='95% Confidence Interval')
            
            if future_year_end >= 2050:
                ax_pred.axhline(0, color='green', linestyle=':', linewidth=2, label='Net Zero Target')
            
            ax_pred.set_xlabel("Year")
            ax_pred.set_ylabel("Emissions (MtCO2e)")
            ax_pred.set_title(f"UK Greenhouse Gas Emissions Trajectory (Forecast to {future_year_end})")
            ax_pred.legend()
            ax_pred.grid(True, alpha=0.3)
            st.pyplot(fig_pred)
            
            with st.expander("View Detailed Forecast Data"):
                st.dataframe(forecast_df.style.format("{:.2f}"))

# ==============================================================================
# OPTION 2: TABLE 1.2 (SECTORS)
# ==============================================================================
elif selected_dataset == "Territorial greenhouse gas emissions by source category":
    
    # Load Data (Using updated load_data that handles 1.2)
    df_sectors = load_data(file_path, '1.2')
    
    if df_sectors is not None:
        # Transpose for Time Series Analysis (Rows = Years, Cols = Sectors)
        df_ts = df_sectors.T
        
        # --- SIDEBAR CONTROLS ---
        st.sidebar.subheader("Sector Filter Options")
        
        # 1. Year Range Slider
        min_year = int(df_ts.index.min())
        max_year = int(df_ts.index.max())
        start_year, end_year = st.sidebar.slider(
            "Select Year Range", 
            min_value=min_year, 
            max_value=max_year, 
            value=(min_year, max_year)
        )
        
        # 2. Sector Multiselect
        all_sectors = df_ts.columns.tolist()
        # Default to Top 5
        top_5_sectors = df_ts.loc[max_year].sort_values(ascending=False).head(5).index.tolist()
        selected_sectors = st.sidebar.multiselect(
            "Select Sectors to Compare", 
            all_sectors, 
            default=top_5_sectors
        )
        
        # Apply Filters
        df_filtered = df_ts.loc[start_year:end_year, selected_sectors]
        
        # --- VISUALIZATIONS ---
        st.header(f"Sectoral Emissions Analysis ({start_year}-{end_year})")

        if not df_filtered.empty:
            
            # VISUALIZATION 1: Multi-Line Trend
            st.subheader("1. Emission Trends by Sector")
            st.markdown("Compare the absolute emissions trajectories of different sectors.")
            
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            for sector in df_filtered.columns:
                ax1.plot(df_filtered.index, df_filtered[sector], marker='o', markersize=3, label=sector)
            
            ax1.set_ylabel("Emissions (MtCO2e)")
            ax1.set_xlabel("Year")
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)
            
            # VISUALIZATION 2: Stacked Area Chart
            st.subheader("2. Total Emissions Composition")
            st.markdown("How much does each sector contribute to the total pool over time?")
            
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.stackplot(df_filtered.index, df_filtered.T.values, labels=df_filtered.columns, alpha=0.8)
            ax2.set_ylabel("Emissions (MtCO2e)")
            ax2.set_xlabel("Year")
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig2)
            
            col1, col2 = st.columns(2)
            
            # VISUALIZATION 3: Normalized Trends
            with col1:
                st.subheader("3. Relative Change (Index: Start Year = 100)")
                st.markdown("Compare *speed* of reduction/growth, ignoring size differences.")
                
                # Normalize to the first selected year
                df_norm = df_filtered.div(df_filtered.iloc[0], axis=1) * 100
                
                fig3, ax3 = plt.subplots(figsize=(6, 5))
                for sector in df_norm.columns:
                    ax3.plot(df_norm.index, df_norm[sector], linewidth=2, label=sector)
                
                ax3.axhline(100, color='black', linestyle='--', alpha=0.5)
                ax3.set_ylabel(f"% of {start_year} Emissions")
                st.pyplot(fig3)

            # VISUALIZATION 4: Snapshot Bar Chart
            with col2:
                st.subheader(f"4. Top Emitters in {end_year}")
                st.markdown(f"Snapshot of emissions in the final selected year.")
                
                # Sort by value
                snapshot = df_filtered.loc[end_year].sort_values(ascending=True)
                
                fig4, ax4 = plt.subplots(figsize=(6, 5))
                snapshot.plot(kind='barh', color='teal', ax=ax4)
                ax4.set_xlabel("MtCO2e")
                ax4.grid(axis='x', alpha=0.3)
                st.pyplot(fig4)

            # VISUALIZATION 5: Heatmap
            st.subheader("5. Emission Intensity Heatmap")
            st.markdown("A bird's-eye view of emission intensity. Darker colors = Higher emissions.")
            
            fig5, ax5 = plt.subplots(figsize=(12, 6))
            # Transpose back so Years are X-axis, Sectors are Y-axis
            sns.heatmap(df_filtered.T, cmap="YlOrRd", linewidths=.5, ax=ax5, cbar_kws={'label': 'MtCO2e'})
            ax5.set_xlabel("Year")
            ax5.set_ylabel("Sector")
            st.pyplot(fig5)

        else:
            st.warning("Please select at least one sector to visualize.")