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
    Loads and cleans the specified sheet from the Excel file.
    """
    try:
        # Load data with header at row 5
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=5)
        
        # Rename first column to 'Gas' and set as index
        df.rename(columns={df.columns[0]: 'Gas'}, inplace=True)
        df.set_index('Gas', inplace=True)
        
        # --- CRITICAL FIX: Handle Year Columns ---
        # 1. Force columns to numeric
        df.columns = pd.to_numeric(df.columns, errors='coerce')
        
        # 2. Drop columns that are not years
        df = df.dropna(axis=1, how='all')
        
        # 3. Convert column names to Integers
        df.columns = df.columns.astype(int)
        
        # 4. Drop empty rows
        df.dropna(how='all', inplace=True)
        
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
    "Territorial greenhouse gas emissions by gas, in million tonnes carbon dioxide equivalent (MtCO2e), UK"
]
selected_dataset = st.sidebar.selectbox("Select Dataset to Analyze", dataset_options)

# --- 3. Analysis Logic ---
if selected_dataset == "Territorial greenhouse gas emissions by gas, in million tonnes carbon dioxide equivalent (MtCO2e), UK":
    
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

        # 3. Forecast Slider (Added here for the section below)
        st.sidebar.markdown("---")
        st.sidebar.subheader("Forecast Settings")
        forecast_years = st.sidebar.slider("Forecast Horizon (Years)", 1, 30, 10)
        
        # Apply Filters
        df_ts_filtered = df_ts.loc[start_year:end_year, selected_gases]
        df_total_filtered = df_total_ts.loc[start_year:end_year]
        
        # =====================================================
        # PART A: DESCRIPTIVE ANALYSIS (EXISTING PLOTS)
        # =====================================================
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

        # =====================================================
        # PART B: PREDICTIVE FORECASTING (NEW SECTION)
        # =====================================================
        st.markdown("---")
        st.header("Predictive Forecasting (ARIMA)")
        st.info(f"Forecasting Total GHG Emissions for the next {forecast_years} years based on historical trends.")

        # Load Model
        model = load_model()
        
        if model is not None:
            # Prepare Data for Plotting
            last_hist_year = df_total_ts.index.max()
            future_year_end = last_hist_year + forecast_years
            
            # Generate Forecast
            forecast_result = model.get_forecast(steps=forecast_years)
            forecast_values = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int()
            
            # Create Future Index
            future_index = range(last_hist_year + 1, future_year_end + 1)
            forecast_df = pd.DataFrame({
                'Year': future_index,
                'Forecast': forecast_values.values,
                'Lower CI': conf_int.iloc[:, 0].values,
                'Upper CI': conf_int.iloc[:, 1].values
            }).set_index('Year')
            
            # --- Visualization: Forecast ---
            fig_pred, ax_pred = plt.subplots(figsize=(12, 6))
            
            # Plot History (Full History for context)
            ax_pred.plot(df_total_ts.index, df_total_ts.values, label='Historical Data', color='black', linewidth=2)
            
            # Plot Forecast
            ax_pred.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='red', linestyle='--', linewidth=2)
            
            # Plot Confidence Interval
            ax_pred.fill_between(
                forecast_df.index, 
                forecast_df['Lower CI'], 
                forecast_df['Upper CI'], 
                color='pink', alpha=0.4, label='95% Confidence Interval'
            )
            
            # Net Zero Target Line (2050)
            if future_year_end >= 2050:
                ax_pred.axhline(0, color='green', linestyle=':', linewidth=2, label='Net Zero Target')
            
            ax_pred.set_xlabel("Year")
            ax_pred.set_ylabel("Emissions (MtCO2e)")
            ax_pred.set_title(f"UK Greenhouse Gas Emissions Trajectory (Forecast to {future_year_end})")
            ax_pred.legend()
            ax_pred.grid(True, alpha=0.3)
            
            st.pyplot(fig_pred)
            
            # --- Forecast Data Table ---
            with st.expander("View Detailed Forecast Data"):
                st.dataframe(forecast_df.style.format("{:.2f}"))