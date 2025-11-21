import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from sklearn.decomposition import PCA # Required for the Scatter Plot

# Set page configuration
st.set_page_config(page_title="UK GHG Emissions Dashboard", layout="wide")

# --- 1. Data Loading Functions ---
@st.cache_data
def load_data(file_path, sheet_name):
    """
    Loads and cleans the specified sheet.
    Includes specific logic for Tables 1.1 through 1.7.
    """
    try:
        # --- LOGIC FOR TABLE 1.1 (GASES) ---
        if sheet_name in ['1.1', 'Table 1.1']:
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=5)
            df.rename(columns={df.columns[0]: 'Gas'}, inplace=True)
            df.set_index('Gas', inplace=True)
            df.columns = pd.to_numeric(df.columns, errors='coerce')
            df = df.dropna(axis=1, how='all')
            df.columns = df.columns.astype(int)
            df.dropna(how='all', inplace=True)
            return df

        # --- LOGIC FOR TABLE 1.6 (F-GASES - Header Row 6) ---
        elif sheet_name in ['1.6', 'Table 1.6']:
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=6)
            df.columns = [c if isinstance(c, int) else str(c).strip() for c in df.columns]
            
            if 'TES sector' in df.columns:
                mask = (df['TES sector'].str.endswith(' total', na=False)) & \
                       (df['TES sector'] != 'Grand total')
                df = df[mask].copy()
                df['TES sector'] = df['TES sector'].str.replace(' total', '')
                df.set_index('TES sector', inplace=True)
                
                year_cols = [c for c in df.columns if str(c).replace('.','').isdigit()]
                df = df[year_cols]
                df.columns = pd.to_numeric(df.columns).astype(int)
                df = df.apply(pd.to_numeric, errors='coerce')
                return df
            else:
                st.error(f"Column 'TES sector' not found in {sheet_name}")
                return None

        # --- LOGIC FOR TABLE 1.7 (FUEL TYPE - Header Row 4) ---
        elif sheet_name in ['1.7', 'Table 1.7']:
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=4)
            df.columns = [c if isinstance(c, int) else str(c).strip() for c in df.columns]
            
            # Handle column naming variation
            if 'Fuel group' not in df.columns:
                # Fallback: Rename first column if it looks right
                df.rename(columns={df.columns[0]: 'Fuel group'}, inplace=True)
            
            # Group by Fuel Group and sum
            year_cols = [c for c in df.columns if str(c).replace('.','').isdigit()]
            # We must sum because Table 1.7 has multiple rows per Fuel Group
            df_agg = df.groupby('Fuel group')[year_cols].sum()
            
            # Fix Columns
            df_agg.columns = df_agg.columns.astype(int)
            
            # Remove totals
            if 'Grand total' in df_agg.index: df_agg = df_agg.drop('Grand total')
            if 'Total' in df_agg.index: df_agg = df_agg.drop('Total')
            
            return df_agg

        # --- LOGIC FOR SECTOR TABLES (1.2, 1.3, 1.4, 1.5 - Header Row 5) ---
        elif sheet_name in ['1.2', 'Table 1.2', '1.3', 'Table 1.3', '1.4', 'Table 1.4', '1.5', 'Table 1.5']:
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=5)
            df.columns = [c if isinstance(c, int) else str(c).strip() for c in df.columns]
            
            if 'TES sector' in df.columns:
                mask = (df['TES sector'].str.endswith(' total', na=False)) & \
                       (df['TES sector'] != 'Grand total')
                df = df[mask].copy()
                df['TES sector'] = df['TES sector'].str.replace(' total', '')
                df.set_index('TES sector', inplace=True)
                
                year_cols = [c for c in df.columns if str(c).replace('.','').isdigit()]
                df = df[year_cols]
                df.columns = pd.to_numeric(df.columns).astype(int)
                df = df.apply(pd.to_numeric, errors='coerce')
                return df
            else:
                st.error(f"Column 'TES sector' not found in {sheet_name}")
                return None
                
        return df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def load_model():
    try:
        return joblib.load('arima_ghg_model.pkl')
    except FileNotFoundError:
        return None

def load_kmeans_model():
    try:
        return joblib.load('kmeans_clustering_model.pkl')
    except FileNotFoundError:
        return None

def load_rf_model():
    try:
        return joblib.load('fuel_classification_model.pkl')
    except FileNotFoundError:
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
    "Territorial greenhouse gas emissions by source category",
    "Territorial emissions of carbon dioxide (CO2) by source category",
    "Territorial emissions of methane (CH4) by source category",
    "Territorial emissions of nitrous oxide (N2O) by source category",
    "Territorial emissions of fluorinated gases (F gases) by source category",
    "Territorial greenhouse gas emissions by type of fuel"
]
selected_dataset = st.sidebar.selectbox("Select Dataset to Analyze", dataset_options)

# ==============================================================================
# OPTION 1: TABLE 1.1 (GASES + FORECASTING)
# ==============================================================================
if selected_dataset == "Territorial greenhouse gas emissions by gas":
    
    df = load_data(file_path, '1.1')
    
    if df is not None:
        total_row = df.loc['Total greenhouse gas emissions']
        components = df.drop('Total greenhouse gas emissions')
        df_ts = components.T
        df_total_ts = total_row.T
        
        st.sidebar.subheader("Filter Options (Descriptive)")
        min_year, max_year = int(df_ts.index.min()), int(df_ts.index.max())
        start_year, end_year = st.sidebar.slider("Historical Year Range", min_year, max_year, (min_year, max_year))
        selected_gases = st.sidebar.multiselect("Select Gases", df_ts.columns.tolist(), default=df_ts.columns.tolist())
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Forecast Settings")
        forecast_years = st.sidebar.slider("Forecast Horizon (Years)", 1, 30, 10)
        
        df_ts_filtered = df_ts.loc[start_year:end_year, selected_gases]
        df_total_filtered = df_total_ts.loc[start_year:end_year]
        
        st.header("Descriptive Analysis")
        st.subheader(f"1. Total Emissions Trend ({start_year}-{end_year})")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(df_total_filtered.index, df_total_filtered.values, color='#1f77b4', linewidth=3, label='Total Emissions')
        ax1.plot(df_total_filtered.index, df_total_filtered.rolling(window=5).mean(), color='orange', linestyle='--', label='5-Year Moving Avg')
        ax1.set_ylabel('Emissions (MtCO2e)')
        ax1.set_xlabel('Year')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("2. Composition by Gas")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.stackplot(df_ts_filtered.index, df_ts_filtered.T.values, labels=df_ts_filtered.columns, alpha=0.8)
            ax2.legend(loc='upper left', fontsize='small')
            st.pyplot(fig2)
        with col2:
            st.subheader("3. Relative Change (Index 1990=100)")
            if not df_ts_filtered.empty:
                df_normalized = df_ts_filtered.div(df_ts_filtered.iloc[0]) * 100
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                for column in df_normalized.columns:
                    ax3.plot(df_normalized.index, df_normalized[column], label=column)
                ax3.axhline(100, color='black', linestyle='--')
                st.pyplot(fig3)

        st.subheader(f"4. Gas Share Comparison: {start_year} vs {end_year}")
        fig4, axes = plt.subplots(1, 2, figsize=(10, 5))
        if start_year in components.columns:
            axes[0].pie(components.loc[selected_gases, start_year], labels=None, startangle=90, autopct='%1.1f%%', pctdistance=0.85)
            axes[0].set_title(f'{start_year} Share')
            axes[0].add_artist(plt.Circle((0,0),0.70,fc='white'))
        if end_year in components.columns:
            axes[1].pie(components.loc[selected_gases, end_year], labels=None, startangle=90, autopct='%1.1f%%', pctdistance=0.85)
            axes[1].set_title(f'{end_year} Share')
            axes[1].add_artist(plt.Circle((0,0),0.70,fc='white'))
        axes[1].legend(selected_gases, loc='center left', bbox_to_anchor=(1, 0.5))
        st.pyplot(fig4)

        st.subheader(f"5. Absolute Reduction by Gas ({start_year} to {end_year})")
        change = components.loc[selected_gases, end_year] - components.loc[selected_gases, start_year]
        colors = ['green' if x < 0 else 'red' for x in change]
        fig5, ax5 = plt.subplots(figsize=(10, 5))
        change.sort_values().plot(kind='bar', color=colors, ax=ax5)
        ax5.axhline(0, color='black')
        st.pyplot(fig5)

        st.markdown("---")
        st.header("Predictive Forecasting (ARIMA)")
        model = load_model()
        if model:
            last_hist = df_total_ts.index.max()
            future_end = last_hist + forecast_years
            fc_res = model.get_forecast(steps=forecast_years)
            fc_df = pd.DataFrame({
                'Forecast': fc_res.predicted_mean.values,
                'Lower': fc_res.conf_int().iloc[:, 0].values,
                'Upper': fc_res.conf_int().iloc[:, 1].values
            }, index=range(last_hist+1, future_end+1))
            
            fig_p, ax_p = plt.subplots(figsize=(12, 6))
            ax_p.plot(df_total_ts.index, df_total_ts.values, label='History', color='black')
            ax_p.plot(fc_df.index, fc_df['Forecast'], label='Forecast', color='red', linestyle='--')
            ax_p.fill_between(fc_df.index, fc_df['Lower'], fc_df['Upper'], color='pink', alpha=0.4)
            if future_end >= 2050: ax_p.axhline(0, color='green', linestyle=':', label='Net Zero Target')
            ax_p.legend()
            ax_p.set_title(f"Forecast to {future_end}")
            st.pyplot(fig_p)
            with st.expander("View Data"): st.dataframe(fc_df.style.format("{:.2f}"))
        else:
            st.warning("Model not found.")

# ==============================================================================
# OPTION 2: TABLE 1.2 (SECTORS + CLUSTERING)
# ==============================================================================
elif selected_dataset == "Territorial greenhouse gas emissions by source category":
    
    df_sectors = load_data(file_path, '1.2')
    
    if df_sectors is not None:
        df_ts = df_sectors.T
        
        st.sidebar.subheader("Sector Options")
        min_y, max_y = int(df_ts.index.min()), int(df_ts.index.max())
        start_year, end_year = st.sidebar.slider("Year Range", min_y, max_y, (min_y, max_y))
        all_secs = df_ts.columns.tolist()
        top_5 = df_ts.loc[max_y].sort_values(ascending=False).head(5).index.tolist()
        selected_sectors = st.sidebar.multiselect("Select Sectors", all_secs, default=top_5)
        
        df_filt = df_ts.loc[start_year:end_year, selected_sectors]
        
        st.header(f"Sectoral Emissions Analysis ({start_year}-{end_year})")
        
        if not df_filt.empty:
            st.subheader("1. Emission Trends")
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            for s in df_filt.columns:
                ax1.plot(df_filt.index, df_filt[s], marker='o', markersize=3, label=s)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)
            
            st.subheader("2. Composition")
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.stackplot(df_filt.index, df_filt.T.values, labels=df_filt.columns, alpha=0.8)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig2)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("3. Relative Change (Start=100)")
                df_norm = df_filt.div(df_filt.iloc[0], axis=1) * 100
                fig3, ax3 = plt.subplots(figsize=(6, 5))
                for s in df_norm.columns: ax3.plot(df_norm.index, df_norm[s], label=s)
                ax3.axhline(100, color='black', linestyle='--')
                st.pyplot(fig3)
            with col2:
                st.subheader(f"4. Top Emitters in {end_year}")
                snap = df_filt.loc[end_year].sort_values()
                fig4, ax4 = plt.subplots(figsize=(6, 5))
                snap.plot(kind='barh', color='teal', ax=ax4)
                st.pyplot(fig4)

            st.subheader("5. Emission Intensity Heatmap")
            fig5, ax5 = plt.subplots(figsize=(12, 6))
            sns.heatmap(df_filt.T, cmap="YlOrRd", linewidths=.5, ax=ax5)
            st.pyplot(fig5)
        else:
            st.warning("Select at least one sector.")

        st.markdown("---")
        st.header("Unsupervised Learning: Sector Clustering")
        kmeans_model = load_kmeans_model()
        if kmeans_model:
            start_vals = df_sectors.iloc[:, 0].replace(0, 1e-9) 
            df_clus_norm = df_sectors.div(start_vals, axis=0).fillna(0)
            clusters = kmeans_model.predict(df_clus_norm)
            df_results = df_clus_norm.copy()
            df_results['Cluster_ID'] = clusters
            
            cluster_means = df_results.groupby('Cluster_ID').mean().iloc[:, -1]
            sorted_clusters = cluster_means.sort_values().index.tolist()
            cluster_names = {}
            if len(sorted_clusters) >= 3:
                cluster_names[sorted_clusters[0]] = "Rapid Decarbonizers"
                cluster_names[sorted_clusters[1]] = "Moderate Reducers"
                cluster_names[sorted_clusters[2]] = "Hard-to-Abate Sectors"
                for i in sorted_clusters[3:]: cluster_names[i] = f"Cluster {i}"
            else:
                cluster_names[sorted_clusters[0]] = "Leaders"
                cluster_names[sorted_clusters[1]] = "Laggards"
            df_results['Cluster_Name'] = df_results['Cluster_ID'].map(cluster_names)
            
            st.subheader("Cluster Trajectories")
            fig_c, ax_c = plt.subplots(figsize=(12, 6))
            colors = sns.color_palette("deep", len(sorted_clusters))
            unique_names = df_results['Cluster_Name'].unique()
            for i, name in enumerate(unique_names):
                subset = df_results[df_results['Cluster_Name'] == name]
                centroid = subset.iloc[:, :-2].mean(axis=0)
                ax_c.plot(centroid.index, centroid.values, color=colors[i], linewidth=3, label=name)
                for sec in subset.index:
                    ax_c.plot(subset.columns[:-2], subset.loc[sec, subset.columns[:-2]], color=colors[i], alpha=0.1)
            ax_c.axhline(1.0, color='black', linestyle='--')
            ax_c.legend()
            st.pyplot(fig_c)
            
            st.subheader("Cluster Groups (PCA)")
            pca = PCA(n_components=2)
            pcs = pca.fit_transform(df_clus_norm)
            df_pca = pd.DataFrame(data=pcs, columns=['PC1', 'PC2'], index=df_clus_norm.index)
            df_pca['Group'] = df_results['Cluster_Name']
            fig_pca, ax_pca = plt.subplots(figsize=(10, 8))
            sns.scatterplot(x='PC1', y='PC2', hue='Group', data=df_pca, palette='deep', s=100, ax=ax_pca)
            for i in range(df_pca.shape[0]):
                ax_pca.text(df_pca.PC1[i]+0.05, df_pca.PC2[i], df_pca.index[i], fontsize=8, alpha=0.7)
            st.pyplot(fig_pca)
            with st.expander("View Clusters"): st.dataframe(df_results[['Cluster_Name']].sort_values('Cluster_Name'))
        else:
            st.warning("Clustering Model not found.")

# ==============================================================================
# OPTION 3, 4, 5, 6: SECTOR TABLES (CO2, CH4, N2O, F-GASES)
# ==============================================================================
elif selected_dataset in [
    "Territorial emissions of carbon dioxide (CO2) by source category",
    "Territorial emissions of methane (CH4) by source category",
    "Territorial emissions of nitrous oxide (N2O) by source category",
    "Territorial emissions of fluorinated gases (F gases) by source category"
]:
    # Map selection to table ID
    table_map = {
        "Territorial emissions of carbon dioxide (CO2) by source category": '1.3',
        "Territorial emissions of methane (CH4) by source category": '1.4',
        "Territorial emissions of nitrous oxide (N2O) by source category": '1.5',
        "Territorial emissions of fluorinated gases (F gases) by source category": '1.6'
    }
    
    # Set color palettes based on gas type
    color_map = {
        '1.3': 'Reds',
        '1.4': 'Greens',
        '1.5': 'Purples',
        '1.6': 'Blues'
    }
    
    sheet_id = table_map[selected_dataset]
    df_gas = load_data(file_path, sheet_id)
    
    if df_gas is not None:
        df_ts = df_gas.T
        
        st.sidebar.subheader("Filter Options")
        # Rename to min_year and max_year to allow usage in top_5 calculation
        min_year, max_year = int(df_ts.index.min()), int(df_ts.index.max())
        start_year, end_year = st.sidebar.slider("Year Range", min_year, max_year, (min_year, max_year))
        
        all_sectors = df_ts.columns.tolist()
        # Filter active sectors (non-zero sum) to reduce clutter
        active_sectors = df_ts.columns[df_ts.sum() > 0].tolist()
        
        # Dynamic top 5 logic
        if not active_sectors:
            st.warning("No active sectors found for this dataset.")
            selected_sectors = []
        else:
            # Use max_year defined above
            top_5 = df_ts.loc[max_year].sort_values(ascending=False).head(5).index.tolist()
            # If top_5 is empty (all zeros), fallback to top_5 from sum
            if not top_5:
                top_5 = df_ts.sum().sort_values(ascending=False).head(5).index.tolist()
            
            # --- CRITICAL FIX FOR STREAMLIT API EXCEPTION ---
            # Ensure default 'top_5' only contains values present in 'active_sectors' options
            default_selection = [s for s in top_5 if s in active_sectors]
            
            selected_sectors = st.sidebar.multiselect("Select Sectors", active_sectors, default=default_selection)
        
        if selected_sectors:
            df_filtered = df_ts.loc[start_year:end_year, selected_sectors]
            
            gas_name = selected_dataset.split(' of ')[1].split(' by ')[0]
            st.header(f"{gas_name} Analysis ({start_year}-{end_year})")
            
            if not df_filtered.empty:
                # Vis 1: Trend
                st.subheader("1. Emission Trends")
                fig1, ax1 = plt.subplots(figsize=(10, 5))
                for s in df_filtered.columns:
                    ax1.plot(df_filtered.index, df_filtered[s], marker='o', markersize=3, label=s)
                ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax1.grid(True, alpha=0.3)
                st.pyplot(fig1)
                
                # Vis 2: Stacked Area
                st.subheader("2. Composition")
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                ax2.stackplot(df_filtered.index, df_filtered.T.values, labels=df_filtered.columns, alpha=0.8)
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                st.pyplot(fig2)
                
                col1, col2 = st.columns(2)
                
                # Vis 3: Snapshot
                with col1:
                    st.subheader(f"3. Top Emitters in {end_year}")
                    snap = df_filtered.loc[end_year].sort_values()
                    snap = snap[snap > 0]
                    if not snap.empty:
                        fig3, ax3 = plt.subplots(figsize=(6, 5))
                        snap.plot(kind='barh', ax=ax3)
                        st.pyplot(fig3)
                    else:
                        st.info("No emissions in selected year.")
                    
                # Vis 4: Diverging Change
                with col2:
                    st.subheader("4. Absolute Change")
                    change = df_filtered.loc[end_year] - df_filtered.loc[start_year]
                    change = change.sort_values()
                    cols = ['green' if x < 0 else 'red' for x in change.values]
                    fig4, ax4 = plt.subplots(figsize=(6, 5))
                    ax4.barh(change.index, change.values, color=cols)
                    ax4.axvline(0, color='black')
                    st.pyplot(fig4)
                    
                # Vis 5: Heatmap
                st.subheader("5. Intensity Heatmap")
                fig5, ax5 = plt.subplots(figsize=(12, 6))
                sns.heatmap(df_filtered.T, cmap=color_map[sheet_id], linewidths=.5, ax=ax5)
                st.pyplot(fig5)
            else:
                st.warning("Select at least one sector.")
        else:
            st.warning("No sectors selected.")

# ==============================================================================
# OPTION 7: TABLE 1.7 (FUEL TYPE + CLASSIFICATION)
# ==============================================================================
elif selected_dataset == "Territorial greenhouse gas emissions by type of fuel":
    
    df_fuel = load_data(file_path, '1.7')
    
    if df_fuel is not None:
        df_ts = df_fuel.T
        
        st.sidebar.subheader("Fuel Filters")
        min_y, max_y = int(df_ts.index.min()), int(df_ts.index.max())
        start_year, end_year = st.sidebar.slider("Year Range", min_y, max_y, (min_y, max_y))
        
        all_fuels = df_ts.columns.tolist()
        # Pre-select main fuels for better default view
        defaults = [f for f in all_fuels if any(x in f for x in ['Coal', 'Gas', 'Petroleum'])]
        if not defaults: defaults = all_fuels[:5]
        
        selected_fuels = st.sidebar.multiselect("Select Fuels", all_fuels, default=defaults)
        
        df_filtered = df_ts.loc[start_year:end_year, selected_fuels]
        
        st.header(f"Emissions by Fuel Type Analysis ({start_year}-{end_year})")
        
        if not df_filtered.empty:
            
            # Vis 1: The Energy Transition Line Chart
            st.subheader("1. The Energy Transition (Trends)")
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            for f in df_filtered.columns:
                lw = 3 if any(x in f for x in ['Coal', 'Petroleum', 'Gas']) else 1.5
                alpha = 1.0 if lw==3 else 0.6
                ax1.plot(df_filtered.index, df_filtered[f], label=f, linewidth=lw, alpha=alpha)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)
            
            # Vis 2: Market Share (Stacked Area 100%)
            st.subheader("2. Fuel Share of Total Emissions (%)")
            row_sums = df_filtered.sum(axis=1).replace(0, np.nan)
            df_pct = df_filtered.div(row_sums, axis=0) * 100
            
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.stackplot(df_pct.index, df_pct.T.values, labels=df_pct.columns, alpha=0.85, cmap='inferno')
            ax2.set_ylabel("Percentage Share")
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig2)
            
            col1, col2 = st.columns(2)
            
            # Vis 3: Diverging Bar
            with col1:
                st.subheader(f"3. Change in Fuel Emissions")
                change = df_filtered.loc[end_year] - df_filtered.loc[start_year]
                change = change.sort_values()
                cols = ['green' if x < 0 else 'red' for x in change.values]
                fig3, ax3 = plt.subplots(figsize=(6, 5))
                ax3.barh(change.index, change.values, color=cols)
                ax3.axvline(0, color='black')
                ax3.set_xlabel("Change in MtCO2e")
                st.pyplot(fig3)
                
            # Vis 4: Heatmap
            with col2:
                st.subheader("4. Fuel Intensity Heatmap")
                fig4, ax4 = plt.subplots(figsize=(6, 5))
                sns.heatmap(df_filtered.T, cmap="YlOrBr", ax=ax4, cbar=False)
                st.pyplot(fig4)
                
            # --- CLASSIFICATION SECTION (REPLACED) ---
            st.markdown("---")
            st.header("Supervised Learning: Fuel Trend Classification")
            st.info("Classifies ALL fuel types as 'Increasing', 'Decreasing', or 'Stable' based on their historical trajectory.")
            
            rf_model = load_rf_model()
            if rf_model:
                # 1. USE ALL DATA (Ignore selection) to show full landscape
                df_all = df_ts.loc[start_year:end_year, :]
                
                # 2. Calculate Features
                feats = pd.DataFrame(index=df_all.columns)
                feats['Mean_Emissions'] = df_all.mean(axis=0)
                feats['Volatility'] = df_all.std(axis=0)
                feats['Max_Emissions'] = df_all.max(axis=0)
                
                # 3. Predict
                feats['Predicted Trend'] = rf_model.predict(feats)
                
                # 4. VIZ 1: Trend Distribution (Bar Chart)
                col_c1, col_c2 = st.columns(2)
                
                with col_c1:
                    st.subheader("Overview: How many fuels are Increasing?")
                    trend_counts = feats['Predicted Trend'].value_counts()
                    fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
                    colors = {'Decreasing': 'green', 'Increasing': 'red', 'Stable': 'gray'}
                    trend_counts.plot(kind='bar', color=[colors.get(x, 'blue') for x in trend_counts.index], ax=ax_bar)
                    ax_bar.set_ylabel("Count of Fuel Types")
                    st.pyplot(fig_bar)
                    
                # 5. VIZ 2: Watchlist Table
                with col_c2:
                    st.subheader("⚠️ Rising Emissions Watchlist")
                    rising_fuels = feats[feats['Predicted Trend'] == 'Increasing'].sort_values('Mean_Emissions', ascending=False)
                    if not rising_fuels.empty:
                        st.dataframe(rising_fuels[['Predicted Trend', 'Mean_Emissions']].style.format("{:.2f}"))
                    else:
                        st.success("Good news! No fuels are currently classified as 'Increasing' in this period.")
                        
                # 6. VIZ 3: Feature Importance (Keep as it helps explain model)
                st.subheader("What drives the classification?")
                importances = pd.Series(rf_model.feature_importances_, index=['Mean', 'Volatility', 'Max'])
                fig_imp, ax_imp = plt.subplots(figsize=(8, 3))
                importances.sort_values().plot(kind='barh', color='teal', ax=ax_imp)
                ax_imp.set_title("Feature Importance (Random Forest)")
                st.pyplot(fig_imp)

            else:
                st.warning("Classification Model not found. Run training notebook first.")
                
        else:
            st.warning("Select at least one fuel type.")