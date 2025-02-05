import streamlit as st
import pandas as pd
import warnings
import plotly.express as px
import datetime
import re

# Filter warnings for a clean output
warnings.filterwarnings("ignore")

# --- Page Configuration ---
st.set_page_config(page_title="YOY Dashboard", page_icon=":bar_chart:", layout="wide")

# --- Custom CSS (Optional) ---
st.markdown(
    """
    <style>
    .main {background-color: #f8f9fa;}
    .sidebar .sidebar-content {background-color: #343a40; color: white;}
    /* Reduce gap between columns */
    .css-1lcbmhc.e1tzin5v3 { gap: 0.5rem !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Title and Logo ---
col1, col2 = st.columns([3, 1])
with col1:
    st.title("YOY Dashboard")
with col2:
    st.image("C:\\Users\\MaciejWach\\OneDrive - AFG Media Ltd\\Desktop\\streamlit\\logo.png", width=300)

# --- File Upload ---
uploaded_sales = st.file_uploader("Upload Sales Revenue CSV", type=["csv"])
uploaded_skus = st.file_uploader("Upload SKU Data", type=["xlsx"])
if not uploaded_sales or not uploaded_skus:
    st.warning("Please upload both the Sales Revenue CSV and SKU Data Excel file to proceed.")
    st.stop()

# =============================================================================
# DATA LOADING & CLEANING FUNCTIONS
# =============================================================================
@st.cache_data(show_spinner=True)
def load_data(sales_file, sku_file):
    sales_df = pd.read_csv(sales_file, sep=';', engine='python')
    sku_df = pd.read_excel(sku_file)
    return sales_df, sku_df

def clean_sap_data(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(inplace=True)
    exclude_list = [
        'Delivery', 'MC Historic Kids', 'Fun Shack Adults', 'Hot 50 Expansion Adults',
        'MC Inflatable Adults', 'MC Historic Adults', 'MC Piggyback Adults',
        'Fun Shack Accessories', 'MC Hot Kids'
    ]
    if 'Unnamed: 8' in df.columns:
        df = df[~df['Unnamed: 8'].isin(exclude_list)]
    if 'Customer' in df.columns:
        df = df[df['Customer'] != 'Sample orders - R&A']
    df = df.copy()
    df.rename(columns={
        'Unnamed: 6': 'Product Name',
        'Unnamed: 1': 'Revenue Region',
        'Customer': 'Sales Channel',
        'Amount Company Currency': 'Sales Value (£)',
        'Product': 'Product SKU',
        'Actual Entry Quantity': 'Order Quantity',
        'Amount in Transaction Currency': 'Sales Value in Transaction Currency',
        'Posting Date': 'Date'
    }, inplace=True)
    drop_cols = ['Unnamed: 8', 'Product Category', 'Unnamed: 4', 'G/L Account', 'Product Name']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    def clean_quantity(value):
        if isinstance(value, str):
            match = re.search(r'-?([\d,]+)', value)
            if match:
                return match.group(1)
        elif isinstance(value, (int, float)):
            return abs(value)
        return value
    def extract_currency(value):
        if isinstance(value, str):
            match = re.search(r'([A-Z]{3})$', value)
            if match:
                return match.group(1)
        return None
    df['Original Currency'] = df['Sales Value in Transaction Currency'].apply(extract_currency)
    df['Sales Value in Transaction Currency'] = df['Sales Value in Transaction Currency'].apply(clean_quantity)
    df['Order Quantity'] = df['Order Quantity'].apply(clean_quantity)
    df['Sales Value (£)'] = df['Sales Value (£)'].apply(clean_quantity)
    df['Order Quantity'] = df['Order Quantity'].astype(str).str.split(',').str[0]
    df['Order Quantity'] = pd.to_numeric(df['Order Quantity'], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Year'] = df['Date'].dt.year
    df['Sales Value (£)'] = df['Sales Value (£)'].astype(str).str.replace(',', '.')
    df['Sales Value in Transaction Currency'] = df['Sales Value in Transaction Currency'].astype(str).str.replace(',', '.')
    df['Sales Value (£)'] = pd.to_numeric(df['Sales Value (£)'], errors='coerce')
    df['Sales Value in Transaction Currency'] = pd.to_numeric(df['Sales Value in Transaction Currency'], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
    return df

def merge_sku_data(sap_df: pd.DataFrame, sku_df: pd.DataFrame) -> pd.DataFrame:
    sku_columns = ['Product Code', 'Consumer Description', 'Product', 'Design', 'Listing']
    merged_df = sap_df.merge(
        sku_df[sku_columns],
        how='left',
        left_on='Product SKU',
        right_on='Product Code',
        indicator=True
    )
    for col in ['Consumer Description', 'Product', 'Design', 'Listing']:
        merged_df.loc[merged_df['_merge'] == 'left_only', col] = 'Unrecognised'
    merged_df.drop(columns=['_merge', 'Product Code'], inplace=True)
    cols = list(merged_df.columns)
    if 'Product SKU' in cols:
        sku_index = cols.index('Product SKU')
        new_cols = ['Consumer Description', 'Product', 'Design', 'Listing']
        for col in new_cols:
            if col in cols:
                cols.remove(col)
        for i, col in enumerate(new_cols, start=sku_index + 1):
            cols.insert(i, col)
        merged_df = merged_df[cols]
    return merged_df

def clean_sales_revenue(value):
    if isinstance(value, str):
        if "UK & ROW" in value:
            return "UK"
        elif "US" in value:
            return "US"
    return value

def simplify_order_source(value):
    if isinstance(value, str):
        if 'FBM-US-' in value or 'FBM-US-Other' in value or 'FBA-US-' in value or 'FBA-US-Other' in value:
            return 'RA Amazon US'
        elif ('Website-Bundle-US-' in value or 'Website-US-' in value or 
              'RA Website US-Other' in value or 'RA Website-Bundle-US' in value or 
              'RA Website-US' in value):
            return 'RA Website US'
        elif 'eBay-US-' in value:
            return 'eBay US'
        elif 'FBA-UK-' in value or 'RA FBA-UK' in value or 'RA FBM-UK' in value:
            return 'RA Amazon UK'
        elif 'RA-Website-Bundle-UK' in value or 'RA Website-UK' in value or 'Website UK' in value:
            return 'RA Website UK'
        elif 'RA ebay-UK' in value or 'Morph eBay-UK' in value:
            return 'UK eBay'
        elif 'RA Website-AU' in value or 'Website AU' in value:
            return 'RA Website AU'
        elif 'RA FBA-CA' in value:
            return 'RA Amazon CA'
        elif 'RA FBA-MX' in value:
            return 'RA Amazon MX'
        elif 'RA FBA-DE' in value:
            return 'RA Amazon DE'
        elif 'RA FBA-ES' in value:
            return 'RA Amazon ES'
        elif 'RA FBA-FR' in value:
            return 'RA Amazon FR'
        elif 'RA FBA-IT' in value:
            return 'RA Amazon IT'
        elif 'RA FBA-NL' in value:
            return 'RA Amazon NL'
        elif 'RA FBA-SE' in value:
            return 'RA Amazon SE'
        elif 'RA Tiktok-UK' in value or 'RA Tiktok UK' in value:
            return 'RA Tiktok UK'
        elif 'RA Tiktok-US' in value or 'RA Tiktok US' in value:
            return 'RA Tiktok US'
    return value

# ---------------------------
# LOAD, CLEAN & MERGE DATA
# ---------------------------
sales_df, sku_df = load_data(uploaded_sales, uploaded_skus)
sap_df = clean_sap_data(sales_df)
df = merge_sku_data(sap_df, sku_df)

if 'Revenue Region' in df.columns:
    df['Revenue Region'] = df['Revenue Region'].apply(clean_sales_revenue)
if 'Sales Channel' in df.columns:
    df['Sales Channel'] = df['Sales Channel'].apply(simplify_order_source)

st.subheader("Cleaned Data Preview")
st.dataframe(df.head())

# =============================================================================
# DASHBOARD HELPER FUNCTIONS (KPIs, Charts, etc.)
# =============================================================================
@st.cache_data(show_spinner=False)
def get_quarter(week):
    if 1 <= week <= 13:
        return "Q1"
    elif 14 <= week <= 26:
        return "Q2"
    elif 27 <= week <= 39:
        return "Q3"
    elif 40 <= week <= 52:
        return "Q4"
    else:
        return None

def format_currency(value):
    return f"£{value:,.2f}"

@st.cache_data(show_spinner=False)
def preprocess_data(data):
    required_cols = {"Week", "Year", "Sales Value (£)"}
    if not required_cols.issubset(set(data.columns)):
        st.error(f"Dataset is missing one or more required columns: {required_cols}")
        st.stop()
    if "Week" in data.columns:
        data["Quarter"] = data["Week"].apply(get_quarter)
    return data

def week_monday(row):
    try:
        return datetime.date.fromisocalendar(int(row["Year"]), int(row["Week"]), 1)
    except Exception:
        return None

def create_yoy_trends_chart(data, selected_years, selected_quarters, selected_channels=None, selected_listings=None, selected_products=None):
    filtered = data.copy()
    if selected_years:
        filtered = filtered[filtered["Year"].isin(selected_years)]
    if selected_quarters:
        filtered = filtered[filtered["Quarter"].isin(selected_quarters)]
    if selected_channels and len(selected_channels) > 0:
        filtered = filtered[filtered["Sales Channel"].isin(selected_channels)]
    if selected_listings and len(selected_listings) > 0:
        filtered = filtered[filtered["Listing"].isin(selected_listings)]
    if selected_products and len(selected_products) > 0:
        filtered = filtered[filtered["Product"].isin(selected_products)]
    
    weekly_rev = (
        filtered.groupby(["Year", "Week"])["Sales Value (£)"]
        .sum().reset_index().sort_values(by=["Year", "Week"])
    )
    weekly_rev["RevenueK"] = weekly_rev["Sales Value (£)"] / 1000
    if not filtered.empty:
        min_week, max_week = int(filtered["Week"].min()), int(filtered["Week"].max())
    else:
        min_week, max_week = 1, 52

    fig = px.line(
        weekly_rev,
        x="Week",
        y="Sales Value (£)",
        color="Year",
        markers=True,
        title="Weekly Revenue Trends by Year",
        labels={"Sales Value (£)": "Revenue (£)"},
        custom_data=["RevenueK"]
    )
    fig.update_traces(hovertemplate="Week: %{x}<br>Revenue: %{customdata[0]:.1f}K")
    fig.update_layout(xaxis=dict(tickmode="linear", range=[min_week, max_week]), margin=dict(t=50, b=50))
    return fig

def create_sales_channel_chart(data, selected_years, selected_quarter):
    filtered = data.copy()
    if selected_years:
        filtered = filtered[filtered["Year"].isin(selected_years)]
    if selected_quarter != "All Quarters":
        filtered = filtered[filtered["Quarter"] == selected_quarter]
    
    revenue = (filtered.groupby("Sales Channel")["Sales Value (£)"]
               .sum().reset_index().sort_values(by="Sales Value (£)", ascending=False))
    
    fig = px.bar(
        revenue,
        x="Sales Channel",
        y="Sales Value (£)",
        text="Sales Value (£)",
        title="Revenue by Sales Channel",
        color_discrete_sequence=["#db53bb"],
        labels={"Sales Value (£)": "Revenue (£)"}
    )
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig.update_layout(margin=dict(t=50, b=100))
    return fig

def create_listings_chart(data, selected_years, selected_quarters, selected_listings):
    filtered = data.copy()
    if selected_years:
        filtered = filtered[filtered["Year"].isin(selected_years)]
    if selected_quarters and "All Quarters" not in selected_quarters:
        filtered = filtered[filtered["Quarter"].isin(selected_quarters)]
    filtered = filtered[filtered["Listing"].isin(selected_listings)]
    
    if len(selected_years) > 1:
        weekly_listing = (filtered.groupby(["Listing", "Year", "Week"])
                          .agg({"Sales Value (£)": "sum", "Order Quantity": "sum"})
                          .reset_index().sort_values(by=["Listing", "Year", "Week"]))
        weekly_listing.rename(columns={"Order Quantity": "UnitsSold"}, inplace=True)
        weekly_listing["RevenueK"] = weekly_listing["Sales Value (£)"] / 1000
        weekly_listing["Listing_Year"] = weekly_listing["Listing"] + " (" + weekly_listing["Year"].astype(str) + ")"
        color_field = "Listing_Year"
    else:
        weekly_listing = (filtered.groupby(["Listing", "Week"])
                          .agg({"Sales Value (£)": "sum", "Order Quantity": "sum"})
                          .reset_index().sort_values(by=["Listing", "Week"]))
        weekly_listing.rename(columns={"Order Quantity": "UnitsSold"}, inplace=True)
        weekly_listing["RevenueK"] = weekly_listing["Sales Value (£)"] / 1000
        color_field = "Listing"
    
    if selected_quarters and "All Quarters" not in selected_quarters and not filtered.empty:
        min_week, max_week = int(filtered["Week"].min()), int(filtered["Week"].max())
    else:
        min_week, max_week = 1, 52
    
    fig = px.line(
        weekly_listing,
        x="Week",
        y="Sales Value (£)",
        color=color_field,
        markers=True,
        title="Weekly Revenue by Listing",
        labels={"Sales Value (£)": "Revenue (£)"},
        custom_data=["RevenueK", "UnitsSold"]
    )
    fig.update_traces(hovertemplate="Week: %{x}<br>Revenue: %{customdata[0]:.1f}K<br>Units Sold: %{customdata[1]}")
    fig.update_layout(xaxis=dict(tickmode="linear", range=[min_week, max_week]), margin=dict(t=50, b=50))
    return fig

def create_pivot_table(data, selected_years, sort_order):
    filtered = data.copy()
    if selected_years:
        filtered = filtered[filtered["Year"].isin(selected_years)]
    
    pivot = pd.pivot_table(
        filtered,
        values="Sales Value (£)",
        index="Listing",
        columns="Week",
        aggfunc="sum",
        fill_value=0
    )
    pivot["Total Revenue"] = pivot.sum(axis=1)
    pivot = pivot.round(0)
    
    if sort_order == "Highest to Lowest":
        pivot = pivot.sort_values(by="Total Revenue", ascending=False)
    else:
        pivot = pivot.sort_values(by="Total Revenue", ascending=True)
    return pivot

def create_sku_line_chart(data, sku_text, selected_years, selected_quarter):
    if "Product SKU" not in data.columns:
        st.error("The dataset does not contain a 'Product SKU' column.")
        st.stop()
    filtered = data.copy()
    filtered = filtered[filtered["Product SKU"].str.contains(sku_text, case=False, na=False)]
    if selected_years:
        filtered = filtered[filtered["Year"].isin(selected_years)]
    if selected_quarter != "All Quarters":
        filtered = filtered[filtered["Quarter"] == selected_quarter]
    if filtered.empty:
        st.warning("No data available for the entered SKU and filters.")
        return None
    weekly_sku = (filtered.groupby(["Year", "Week"])["Sales Value (£)"]
                  .sum().reset_index().sort_values(by=["Year", "Week"]))
    weekly_sku["RevenueK"] = weekly_sku["Sales Value (£)"] / 1000
    if selected_quarter != "All Quarters" and not filtered.empty:
        min_week, max_week = int(filtered["Week"].min()), int(filtered["Week"].max())
    else:
        min_week, max_week = 1, 52
    fig = px.line(
        weekly_sku,
        x="Week",
        y="Sales Value (£)",
        color="Year",
        markers=True,
        title=f"Weekly Revenue Trends for SKU matching: '{sku_text}'",
        labels={"Sales Value (£)": "Revenue (£)"},
        custom_data=["RevenueK"]
    )
    fig.update_traces(hovertemplate="Week: %{x}<br>Revenue: %{customdata[0]:.1f}K")
    fig.update_layout(xaxis=dict(tickmode="linear", range=[min_week, max_week]), margin=dict(t=50, b=50))
    return fig

def create_daily_price_chart(data, listing, selected_years, selected_quarters, selected_channels):
    if "Date" not in data.columns:
        st.error("The dataset does not contain a 'Date' column required for daily price analysis.")
        return None
    df_listing = data[(data["Listing"] == listing) & (data["Year"].isin(selected_years))].copy()
    if selected_quarters:
        df_listing = df_listing[df_listing["Quarter"].isin(selected_quarters)]
    if selected_channels:
        df_listing = df_listing[df_listing["Sales Channel"].isin(selected_channels)]
    if df_listing.empty:
        st.warning(f"No data available for {listing} for the selected filters.")
        return None
    df_listing["Date"] = pd.to_datetime(df_listing["Date"])
    grouped = df_listing.groupby([df_listing["Date"].dt.date, "Year"]).agg({
        "Sales Value (£)": "sum",
        "Order Quantity": "sum"
    }).reset_index()
    grouped.rename(columns={"Date": "Date"}, inplace=True)
    grouped["Average Price"] = grouped["Sales Value (£)"] / grouped["Order Quantity"]
    grouped["Date"] = pd.to_datetime(grouped["Date"])
    dfs = []
    for yr in selected_years:
        df_year = grouped[grouped["Year"] == yr].copy()
        if df_year.empty:
            continue
        df_year["Day"] = df_year["Date"].dt.dayofyear
        start_day = int(df_year["Day"].min())
        end_day = int(df_year["Day"].max())
        df_year = df_year.set_index("Day").reindex(range(start_day, end_day + 1))
        df_year.index.name = "Day"
        df_year["Average Price"] = df_year["Average Price"].interpolate(method="linear")
        df_year["Year"] = yr
        df_year = df_year.reset_index()
        dfs.append(df_year)
    if not dfs:
        st.warning("No data available after processing for the selected filters.")
        return None
    combined = pd.concat(dfs, ignore_index=True)
    fig = px.line(combined, x="Day", y="Average Price", color="Year",
                  title=f"Daily Average Price for {listing}",
                  labels={"Day": "Day of Year", "Average Price": "Average Price (£)"},
                  color_discrete_sequence=px.colors.qualitative.Set1)
    fig.update_layout(margin=dict(t=50, b=50))
    return fig

# =============================================================================
# MAIN CODE: Preprocess Data & Dashboard Tabs
# =============================================================================
df = preprocess_data(df)
available_years = sorted(df["Year"].dropna().unique())
if not available_years:
    st.error("No year data available.")
    st.stop()
current_year = available_years[-1]
if len(available_years) >= 2:
    prev_year = available_years[-2]
    yoy_default_years = [prev_year, current_year]
else:
    yoy_default_years = [current_year]
default_current_year = [current_year]

tabs = st.tabs([
    "KPIs", 
    "YOY Trends", 
    "Daily Prices", 
    "SKU Trends", 
    "Sales Channel", 
    "Pivot Table", 
    "Listings"
])

# -----------------------------------------
# Tab 1: KPIs
# -----------------------------------------
with tabs[0]:
    st.header("Key Performance Indicators")
    with st.expander("KPI Filters", expanded=True):
        today = datetime.date.today()
        if today.weekday() != 6:
            last_full_week_end = today - datetime.timedelta(days=today.weekday() + 1)
        else:
            last_full_week_end = today
        default_week = last_full_week_end.isocalendar()[1]
        available_weeks = sorted(df["Week"].dropna().unique())
        default_index = available_weeks.index(default_week) if default_week in available_weeks else 0
        selected_week = st.selectbox(
            "Select Week for KPI Calculation",
            options=available_weeks,
            index=default_index,
            key="kpi_week",
            help="Select the week to calculate KPIs for. (Defaults to the last full week)"
        )
    kpi_data = df[df["Week"] == selected_week]
    kpi_summary = kpi_data.groupby("Year")["Sales Value (£)"].sum()
    all_years = sorted(df["Year"].dropna().unique())
    kpi_summary = kpi_summary.reindex(all_years, fill_value=0)
    pct_change = kpi_summary.pct_change() * 100
    kpi_cols = st.columns(len(all_years))
    for idx, year in enumerate(all_years):
        with kpi_cols[idx]:
            st.subheader(f"Year {year}")
            if kpi_summary[year] == 0:
                st.write("Revenue: N/A")
            else:
                st.metric(
                    label=f"Week {selected_week} Revenue",
                    value=format_currency(kpi_summary[year]),
                    delta=f"{pct_change[year]:.2f}%" if idx > 0 and not pd.isna(pct_change[year]) else ""
                )
    st.markdown("---")
    st.subheader("Top Gainers and Losers for Week " + str(selected_week) + " (" + str(current_year) + ")")
    current_year_data = df[df["Year"] == current_year]
    week_data_current = current_year_data[current_year_data["Week"] == selected_week]
    if selected_week > 1:
        previous_week = selected_week - 1
        week_data_prev = current_year_data[current_year_data["Week"] == previous_week]
    else:
        previous_years = [year for year in available_years if year < current_year]
        if previous_years:
            prev_year = max(previous_years)
            week_data_prev = df[(df["Year"] == prev_year) & (df["Week"] == 52)]
        else:
            week_data_prev = pd.DataFrame(columns=current_year_data.columns)
    rev_current = week_data_current.groupby("Listing")["Sales Value (£)"].sum()
    rev_previous = week_data_prev.groupby("Listing")["Sales Value (£)"].sum()
    combined = pd.concat([rev_current, rev_previous], axis=1, keys=["Current", "Previous"]).fillna(0)
    def compute_pct_change(row):
        if row["Previous"] == 0:
            return None
        return ((row["Current"] - row["Previous"]) / row["Previous"]) * 100
    combined["pct_change"] = combined.apply(compute_pct_change, axis=1)
    num_items = 3
    top_gainers = combined[combined["pct_change"].notnull()].sort_values("pct_change", ascending=False).head(num_items)
    top_losers = combined[combined["pct_change"].notnull()].sort_values("pct_change", ascending=True).head(num_items)
    def format_change(pct):
        if pct is None:
            return "<span style='color:gray;'>N/A</span>"
        if pct > 0:
            return f"<span style='color:green;'>↑ {pct:.2f}%</span>"
        elif pct < 0:
            return f"<span style='color:red;'>↓ {abs(pct):.2f}%</span>"
        else:
            return f"<span style='color:gray;'>→ {pct:.2f}%</span>"
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("<div style='text-align:center;'><h4>Top Gainers</h4></div>", unsafe_allow_html=True)
        if top_gainers.empty:
            st.info("No gainers data available for this week.")
        else:
            for listing, row in top_gainers.iterrows():
                change_html = format_change(row["pct_change"])
                st.markdown(
                    f"""
                    <div style="background:#94f7bb; color:black; padding:10px; border-radius:8px; margin-bottom:8px; text-align:center;">
                        <strong>{listing}</strong><br>
                        {change_html}<br>{format_currency(row['Current'])}
                    </div>
                    """, unsafe_allow_html=True
                )
    with col2:
        st.markdown("<div style='text-align:center;'><h4>Top Losers</h4></div>", unsafe_allow_html=True)
        if top_losers.empty:
            st.info("No losers data available for this week.")
        else:
            for listing, row in top_losers.iterrows():
                change_html = format_change(row["pct_change"])
                st.markdown(
                    f"""
                    <div style="background:#fcb8b8; color:black; padding:10px; border-radius:8px; margin-bottom:8px; text-align:center;">
                        <strong>{listing}</strong><br>
                        {change_html}<br>{format_currency(row['Current'])}
                    </div>
                    """, unsafe_allow_html=True
                )
    st.markdown("---")
    st.subheader("Top and Bottom Performing Listings")
    current_week_data = current_year_data[current_year_data["Week"] == selected_week]
    current_group = current_week_data.groupby("Listing").agg({
        "Sales Value (£)": "sum",
        "Order Quantity": "sum"
    }).reset_index()
    current_group["Avg Order Price"] = current_group["Sales Value (£)"] / current_group["Order Quantity"]
    if selected_week > 1:
        previous_week_data = current_year_data[current_year_data["Week"] == (selected_week - 1)]
    else:
        previous_years = [year for year in available_years if year < current_year]
        if previous_years:
            prev_year = max(previous_years)
            previous_week_data = df[(df["Year"] == prev_year) & (df["Week"] == 52)]
        else:
            previous_week_data = pd.DataFrame(columns=current_year_data.columns)
    if not previous_week_data.empty:
        previous_group = previous_week_data.groupby("Listing").agg({
            "Sales Value (£)": "sum",
            "Order Quantity": "sum"
        }).reset_index()
        previous_group["Avg Order Price Prev"] = previous_group["Sales Value (£)"] / previous_group["Order Quantity"]
    else:
        previous_group = pd.DataFrame(columns=["Listing", "Sales Value (£)", "Order Quantity", "Avg Order Price Prev"])
    merged = pd.merge(current_group, previous_group[["Listing", "Avg Order Price Prev"]], on="Listing", how="left")
    merged["Change in Avg Order Price (%)"] = ((merged["Avg Order Price"] - merged["Avg Order Price Prev"]) / merged["Avg Order Price Prev"]) * 100
    merged["Change in Avg Order Price (%)"] = merged["Change in Avg Order Price (%)"].fillna(0)
    top_performers = merged.sort_values("Sales Value (£)", ascending=False).head(num_items)
    bottom_performers = merged.sort_values("Sales Value (£)", ascending=True).head(num_items)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div style='text-align:center;'><h4>Top Performing Listings</h4></div>", unsafe_allow_html=True)
        st.dataframe(top_performers[["Listing", "Sales Value (£)", "Avg Order Price", "Change in Avg Order Price (%)"]])
    with col2:
        st.markdown("<div style='text-align:center;'><h4>Bottom Performing Listings</h4></div>", unsafe_allow_html=True)
        st.dataframe(bottom_performers[["Listing", "Sales Value (£)", "Avg Order Price", "Change in Avg Order Price (%)"]])

# -----------------------------------------
# Tab 2: YOY Trends
# -----------------------------------------
with tabs[1]:
    st.header("YOY Weekly Revenue Trends")
    with st.expander("Chart Filters", expanded=True):
        yoy_years = st.multiselect("Select Year(s)", options=available_years, default=yoy_default_years, key="yoy_years", help="Default is previous and current year.")
        selected_quarters = st.multiselect("Select Quarter(s)", options=["Q1", "Q2", "Q3", "Q4"], default=["Q1", "Q2", "Q3", "Q4"], key="yoy_quarters", help="Select one or more quarters to filter by.")
        selected_channels = st.multiselect("Select Sales Channel(s)", options=sorted(df["Sales Channel"].dropna().unique()), default=[], key="yoy_channels", help="Select one or more channels to filter. If empty, data for all channels is shown.")
        selected_listings = st.multiselect("Select Listing(s)", options=sorted(df["Listing"].dropna().unique()), default=[], key="yoy_listings", help="Select one or more listings to filter. If empty, data for all listings is shown.")
        if selected_listings:
            product_options = sorted(df[df["Listing"].isin(selected_listings)]["Product"].dropna().unique())
        else:
            product_options = sorted(df["Product"].dropna().unique())
        selected_products = st.multiselect("Select Product(s)", options=product_options, default=[], key="yoy_products", help="Select one or more products to filter. If empty, data for all products is shown.")
    fig_yoy = create_yoy_trends_chart(df, yoy_years, selected_quarters, selected_channels, selected_listings, selected_products)
    st.plotly_chart(fig_yoy, use_container_width=True)
    
    st.markdown("### Revenue Summary by Listing")
    st.markdown("")
    today = datetime.date.today()
    if today.weekday() != 6:
        days_since_sunday = today.weekday() + 1
        last_complete_week_end = today - datetime.timedelta(days=days_since_sunday)
    else:
        last_complete_week_end = today
    last_complete_week_start = last_complete_week_end - datetime.timedelta(days=6)
    df_revenue = df.copy()
    df_revenue["Year"] = df_revenue["Year"].astype(int)
    df_revenue["Week"] = df_revenue["Week"].astype(int)
    df_revenue_current = df_revenue[df_revenue["Year"] == current_year].copy()
    df_revenue_current["Week_Monday"] = df_revenue_current.apply(week_monday, axis=1)
    df_revenue_current["Week_Sunday"] = df_revenue_current["Week_Monday"].apply(lambda d: d + datetime.timedelta(days=6) if d else None)
    df_full_weeks_current = df_revenue_current[df_revenue_current["Week_Sunday"] <= last_complete_week_end].copy()
    unique_weeks_current = (
        df_full_weeks_current.groupby(["Year", "Week"])
        .first()
        .reset_index()[["Year", "Week", "Week_Sunday"]]
    ).sort_values("Week_Sunday")
    if unique_weeks_current.empty:
        st.info("Not enough complete week data in the current year to build the revenue summary table.")
    else:
        last_complete_week_row_current = unique_weeks_current.iloc[-1]
        last_week_tuple_current = (last_complete_week_row_current["Year"], last_complete_week_row_current["Week"])
        last_4_weeks_current = unique_weeks_current.tail(4)
        last_4_week_tuples_current = set(last_4_weeks_current[["Year", "Week"]].apply(tuple, axis=1))
        rev_last_4_current = (
            df_full_weeks_current[df_full_weeks_current.apply(lambda row: (row["Year"], row["Week"]) in last_4_week_tuples_current, axis=1)]
            .groupby("Listing")["Sales Value (£)"]
            .sum()
            .rename("Last 4 Weeks Revenue (Current Year)")
        )
        rev_last_1_current = (
            df_full_weeks_current[df_full_weeks_current.apply(lambda row: (row["Year"], row["Week"]) == last_week_tuple_current, axis=1)]
            .groupby("Listing")["Sales Value (£)"]
            .sum()
            .rename("Last Week Revenue (Current Year)")
        )
        if len(available_years) >= 2:
            last_year = available_years[-2]
            reference_date_last_year = datetime.date(last_year, 12, 31)
            if reference_date_last_year.weekday() != 6:
                days_since_sunday = reference_date_last_year.weekday() + 1
                last_complete_week_end_last_year = reference_date_last_year - datetime.timedelta(days=days_since_sunday)
            else:
                last_complete_week_end_last_year = reference_date_last_year
            last_complete_week_start_last_year = last_complete_week_end_last_year - datetime.timedelta(days=6)
            df_revenue_last_year = df_revenue[df_revenue["Year"] == last_year].copy()
            df_revenue_last_year["Week_Monday"] = df_revenue_last_year.apply(week_monday, axis=1)
            df_revenue_last_year["Week_Sunday"] = df_revenue_last_year["Week_Monday"].apply(lambda d: d + datetime.timedelta(days=6) if d else None)
            df_full_weeks_last_year = df_revenue_last_year[df_revenue_last_year["Week_Sunday"] <= last_complete_week_end_last_year].copy()
            unique_weeks_last_year = (
                df_full_weeks_last_year.groupby(["Year", "Week"])
                .first()
                .reset_index()[["Year", "Week", "Week_Sunday"]]
            ).sort_values("Week_Sunday")
            if unique_weeks_last_year.empty:
                rev_last_4_last_year = pd.Series(dtype=float)
                rev_last_1_last_year = pd.Series(dtype=float)
            else:
                last_complete_week_row_last_year = unique_weeks_last_year.iloc[-1]
                last_week_tuple_last_year = (last_complete_week_row_last_year["Year"], last_complete_week_row_last_year["Week"])
                last_4_weeks_last_year = unique_weeks_last_year.tail(4)
                last_4_week_tuples_last_year = set(last_4_weeks_last_year[["Year", "Week"]].apply(tuple, axis=1))
                rev_last_4_last_year = (
                    df_full_weeks_last_year[df_full_weeks_last_year.apply(lambda row: (row["Year"], row["Week"]) in last_4_week_tuples_last_year, axis=1)]
                    .groupby("Listing")["Sales Value (£)"]
                    .sum()
                    .rename("Last 4 Weeks Revenue (Last Year)")
                )
                rev_last_1_last_year = (
                    df_full_weeks_last_year[df_full_weeks_last_year.apply(lambda row: (row["Year"], row["Week"]) == last_week_tuple_last_year, axis=1)]
                    .groupby("Listing")["Sales Value (£)"]
                    .sum()
                    .rename("Last Week Revenue (Last Year)")
                )
        else:
            rev_last_4_last_year = pd.Series(dtype=float)
            rev_last_1_last_year = pd.Series(dtype=float)
        all_listings_current = pd.Series(sorted(df_revenue_current["Listing"].unique()), name="Listing")
        revenue_summary = pd.DataFrame(all_listings_current).set_index("Listing")
        revenue_summary = revenue_summary.join(rev_last_4_current, how="left").join(rev_last_1_current, how="left")
        revenue_summary = revenue_summary.fillna(0)
        revenue_summary = revenue_summary.join(rev_last_4_last_year, how="left").join(rev_last_1_last_year, how="left")
        revenue_summary = revenue_summary.fillna(0).reset_index()
        revenue_summary = revenue_summary[["Listing", 
                                           "Last 4 Weeks Revenue (Current Year)", "Last 4 Weeks Revenue (Last Year)",
                                           "Last Week Revenue (Current Year)", "Last Week Revenue (Last Year)"]]
        # --- Compute Percentage Changes ---
        revenue_summary["4W % Change"] = revenue_summary.apply(
            lambda row: ((row["Last 4 Weeks Revenue (Current Year)"] - row["Last 4 Weeks Revenue (Last Year)"]) / row["Last 4 Weeks Revenue (Last Year)"] * 100)
                        if row["Last 4 Weeks Revenue (Last Year)"] != 0 else None,
            axis=1
        )
        revenue_summary["1W % Change"] = revenue_summary.apply(
            lambda row: ((row["Last Week Revenue (Current Year)"] - row["Last Week Revenue (Last Year)"]) / row["Last Week Revenue (Last Year)"] * 100)
                        if row["Last Week Revenue (Last Year)"] != 0 else None,
            axis=1
        )
        # --- Sorting Toggle Button with Dynamic Label ---
        def toggle_sort():
            st.session_state.sort_order = "asc" if st.session_state.sort_order == "desc" else "desc"
        if "sort_order" not in st.session_state:
            st.session_state.sort_order = "desc"
        button_label = "Sort: High ↔ Low" if st.session_state.sort_order == "desc" else "Sort: Low ↔ High"
        st.button(button_label, on_click=toggle_sort)
        if st.session_state.sort_order == "desc":
            revenue_summary = revenue_summary.sort_values("Last 4 Weeks Revenue (Current Year)", ascending=False)
            sort_text = "Highest to Lowest"
        else:
            revenue_summary = revenue_summary.sort_values("Last 4 Weeks Revenue (Current Year)", ascending=True)
            sort_text = "Lowest to Highest"
        def combine_revenue_pct(row, revenue_col, pct_col):
            revenue = row[revenue_col]
            pct = row[pct_col]
            revenue_str = f"£{revenue:,.0f}"
            if pd.isna(pct) or revenue == 0:
                return revenue_str
            else:
                if pct > 0:
                    pct_str = f"<span style='color:green;'>↑ {pct:.0f}%</span>"
                elif pct < 0:
                    pct_str = f"<span style='color:red;'>↓ {abs(pct):.0f}%</span>"
                else:
                    pct_str = f"<span style='color:gray;'>→ {pct:.0f}%</span>"
                return f"{revenue_str}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;({pct_str})"
        revenue_summary["Last 4 Weeks (Current)"] = revenue_summary.apply(
            lambda row: combine_revenue_pct(row, "Last 4 Weeks Revenue (Current Year)", "4W % Change"),
            axis=1
        )
        revenue_summary["Last Week (Current)"] = revenue_summary.apply(
            lambda row: combine_revenue_pct(row, "Last Week Revenue (Current Year)", "1W % Change"),
            axis=1
        )
        revenue_summary = revenue_summary[[
            "Listing",
            "Last 4 Weeks (Current)",
            "Last 4 Weeks Revenue (Last Year)",
            "Last Week (Current)",
            "Last Week Revenue (Last Year)"
        ]]
        st.markdown(f"**Current Year Revenue Summary Sorted: {sort_text} (by Last 4 Weeks Revenue)**")
        st.write(
            revenue_summary.style.format({
                "Last 4 Weeks Revenue (Last Year)": "£{:,.0f}",
                "Last Week Revenue (Last Year)": "£{:,.0f}"
            }).to_html(escape=False),
            unsafe_allow_html=True
        )

# -----------------------------------------
# Tab 3: Daily Prices
# -----------------------------------------
with tabs[2]:
    st.header("Daily Prices for Top Listings")
    with st.expander("Daily Price Filters", expanded=True):
        default_daily_years = [year for year in available_years if year in (2024, 2025)]
        if not default_daily_years:
            default_daily_years = [current_year]
        selected_daily_years = st.multiselect("Select Year(s) to compare", options=available_years, default=default_daily_years, key="daily_years", help="Default shows 2024 and 2025 if available.")
        selected_daily_quarters = st.multiselect("Select Quarter(s)", options=["Q1", "Q2", "Q3", "Q4"], default=["Q1", "Q2", "Q3", "Q4"], key="daily_quarters", help="Select one or more quarters to filter.")
        selected_daily_channels = st.multiselect("Select Sales Channel(s)", options=sorted(df["Sales Channel"].dropna().unique()), default=[], key="daily_channels", help="Select one or more sales channels to filter the daily price data.")
    main_listings = ["Pattern Pants", "Pattern Shorts", "Solid Pants", "Solid Shorts", "Patterned Polos"]
    for listing in main_listings:
        st.subheader(listing)
        fig_daily = create_daily_price_chart(df, listing, selected_daily_years, selected_daily_quarters, selected_daily_channels)
        if fig_daily:
            st.plotly_chart(fig_daily, use_container_width=True)
    st.markdown("## Daily Prices Comparison")
    with st.expander("Comparison Chart Filters", expanded=True):
        comp_years = st.multiselect("Select Year(s)", options=available_years, default=default_daily_years, key="comp_years", help="Select the year(s) for the comparison chart.")
        comp_channels = st.multiselect("Select Sales Channel(s)", options=sorted(df["Sales Channel"].dropna().unique()), default=[], key="comp_channels", help="Select the sales channel(s) for the comparison chart.")
        comp_listing = st.selectbox("Select Listing", options=sorted(df["Listing"].dropna().unique()), key="comp_listing", help="Select a listing for daily prices comparison.")
    all_quarters = ["Q1", "Q2", "Q3", "Q4"]
    fig_comp = create_daily_price_chart(df, comp_listing, comp_years, all_quarters, comp_channels)
    if fig_comp:
        st.plotly_chart(fig_comp, use_container_width=True)

# -----------------------------------------
# Tab 4: SKU Trends
# -----------------------------------------
with tabs[3]:
    st.header("SKU Trends")
    if "Product SKU" not in df.columns:
        st.error("The dataset does not contain a 'Product SKU' column.")
    else:
        with st.expander("SKU Chart Filters", expanded=True):
            sku_text = st.text_input("Enter Product SKU", value="", key="sku_input", help="Enter a SKU (or part of it) to display its weekly revenue trends.")
            sku_years = st.multiselect("Select Year(s)", options=available_years, default=default_current_year, key="sku_years", help="Default is the current year.")
            sku_quarter = st.selectbox("Select Quarter", options=["All Quarters", "Q1", "Q2", "Q3", "Q4"], index=0, key="sku_quarter", help="Select a specific quarter or All Quarters.")
        if sku_text.strip() == "":
            st.info("Please enter a Product SKU to view its trends.")
        else:
            fig_sku = create_sku_line_chart(df, sku_text, sku_years, sku_quarter)
            if fig_sku is not None:
                st.plotly_chart(fig_sku, use_container_width=True)

# -----------------------------------------
# Tab 5: Sales Channel
# -----------------------------------------
with tabs[4]:
    st.header("Sales Revenue by Sales Channel")
    with st.expander("Chart Filters", expanded=True):
        st.write("Select the Year(s) to include:")
        selected_sc_years = []
        for year in available_years:
            if st.checkbox(f"{year}", key=f"sc_year_{year}"):
                selected_sc_years.append(year)
        if not selected_sc_years:
            selected_sc_years = default_current_year
        sc_quarter = st.selectbox("Select Quarter", options=["All Quarters", "Q1", "Q2", "Q3", "Q4"], index=0, key="sc_quarter", help="Select a specific quarter or All Quarters.")
    fig_sc = create_sales_channel_chart(df, selected_sc_years, sc_quarter)
    st.plotly_chart(fig_sc, use_container_width=True)

# -----------------------------------------
# Tab 6: Pivot Table
# -----------------------------------------
with tabs[5]:
    st.header("Pivot Table: Revenue by Listing & Week")
    with st.expander("Pivot Table Filters", expanded=True):
        pivot_years = st.multiselect("Select Year(s) for Pivot Table", options=available_years, default=default_current_year, key="pivot_years", help="Default is the current year.")
        sort_order = st.radio("Sort by Total Revenue:", options=["Highest to Lowest", "Lowest to Highest"], index=0, horizontal=True, key="pivot_sort_order")
    pivot = create_pivot_table(df, pivot_years, sort_order)
    st.dataframe(pivot, use_container_width=True)

# -----------------------------------------
# Tab 7: Listings
# -----------------------------------------
with tabs[6]:
    st.header("Weekly Sales by Listing")
    with st.expander("Chart Filters", expanded=True):
        listings_years = st.multiselect("Select Year(s)", options=available_years, default=default_current_year, key="listings_years", help="Default is the current year.")
        listings_quarters = st.multiselect("Select Quarter(s)", options=["All Quarters", "Q1", "Q2", "Q3", "Q4"], default=["All Quarters"], key="listings_quarters", help="Select one or more quarters. Choose 'All Quarters' to include every quarter.")
        available_listings = sorted(df["Listing"].dropna().unique())
        default_listings = ["Pattern Pants", "Pattern Shorts", "Solid Pants", "Solid Shorts", "Pattern Polo"]
        selected_listings = st.multiselect("Select Listing(s)", options=available_listings, default=[lst for lst in default_listings if lst in available_listings], key="selected_listings", help="Select one or more listings to display.")
    if not selected_listings:
        st.info("Please select one or more listings to display the graph.")
    else:
        fig_listing = create_listings_chart(df, listings_years, listings_quarters, selected_listings)
        st.plotly_chart(fig_listing, use_container_width=True)
