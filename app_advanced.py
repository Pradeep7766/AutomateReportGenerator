import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from rapidfuzz import fuzz
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
import warnings
warnings.filterwarnings('ignore')

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="📊 Advanced Sales Report Generator",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://www.example.com/help",
        "Report a bug": "https://www.example.com/bug",
    }
)

# Custom styling
st.markdown("""
    <style>
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .report-header {
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .insight-box {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
    </style>
""", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================

@st.cache_data
def read_uploaded_files(uploaded_files):
    """Read and merge multiple CSV/Excel files."""
    dfs = []
    for up in uploaded_files:
        try:
            if up.name.lower().endswith('.csv'):
                df = pd.read_csv(up)
            else:
                df = pd.read_excel(up)
            df['__source_file'] = up.name
            dfs.append(df)
        except Exception as e:
            st.error(f"❌ Could not read {up.name}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True, sort=False)

def auto_detect_columns(df):
    """Auto-detect column mappings using fuzzy matching."""
    columns = {col: col.lower() for col in df.columns}
    detected = {}
    
    keywords = {
        'date': ['date', 'day', 'time', 'order_date', 'sales_date', 'transaction_date'],
        'product': ['product', 'model', 'item', 'product_name', 'model_name', 'sku'],
        'quantity': ['quantity', 'qty', 'units', 'units_sold', 'quantity_sold'],
        'price': ['price', 'amount', 'revenue', 'selling_price', 'sale_price'],
        'cost': ['cost', 'cost_price', 'unit_cost', 'cogs'],
        'store': ['store', 'location', 'region', 'branch', 'outlet'],
        'customer': ['customer', 'customer_id', 'client', 'customer_name']
    }
    
    for col, col_lower in columns.items():
        for key_type, keywords_list in keywords.items():
            for keyword in keywords_list:
                if fuzz.partial_ratio(col_lower, keyword) > 75:
                    detected[key_type] = col
                    break
    
    return detected

def validate_and_clean_data(df, date_col, qty_col, price_col, cost_col=None):
    """Validate and clean data with detailed reporting."""
    df = df.copy()
    initial_rows = len(df)
    issues = []
    
    # Parse dates
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    bad_dates = df[date_col].isnull().sum()
    if bad_dates > 0:
        issues.append(f"⚠️ {bad_dates} invalid dates")
    df = df.dropna(subset=[date_col])
    
    # Convert numeric columns
    for col in [qty_col, price_col, cost_col]:
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=[qty_col, price_col])
    
    # Remove negative/zero values
    negative_qty = (df[qty_col] <= 0).sum()
    negative_price = (df[price_col] <= 0).sum()
    if negative_qty > 0:
        issues.append(f"⚠️ {negative_qty} rows with qty ≤ 0 removed")
    if negative_price > 0:
        issues.append(f"⚠️ {negative_price} rows with price ≤ 0 removed")
    
    df = df[(df[qty_col] > 0) & (df[price_col] > 0)]
    
    # Remove cost outliers if cost exists
    if cost_col and cost_col in df.columns:
        cost_outliers = (df[cost_col] < 0).sum()
        if cost_outliers > 0:
            issues.append(f"⚠️ {cost_outliers} rows with negative cost removed")
        df = df[df[cost_col] >= 0]
    
    final_rows = len(df)
    removed = initial_rows - final_rows
    
    return df, {
        'initial_rows': initial_rows,
        'final_rows': final_rows,
        'rows_removed': removed,
        'cleanup_percentage': (removed / initial_rows * 100) if initial_rows > 0 else 0,
        'issues': issues
    }

# ==================== ADVANCED ANALYSIS FUNCTIONS ====================

def calculate_kpis(df, date_col, qty_col, price_col, cost_col=None):
    """Calculate key performance indicators."""
    df['revenue'] = df[price_col] * df[qty_col]
    
    kpis = {
        'Total Revenue': f"${df['revenue'].sum():,.2f}",
        'Total Units': f"{df[qty_col].sum():,.0f}",
        'Avg Price': f"${df[price_col].mean():,.2f}",
        'Transactions': f"{len(df):,.0f}",
        'Date Range': f"{df[date_col].min().date()} to {df[date_col].max().date()}",
    }
    
    if cost_col and cost_col in df.columns:
        df['profit'] = df['revenue'] - (df[cost_col] * df[qty_col])
        df['margin_%'] = (df['profit'] / df['revenue'] * 100).replace([np.inf, -np.inf], 0)
        kpis['Total Profit'] = f"${df['profit'].sum():,.2f}"
        kpis['Avg Margin'] = f"{df['margin_%'].mean():.1f}%"
    
    avg_daily_revenue = df.groupby(df[date_col].dt.date)['revenue'].sum().mean()
    kpis['Daily Avg Revenue'] = f"${avg_daily_revenue:,.2f}"
    
    return kpis

def abc_analysis(df, model_col, qty_col, price_col):
    """Pareto/ABC analysis: identify A, B, C products."""
    df = df.copy()
    df['revenue'] = df[price_col] * df[qty_col]
    
    # Group by product
    product_stats = df.groupby(model_col).agg({
        qty_col: 'sum',
        'revenue': 'sum'
    }).reset_index()
    
    product_stats = product_stats.sort_values('revenue', ascending=False)
    product_stats['cumsum_revenue'] = product_stats['revenue'].cumsum()
    total_revenue = product_stats['revenue'].sum()
    product_stats['cumsum_%'] = (product_stats['cumsum_revenue'] / total_revenue * 100)
    
    # Classify: A (0-80%), B (80-95%), C (95-100%)
    def classify_abc(cum_pct):
        if cum_pct <= 80:
            return 'A - High Priority'
        elif cum_pct <= 95:
            return 'B - Medium Priority'
        else:
            return 'C - Low Priority'
    
    product_stats['category'] = product_stats['cumsum_%'].apply(classify_abc)
    
    return product_stats[[model_col, qty_col, 'revenue', 'cumsum_%', 'category']].reset_index(drop=True)

def trend_analysis(df, date_col, qty_col, price_col, period='D'):
    """Analyze sales trends over time."""
    df = df.copy()
    df['revenue'] = df[price_col] * df[qty_col]
    
    if period == 'D':
        grouped = df.groupby(df[date_col].dt.date)
    elif period == 'W':
        grouped = df.groupby(df[date_col].dt.to_period('W'))
    elif period == 'M':
        grouped = df.groupby(df[date_col].dt.to_period('M'))
    else:
        grouped = df.groupby(df[date_col].dt.to_period('Y'))
    
    trend = grouped.agg({
        qty_col: 'sum',
        'revenue': 'sum'
    }).reset_index()
    trend.columns = ['period', 'units', 'revenue']
    trend['avg_price'] = trend['revenue'] / trend['units']
    
    return trend.sort_values('period')

def yoy_comparison(df, date_col, qty_col, price_col):
    """Year-over-year comparison."""
    df = df.copy()
    df['revenue'] = df[price_col] * df[qty_col]
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    
    yoy = df.groupby(['year', 'month']).agg({
        qty_col: 'sum',
        'revenue': 'sum'
    }).reset_index()
    
    yoy_pivot = yoy.pivot(index='month', columns='year', values='revenue')
    
    if len(yoy_pivot.columns) >= 2:
        latest_year = yoy_pivot.columns[-1]
        prev_year = yoy_pivot.columns[-2]
        
        yoy_pivot['Growth_%'] = ((yoy_pivot[latest_year] - yoy_pivot[prev_year]) / yoy_pivot[prev_year] * 100).round(2)
    
    return yoy_pivot

def product_insights(df, model_col, qty_col, price_col, date_col):
    """Deep insights into product performance."""
    df = df.copy()
    df['revenue'] = df[price_col] * df[qty_col]
    df['transaction_date'] = df[date_col].dt.date
    
    insights = df.groupby(model_col).agg({
        qty_col: ['sum', 'mean', 'std'],
        price_col: ['mean', 'min', 'max'],
        'revenue': 'sum',
        'transaction_date': 'nunique'
    }).round(2)
    
    insights.columns = ['Total Units', 'Avg Units/Trans', 'Units StDev', 
                       'Avg Price', 'Min Price', 'Max Price', 'Total Revenue', 'Unique Days']
    
    return insights.sort_values('Total Revenue', ascending=False)

def identify_trends(df, date_col, qty_col, price_col):
    """Identify rising and declining trends."""
    df = df.copy()
    df['revenue'] = df[price_col] * df[qty_col]
    monthly = df.groupby(df[date_col].dt.to_period('M'))['revenue'].sum()
    
    if len(monthly) < 2:
        return None, None
    
    # Compare last month vs previous month
    recent = monthly.iloc[-1]
    prev = monthly.iloc[-2]
    pct_change = ((recent - prev) / prev * 100) if prev != 0 else 0
    
    trend_direction = "📈 UP" if pct_change > 0 else "📉 DOWN"
    
    return trend_direction, pct_change

def generate_styled_excel(report_dict, filename):
    """Generate styled Excel workbook with multiple sheets."""
    wb = Workbook()
    wb.remove(wb.active)
    
    header_fill = PatternFill(start_color="667EEA", end_color="667EEA", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF")
    border = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin')
    )
    
    for sheet_name, df_data in report_dict.items():
        ws = wb.create_sheet(title=sheet_name[:31])
        
        # Write data
        for r_idx, row in enumerate(dataframe_to_rows(df_data, index=False, header=True), 1):
            for c_idx, value in enumerate(row, 1):
                cell = ws.cell(row=r_idx, column=c_idx, value=value)
                
                # Style header row
                if r_idx == 1:
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
                
                # Style data rows
                if c_idx in [2, 3, 4]:  # Numeric columns
                    cell.alignment = Alignment(horizontal='right')
                
                cell.border = border
        
        # Auto-adjust column widths
        for column in ws.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            ws.column_dimensions[column_letter].width = adjusted_width
    
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf

# ==================== STREAMLIT UI ====================

st.title("📊 Advanced Business Report Generator")
st.markdown("*Professional-grade sales analytics platform*")

# Step 1: File Upload
st.subheader("Step 1️⃣: Upload Your Sales Data")
col1, col2 = st.columns([3, 1])

with col1:
    uploaded_files = st.file_uploader(
        "Upload sales files (CSV or Excel)",
        type=['csv', 'xlsx', 'xls'],
        accept_multiple_files=True,
        help="Upload one or multiple files - they will be automatically merged"
    )

if not uploaded_files:
    st.info("👈 Upload your sales data to get started")
    st.stop()

# Load data
with st.spinner("📂 Loading and merging files..."):
    raw_df = read_uploaded_files(uploaded_files)

if raw_df.empty:
    st.error('❌ No data loaded')
    st.stop()

# Step 2: Column Mapping
st.subheader("Step 2️⃣: Map Your Columns")

cols_list = list(raw_df.columns)
detected_cols = auto_detect_columns(raw_df)

col1, col2, col3 = st.columns(3)
with col1:
    date_col = st.selectbox('📅 Date Column *', cols_list,
        index=cols_list.index(detected_cols.get('date', cols_list[0])) if detected_cols.get('date') in cols_list else 0)
    qty_col = st.selectbox('📦 Quantity Column *', cols_list,
        index=cols_list.index(detected_cols.get('quantity', cols_list[0])) if detected_cols.get('quantity') in cols_list else 0)

with col2:
    model_col = st.selectbox('🏷️ Product Column *', cols_list,
        index=cols_list.index(detected_cols.get('product', cols_list[0])) if detected_cols.get('product') in cols_list else 0)
    price_col = st.selectbox('💰 Price Column *', cols_list,
        index=cols_list.index(detected_cols.get('price', cols_list[0])) if detected_cols.get('price') in cols_list else 0)

with col3:
    cost_col = st.selectbox('📊 Cost Column (Optional)', ['(None)'] + cols_list, index=0)
    cost_col = None if cost_col == '(None)' else cost_col
    
    store_col = st.selectbox('🏪 Store Column (Optional)', ['(None)'] + cols_list, index=0)
    store_col = None if store_col == '(None)' else store_col

# Step 3: Data Validation
st.subheader("Step 3️⃣: Data Quality Check")

with st.spinner("🔍 Validating data..."):
    clean_df, quality_report = validate_and_clean_data(raw_df, date_col, qty_col, price_col, cost_col)

# Quality metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📥 Initial Rows", f"{quality_report['initial_rows']:,}")
with col2:
    st.metric("🗑️ Removed", f"{quality_report['rows_removed']:,}")
with col3:
    st.metric("✅ Final Rows", f"{quality_report['final_rows']:,}")
with col4:
    st.metric("🧹 Cleanup Rate", f"{quality_report['cleanup_percentage']:.1f}%")

if quality_report['issues']:
    with st.warning("⚠️ Data Quality Issues:"):
        for issue in quality_report['issues']:
            st.write(f"  {issue}")

if clean_df.empty:
    st.error("❌ No valid data after cleaning")
    st.stop()

# Step 4: Filtering
st.subheader("Step 4️⃣: Filter Data (Optional)")

col1, col2, col3 = st.columns(3)

with col1:
    min_date = clean_df[date_col].min()
    max_date = clean_df[date_col].max()
    date_range = st.date_input("📅 Date Range", value=[min_date, max_date], min_value=min_date, max_value=max_date)
    start_date, end_date = (date_range if len(date_range) == 2 else (min_date, max_date))

with col2:
    selected_models = st.multiselect(
        "🏷️ Products (leave blank for all)",
        sorted(clean_df[model_col].dropna().astype(str).unique().tolist()),
        help="Select specific products or leave blank for all"
    )

with col3:
    product_threshold = st.slider("📊 Min Revenue Filter ($)", 0, int(clean_df[price_col].max()), 0,
        help="Hide products below this revenue threshold")

# Apply filters
df = clean_df.copy()
df = df[(df[date_col].dt.date >= pd.to_datetime(start_date).date()) & 
        (df[date_col].dt.date <= pd.to_datetime(end_date).date())]

if selected_models:
    df = df[df[model_col].astype(str).isin(selected_models)]

if df.empty:
    st.warning("⚠️ No data after filters")
    st.stop()

st.success(f"✅ Ready! Analyzing {len(df):,} transactions")

# ==================== DASHBOARD ====================

st.divider()
st.subheader("📊 Executive Dashboard")

# KPIs Row
kpis = calculate_kpis(df, date_col, qty_col, price_col, cost_col)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("💵 Total Revenue", kpis['Total Revenue'])
with col2:
    st.metric("📦 Total Units", kpis['Total Units'])
with col3:
    st.metric("💰 Avg Price", kpis['Avg Price'])
with col4:
    st.metric("📊 Transactions", kpis['Transactions'])
with col5:
    st.metric("📈 Avg Daily Revenue", kpis['Daily Avg Revenue'])

if 'Total Profit' in kpis:
    col1, col2 = st.columns(2)
    with col1:
        st.metric("💲 Total Profit", kpis['Total Profit'])
    with col2:
        st.metric("📊 Avg Margin", kpis['Avg Margin'])

# Trend indicator
trend_dir, trend_pct = identify_trends(df, date_col, qty_col, price_col)
if trend_dir:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### {trend_dir} Month-over-Month: {abs(trend_pct):.1f}%")

# ==================== MAIN REPORTS ====================

st.divider()

# 1. Revenue Trend
st.subheader("📈 Revenue Trend Analysis")
col1, col2 = st.columns([3, 1])

with col2:
    trend_period = st.radio("Period", ["Daily", "Weekly", "Monthly"], horizontal=True, key="trend_period")
    period_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}

with col1:
    trend_data = trend_analysis(df, date_col, qty_col, price_col, period_map[trend_period])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend_data['period'].astype(str), y=trend_data['revenue'],
        mode='lines+markers', name='Revenue', line=dict(color='#667eea', width=3)))
    fig.update_layout(title=f"{trend_period} Revenue Trend", hovermode='x unified', height=400)
    st.plotly_chart(fig, use_container_width=True)

# 2. ABC Analysis
st.subheader("🎯 ABC Analysis (Pareto Principle)")
abc_df = abc_analysis(df, model_col, qty_col, price_col)

col1, col2 = st.columns([2, 1])

with col1:
    # ABC Chart
    color_map = {'A - High Priority': '#2ca02c', 'B - Medium Priority': '#ff7f0e', 'C - Low Priority': '#d62728'}
    colors = [color_map[cat] for cat in abc_df.head(20)['category']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(y=abc_df.head(20)[model_col].astype(str), x=abc_df.head(20)['revenue'],
        orientation='h', marker=dict(color=colors), text=abc_df.head(20)['category'].str.split('-').str[0],
        textposition='auto'))
    fig.update_layout(title="Top 20 Products by Revenue (ABC Classification)", height=500)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # ABC Statistics
    st.markdown("### ABC Distribution")
    for cat in ['A - High Priority', 'B - Medium Priority', 'C - Low Priority']:
        count = len(abc_df[abc_df['category'] == cat])
        pct = (count / len(abc_df) * 100)
        st.write(f"**{cat}**")
        st.write(f"  Products: {count} ({pct:.1f}%)")
        st.write(f"  Revenue: ${abc_df[abc_df['category'] == cat]['revenue'].sum():,.0f}")

# 3. Product Performance
st.subheader("🏆 Product Performance Analysis")
product_perf = product_insights(df, model_col, qty_col, price_col, date_col)
st.dataframe(product_perf.head(15), use_container_width=True)

# 4. Year-over-Year (if multiple years)
if df[date_col].dt.year.nunique() > 1:
    st.subheader("📅 Year-over-Year Comparison")
    yoy_df = yoy_comparison(df, date_col, qty_col, price_col)
    
    fig = go.Figure()
    for col in yoy_df.columns[:-1]:
        fig.add_trace(go.Scatter(x=yoy_df.index, y=yoy_df[col], mode='lines+markers', name=f'Year {int(col)}'))
    fig.update_layout(title="Revenue by Month (Year-over-Year)", hovermode='x unified', height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(yoy_df, use_container_width=True)

# 5. Profit Analysis (if cost data available)
if cost_col and cost_col in df.columns:
    st.subheader("💲 Profit & Margin Analysis")
    
    df['revenue'] = df[price_col] * df[qty_col]
    df['profit'] = df['revenue'] - (df[cost_col] * df[qty_col])
    df['margin_%'] = (df['profit'] / df['revenue'] * 100).replace([np.inf, -np.inf], 0)
    
    profit_by_product = df.groupby(model_col).agg({
        'profit': 'sum',
        'margin_%': 'mean'
    }).sort_values('profit', ascending=False).head(15)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=profit_by_product.index, y=profit_by_product['profit'],
        name='Profit ($)', marker=dict(color='#2ca02c')), secondary_y=False)
    fig.add_trace(go.Scatter(x=profit_by_product.index, y=profit_by_product['margin_%'],
        name='Margin (%)', line=dict(color='#d62728', width=2), mode='lines+markers'), secondary_y=True)
    
    fig.update_layout(title="Profit vs Margin by Product", height=400, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

# 6. Distribution Analysis
st.subheader("📊 Sales Distribution")

col1, col2 = st.columns(2)

with col1:
    # Quantity distribution
    fig = px.box(df, y=qty_col, title="Quantity Distribution")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Price distribution
    fig = px.histogram(df, x=price_col, nbins=50, title="Price Distribution")
    st.plotly_chart(fig, use_container_width=True)

# ==================== EXPORT SECTION ====================

st.divider()
st.subheader("💾 Download Reports")

# Prepare report data
reports = {
    "ABC_Analysis": abc_df,
    "Product_Performance": product_perf,
}

if df[date_col].dt.year.nunique() > 1:
    reports["YoY_Comparison"] = yoy_comparison(df, date_col, qty_col, price_col)

# Download buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Excel Bundle (All Reports)", use_container_width=True):
        excel_buf = generate_styled_excel(reports, "sales_reports.xlsx")
        st.download_button(
            "📥 Download Excel",
            data=excel_buf,
            file_name=f"sales_reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.ms-excel"
        )

with col2:
    if st.button("📋 Cleaned Data (CSV)", use_container_width=True):
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(
            "📥 Download CSV",
            data=csv_buf.getvalue(),
            file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col3:
    if st.button("📄 Summary Report (Excel)", use_container_width=True):
        summary_data = {
            'Metric': list(kpis.keys()),
            'Value': list(kpis.values())
        }
        summary_df = pd.DataFrame(summary_data)
        summary_buf = io.BytesIO()
        with pd.ExcelWriter(summary_buf, engine='openpyxl') as writer:
            summary_df.to_excel(writer, index=False)
        summary_buf.seek(0)
        
        st.download_button(
            "📥 Download Summary",
            data=summary_buf,
            file_name=f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.ms-excel"
        )

st.success("✅ Analysis complete!")
st.info("💡 Tip: Use multiple report formats to suit different stakeholders")
