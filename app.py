import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

st.set_page_config(page_title="Business Report Generator", layout="wide")

# ----------------------------- Helper functions -----------------------------

def read_uploaded_files(uploaded_files):
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
            st.error(f"Could not read {up.name}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True, sort=False)

def safe_parse_date(df, col):
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def generate_monthly_summary(df, date_col, qty_col, price_col):
    df = df.copy()
    df['month'] = df[date_col].dt.to_period('M').dt.to_timestamp()
    grouped = df.groupby('month').agg(
        total_units=(qty_col, 'sum'),
        total_revenue=(price_col, lambda x: (x * df.loc[x.index, qty_col]).sum())
    ).reset_index()
    grouped['avg_selling_price'] = grouped['total_revenue'] / grouped['total_units'].replace(0, np.nan)
    grouped = grouped.fillna(0)
    grouped['moM_growth_%'] = grouped['total_revenue'].pct_change().fillna(0) * 100
    return grouped

def generate_model_ranking(df, model_col, qty_col, price_col):
    df = df.copy()
    agg = df.groupby(model_col).agg(
        units_sold=(qty_col, 'sum'),
        revenue=(price_col, lambda x: (x * df.loc[x.index, qty_col]).sum())
    ).sort_values('units_sold', ascending=False).reset_index()
    total_revenue = agg['revenue'].sum()
    agg['revenue_share_%'] = 100 * agg['revenue'] / total_revenue if total_revenue != 0 else 0
    return agg.fillna(0)

def fast_slow_moving(df, model_col, qty_col, date_col, months_window=3):
    df = df.copy()
    last_date = df[date_col].max()
    cutoff = last_date - pd.DateOffset(months=months_window)
    recent = df[df[date_col] >= cutoff]
    agg = recent.groupby(model_col)[qty_col].sum().sort_values(ascending=False).reset_index()
    if agg.empty:
        return pd.DataFrame(columns=[model_col, qty_col, 'status'])
    q1 = agg[qty_col].quantile(0.25)
    q3 = agg[qty_col].quantile(0.75)
    def status(q):
        if q >= q3:
            return 'fast-moving'
        elif q <= q1:
            return 'slow-moving'
        else:
            return 'moderate'
    agg['status'] = agg[qty_col].apply(status)
    return agg

def peak_sales_periods(df, date_col, qty_col, freq='D'):
    df = df.copy()
    if freq == 'D':
        df['period'] = df[date_col].dt.date
    elif freq == 'W':
        df['period'] = df[date_col].dt.to_period('W').apply(lambda r: r.start_time.date())
    elif freq == 'M':
        df['period'] = df[date_col].dt.to_period('M').apply(lambda r: r.start_time.date())
    agg = df.groupby('period')[qty_col].sum().reset_index().sort_values(qty_col, ascending=False)
    return agg

def create_excel_download(df_dict):
    towrite = io.BytesIO()
    with pd.ExcelWriter(towrite, engine='openpyxl') as writer:
        for sheet, d in df_dict.items():
            d.to_excel(writer, sheet_name=sheet[:31], index=False)
    towrite.seek(0)
    return towrite

def create_pdf_download(report_figures):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        for fig in report_figures:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        pdf.infodict()['Title'] = 'Business Report'
    buf.seek(0)
    return buf

# ----------------------------- Streamlit UI -----------------------------

st.title("Automatic Business Report Generator")
st.markdown("Upload your sales files (CSV or Excel). The app will concatenate files and let you map columns and generate reports.")

uploaded_files = st.file_uploader("Upload CSV / Excel files (multiple)", type=['csv', 'xlsx', 'xls'], accept_multiple_files=True)

if not uploaded_files:
    st.info("Please upload at least one CSV or Excel file to begin.")
    st.stop()

raw_df = read_uploaded_files(uploaded_files)

if raw_df.empty:
    st.error('No data loaded. Check your files and try again.')
    st.stop()

st.subheader('Preview of combined data (first 10 rows)')
st.dataframe(raw_df.head(10))
st.write('Detected columns:')
st.write(list(raw_df.columns))

# Column mapping
st.sidebar.header('Column Mapping')
cols = list(raw_df.columns)

def optional_select(label, options):
    options2 = ['(None)'] + options
    sel = st.sidebar.selectbox(label, options2)
    return None if sel == '(None)' else sel

date_col = st.sidebar.selectbox('Select Column (Date)', cols)
model_col = st.sidebar.selectbox('Select Column (Model)', cols)
qty_col = st.sidebar.selectbox('Select Column (Quantity)', cols)
price_col = st.sidebar.selectbox('Select Column (Price)', cols)

cost_col = optional_select('Cost Column (Optional)', cols)
store_col = optional_select('Store/Location Column (Optional)', cols)
customer_col = optional_select('Customer ID Column (Optional)', cols)

# Parse dates
raw_df = safe_parse_date(raw_df, date_col)
if raw_df[date_col].isnull().all():
    st.error('All values in selected date column could not be parsed to datetime.')
    st.stop()

# Clean numeric columns
for c in [qty_col, price_col, cost_col]:
    if c and c in raw_df.columns:
        raw_df[c] = pd.to_numeric(raw_df[c], errors='coerce')

raw_df = raw_df.dropna(subset=[date_col])

# Date range filter
min_date = raw_df[date_col].min()
max_date = raw_df[date_col].max()

date_range = st.sidebar.date_input(
    "Date range",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date
)
start_date, end_date = (date_range if len(date_range) == 2 else (min_date, max_date))

# Model filter
model_options = ['All'] + sorted(raw_df[model_col].dropna().astype(str).unique().tolist())
model_filter = st.sidebar.selectbox('Choose Model', model_options)

# Apply filters
df = raw_df.copy()
df = df[(df[date_col].dt.date >= pd.to_datetime(start_date).date()) & (df[date_col].dt.date <= pd.to_datetime(end_date).date())]
if model_filter != 'All':
    df = df[df[model_col].astype(str) == model_filter]

if df.empty:
    st.warning('No data available after applying filters.')
    st.stop()

st.write(f"Data after filters: {len(df)} rows")

# ------------------------- Generate Reports -------------------------

report_tables = {}
report_figures = []

#  Monthly Summary
st.subheader('Monthly Sales Summary')
monthly = generate_monthly_summary(df, date_col, qty_col, price_col)
st.dataframe(monthly)
report_tables['Monthly_Summary'] = monthly

f1 = plt.figure(figsize=(8,4))
plt.plot(monthly['month'], monthly['total_revenue'], marker='o')
plt.title('Monthly Revenue')
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.xticks(rotation=45)
plt.tight_layout()
report_figures.append(f1)
st.pyplot(f1)

#  Top Models
st.subheader('Top Models by Units Sold')
model_rank = generate_model_ranking(df, model_col, qty_col, price_col)
st.dataframe(model_rank)
report_tables['Model_Ranking'] = model_rank

f2 = plt.figure(figsize=(8,4))
plt.bar(model_rank.head(10)[model_col].astype(str), model_rank.head(10)['units_sold'])
plt.title('Top 10 Models by Units Sold')
plt.xticks(rotation=45)
plt.tight_layout()
report_figures.append(f2)
st.pyplot(f2)

#  Fast/Slow Moving Models
st.subheader('Fast / Slow Moving Models (Last 3 Months)')
fs = fast_slow_moving(df, model_col, qty_col, date_col)
st.dataframe(fs)
report_tables['FastSlow'] = fs

f3 = plt.figure(figsize=(8,4))
if not fs.empty:
    plt.bar(fs[model_col].astype(str), fs[qty_col], color='skyblue')
    plt.title('Model Movement Status')
    plt.xticks(rotation=45)
    plt.tight_layout()
    report_figures.append(f3)
    st.pyplot(f3)

#  Peak Periods
st.subheader('Peak Sales Periods')
freq = st.selectbox('Aggregation for peak detection', ('D','W','M'))
peak = peak_sales_periods(df, date_col, qty_col, freq=freq)
st.dataframe(peak.head(20))
report_tables['Peak_Periods'] = peak

f4 = plt.figure(figsize=(8,4))
plt.plot(peak['period'].astype(str), peak[qty_col], marker='o')
plt.title('Sales by Period')
plt.xticks(rotation=45)
plt.tight_layout()
report_figures.append(f4)
st.pyplot(f4)

#  AOV / ASP
st.subheader('Average Order Value (AOV)')
df['order_value'] = df[price_col] * df[qty_col]
aov = df.groupby(df[date_col].dt.to_period('M').dt.to_timestamp()).agg(
    total_revenue=('order_value','sum'),
    orders_count=(qty_col, 'count')
).reset_index()
aov['AOV'] = aov['total_revenue'] / aov['orders_count'].replace(0, np.nan)
st.dataframe(aov)
report_tables['AOV'] = aov

f5 = plt.figure(figsize=(8,4))
plt.plot(aov[date_col], aov['AOV'], marker='o')
plt.title('Average Order Value (AOV)')
plt.xticks(rotation=45)
plt.tight_layout()
report_figures.append(f5)
st.pyplot(f5)

# ------------------------- Export Reports -------------------------
st.subheader('Export Reports')
col1, col2 = st.columns(2)

with col1:
    if st.button('Download reports as Excel'):
        buf = create_excel_download(report_tables)
        st.download_button('Download Excel', data=buf, file_name='business_reports.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

with col2:
    if st.button('Download reports as PDF'):
        pdf_buf = create_pdf_download(report_figures)
        st.download_button('Download PDF', data=pdf_buf, file_name='business_report.pdf', mime='application/pdf')
