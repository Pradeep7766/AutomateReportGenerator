Link: https://automatereportgenerator.streamlit.app/

# 📊 Advanced Business Report Generator

**Professional-grade sales analytics platform** for automating business intelligence without complex BI tools.

## 🎯 What's New (Advanced Version)

### **Executive Dashboard**

- **Real-time KPIs** - Total Revenue, Units Sold, Average Price, Transaction Count
- **Profit & Margin Tracking** - Automatically calculated from cost data (if provided)
- **Trend Indicators** - Month-over-month growth with visual indicators (📈 UP / 📉 DOWN)

### **Advanced Analytics**

#### 1. **ABC Analysis (Pareto Principle)** 🎯

Automatically classify products into three categories:

- **A Products (0-80% Revenue)** - High Priority / Core Revenue Drivers
- **B Products (80-95% Revenue)** - Medium Priority / Supporting Products
- **C Products (95-100% Revenue)** - Low Priority / Long-tail Products

Identify which 20% of products drive 80% of revenue.

#### 2. **Product Performance Analysis** 🏆

Deep insights for every product:

- Total units sold & average units per transaction
- Price analysis (average, min, max)
- Total revenue generated
- Number of unique selling days
- Volatility measurements (Standard Deviation)

#### 3. **Year-over-Year (YoY) Comparison** 📅

- Monthly revenue trends across multiple years
- Growth percentage calculations
- Identify seasonal patterns
- Compare performance periods

#### 4. **Trend Analysis** 📈

Select period granularity:

- **Daily** - Track daily performance
- **Weekly** - Identify weekly patterns
- **Monthly** - See long-term trends
- **Yearly** - Annual performance overview

#### 5. **Profit & Margin Analysis** 💲

(When cost data is provided)

- Gross profit by product
- Profit margin percentage
- Cost-benefit analysis
- Identify high-margin vs high-volume products
- Dual-axis visualization for profit vs margin comparison

#### 6. **Sales Distribution** 📊

- Quantity distribution analysis (Box plot)
- Price distribution patterns (Histogram)
- Identify outliers and normal ranges
- Transaction pattern identification

---

## 🚀 Key Features

### **Smart Column Detection**

- Automatically detects column mappings using AI fuzzy matching
- Supports common naming conventions
- No need for manual column selection in most cases

### **Data Quality Assurance**

- Automatic data cleaning and validation
- Identifies and removes:
  - Invalid dates
  - Non-numeric values
  - Negative/zero quantities or prices
  - Outliers and anomalies
- Detailed quality report showing cleanup statistics

### **Advanced Filtering**

- Filter by date range
- Multi-select product filtering
- Revenue threshold filtering
- Real-time data updates

### **Professional Exports**

#### Excel Bundle

- Styled workbook with multiple sheets
- Color-coded headers
- Auto-adjusted column widths
- Professional formatting

#### CSV Export

- Cleaned dataset export
- All calculations included
- Ready for further analysis

#### Summary Report

- Executive KPIs in Excel format
- Key metrics overview
- Print-friendly

---

## 📊 Supported Data Columns

### **Required Columns**

| Column            | Purpose                | Example                     |
| ----------------- | ---------------------- | --------------------------- |
| **Date**          | Transaction date       | 2026-01-15, 01/15/2026      |
| **Product/Model** | Product identifier     | iPhone 15, Model A, SKU-001 |
| **Quantity**      | Units sold             | 5, 10, 100                  |
| **Price**         | Selling price per unit | 99.99, 1500.00              |

### **Optional Columns**

| Column       | Purpose         | Impact                             |
| ------------ | --------------- | ---------------------------------- |
| **Cost**     | Unit cost       | Enables profit & margin analysis   |
| **Store**    | Location/Branch | Enables store performance analysis |
| **Customer** | Customer ID     | Future: Customer segmentation      |

---

## 💾 File Format Support

✅ **CSV** - Comma-separated values
✅ **XLSX** - Modern Excel format (recommended)
✅ **XLS** - Legacy Excel format

**Multiple Files** - Automatically merged into one dataset

---

## 🛠️ Technology Stack

- **Frontend**: Streamlit (Interactive web UI)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly (Interactive charts), Matplotlib
- **Excel Export**: OpenPyXL, XlsxWriter
- **Analytics**: SciPy, Statistical functions
- **Smart Detection**: RapidFuzz (fuzzy string matching)

---

## 📈 Report Types Generated

### 1. **Revenue Trend Analysis**

- Line chart showing revenue over time
- Switchable period (Daily/Weekly/Monthly)
- Identifies growth or decline patterns

### 2. **ABC Classification**

- Bar chart with color-coded priorities
- Revenue contribution for each category
- Strategic inventory management insights

### 3. **Product Performance Metrics**

- Detailed table with 8+ metrics
- Sortable and filterable
- Identify strengths and weaknesses

### 4. **Year-over-Year Comparison**

- Multi-year revenue trends
- Growth percentage by month
- Seasonal pattern identification
- Available only if data spans multiple years

### 5. **Profit & Margin Dashboard**

- Dual axis chart (Profit $ + Margin %)
- Identify profitability vs volume tradeoff
- Pricing strategy insights

### 8. **Distribution Analysis**

- Outlier detection
- Normal range identification
- Transaction pattern analysis

---

## 🎨 Report Customization

### **Report Type Options**

1. **📈 Full Dashboard** - Complete analysis with all features
2. **🎯 Custom Report** - Select specific analyses
3. **📊 Deep Analysis** - Detailed metrics and insights

### **Filtering Options**

- **Date Range** - Analyze specific time periods
- **Product Selection** - Focus on specific products
- **Revenue Threshold** - Filter by minimum revenue

---

## 💡 Use Cases

### **Retail Management**

- Identify best-selling products
- Optimize inventory based on ABC classification
- Detect seasonal demand patterns

### **E-commerce Operations**

- Monitor product performance
- Calculate true profitability (with cost data)
- Identify slow movers for clearance

### **Sales Team**

- Track monthly/quarterly targets
- Identify growth opportunities
- Celebrate top performers

### **Finance & Planning**

- Revenue forecasting
- Profit margin analysis
- Year-over-year growth tracking

### **Strategic Decision Making**

- Product discontinuation candidates
- Cross-selling opportunities
- Price optimization insights

---

## 📊 Export Options

### **Excel Bundle** 📊

- All analyses in one file
- Multiple styled sheets
- Professional formatting
- Shareable with stakeholders

### **CSV Export** 📋

- Cleaned dataset
- All calculations preserved
- Import into other tools
- Data science ready

### **Summary Report** 📄

- Executive KPIs
- One-page overview
- Print-friendly format

---

## 🔄 Workflow

```
1. UPLOAD FILES
   ↓
2. AUTO-DETECT COLUMNS
   ↓
3. VALIDATE & CLEAN DATA
   ↓
4. APPLY FILTERS (Optional)
   ↓
5. GENERATE REPORTS
   ├─ Dashboard KPIs
   ├─ Revenue Trends
   ├─ ABC Analysis
   ├─ Product Performance
   ├─ YoY Comparison
   ├─ Top/Bottom Products
   ├─ Profit Analysis
   └─ Distribution Analysis
   ↓
6. EXPORT RESULTS
   ├─ Excel Bundle
   ├─ CSV Data
   └─ Summary Report
```

---

## 📝 Data Quality Assurance

The system automatically:

- ✅ Parses multiple date formats
- ✅ Converts numeric fields
- ✅ Removes invalid records
- ✅ Detects and removes outliers
- ✅ Reports cleanup statistics
- ✅ Shows data issues and warnings

**Example Cleanup Report:**

```
Initial Rows:        10,000
Rows Removed:          245
Final Rows:          9,755
Cleanup Rate:         2.45%

Issues Found:
⚠️ 145 invalid dates
⚠️ 85 non-numeric values in 'quantity'
⚠️ 15 rows with negative cost
```

---

## 🎯 Key Performance Indicators (KPIs)

- **Total Revenue** - Sum of all sales
- **Total Units** - Sum of all quantities sold
- **Average Price** - Mean selling price across transactions
- **Total Transactions** - Number of individual sales records
- **Daily Average Revenue** - Mean daily revenue
- **Total Profit** - Revenue minus cost of goods sold
- **Average Margin** - Mean profit margin percentage

---

## 🚀 Getting Started

1. **Upload your sales data** (CSV or Excel)
2. **Confirm column mappings** (auto-detected)
3. **Review data quality** (see cleanup stats)
4. **Filter if needed** (date range, products, revenue)
5. **View all reports** (interactive dashboard)
6. **Download results** (Excel, CSV, or Summary)

---

## 🔒 Data Privacy

- All processing happens locally on your machine
- No data is stored or transmitted
- 100% client-side computation
- Your data stays yours

---

## 📧 Support & Feedback

For issues or feature requests, please contact the development team.

---

**Made with ❤️ for business analysts, salespeople, and data-driven decision makers**
