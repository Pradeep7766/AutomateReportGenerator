Link: https://automatereportgenerator.streamlit.app/


# Automated Business Report Generator

The **Automated Business Report Generator** is a Streamlit-based application that processes sales datasets and automatically generates analytical reports, visualizations, and downloadable Excel/PDF summaries.  
It is designed for business users, analysts, and managers who want quick insights without complex tools.

---

## 🚀 Features

### 1. Upload Multiple Files
- Supports **CSV, XLSX, XLS**
- Automatically merges all uploaded files

### 2. Column Mapping
- Choose relevant columns:
  - Date
  - Model / Product Name
  - Quantity
  - Selling Price
  - Optional Columns: Cost, Store, Customer

### 3. Data Cleaning & Processing
- Automatic date parsing  
- Converts numeric fields  
- Filters by:
  - Date range  
  - Model/Product names  

### 4. Reports Generated Automatically
#### Monthly Sales Summary
- Total units  
- Total revenue  
- Avg selling price  
- Month-over-month growth  

#### Top Models
- Highest selling models  
- Revenue share  

#### Fast / Slow Moving Models
- Based on past 3 months  
- Fast, moderate, slow classification  

#### Peak Sales Periods
- Daily, weekly, or monthly peak detection  

#### AOV (Average Order Value)
- For each month based on order value  

---

## 📊 Export Options

### Download as Excel
Exports all summary tables into separate sheets.

### Download as PDF
Exports all matplotlib charts as a structured PDF report.

---

## 🛠️ Technology Used

- **Python**
- **Streamlit**
- **Pandas**
- **Matplotlib**
- **OpenPyXL**
- **NumPy**

---
