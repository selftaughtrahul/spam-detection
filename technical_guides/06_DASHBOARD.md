# Technical Guide 06: Dashboard
## Spam / Fraud Message Detection System

---

## 📋 Overview
Create an interactive Streamlit dashboard to visualize spam trends and test the model interactively.

## Step 1: Dashboard Setup
Create `dashboard/app.py`:

```python
import streamlit as st
import pandas as pd
import plotly.express as px
from src.api.client import check_spam

st.title("🛡️ Spam & Fraud Monitoring")

# 1. Interactive Testing
st.sidebar.header("Test Message")
text_input = st.sidebar.text_area("Enter SMS/Email:")
if st.sidebar.button("Scan"):
    res = check_spam(text_input)
    st.sidebar.write(f"Result: {'🚨 SPAM' if res['is_spam'] else '✅ HAM'}")
    st.sidebar.progress(res['confidence'])

# 2. Analytics
st.header("Spam Analytics")
df = pd.read_csv("logs/predictions.csv")
fig = px.pie(df, names='label', title='Spam vs Ham Ratio')
st.plotly_chart(fig)
```

## Step 2: Run Dashboard
```bash
streamlit run dashboard/app.py
```

## ✅ Checklist
- [ ] Sidebar predictor functional
- [ ] Graph rendering successfully
- [ ] Log data loading from CSV
