import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import sys
from pathlib import Path

# Add project root to path
# Assuming this file is in dashboard/app.py, the root is two levels up
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.models.predictor import SpamPredictor, EnsemblePredictor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
logger.info("Dashboard started")

# Page config
st.set_page_config(
    page_title="Spam Shield Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .spam-alert {
        padding: 20px;
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .ham-alert {
        padding: 20px;
        background-color: #0df07b;
        color: black;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize predictors (cached)
@st.cache_resource
def get_predictor(model_type):
    return SpamPredictor(model_type)

# Header
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/2622/2622089.png", width=100)
with col2:
    st.title("🛡️ Spam Shield Pro")
    st.markdown("### Next-Gen NLP Spam Detection System")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Model Selection
available_models = ["naive_bayes", "xgboost", "svm", "distilbert"]
selected_model = st.sidebar.selectbox(
    "Select Model",
    available_models,
    index=1,  # Default to XGBoost
    help="Choose the AI model for detection"
)

# Model Info
model_info = {
    "naive_bayes": "⚡ Fastest (Baseline)",
    "xgboost": "🌲 High Accuracy (Gradient Boosting)",
    "svm": "📐 Balanced (Linear SVM)",
    "distilbert": "🧠 Smartest (Transformers)"
}
st.sidebar.info(f"Selected: **{selected_model.upper()}**\n\n{model_info.get(selected_model, '')}")

# Main Interface
tabs = st.tabs(["🕵️‍♂️ Scanner", "📊 Analytics", "ℹ️ About"])

# Tab 1: Scanner
with tabs[0]:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📨 Message Scanner")
        
        # Initialize session state if not present
        if "text_input_area" not in st.session_state:
            st.session_state.text_input_area = ""
        if "auto_scan" not in st.session_state:
            st.session_state.auto_scan = False

        # Callback functions
        def clear_text():
            st.session_state.text_input_area = ""
            st.session_state.auto_scan = False
            
        def set_sample(text):
            st.session_state.text_input_area = text
            st.session_state.auto_scan = True

        # Text Input
        text_input = st.text_area(
            "Enter message to analyze:", 
            height=150, 
            placeholder="Paste email or SMS content here...",
            key="text_input_area"
        )
        
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            scan_btn = st.button("🔍 Scan Message", type="primary", use_container_width=True)
            if scan_btn:
                st.session_state.auto_scan = True
                 
        with c2:
            st.button("🗑️ Clear", on_click=clear_text, use_container_width=True)
            
    with col2:
        st.markdown("### 📝 Quick Samples")
        samples = [
            "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt WORD: CLAIM to No: 81010",
            "Hi, are we still meeting for lunch tomorrow at 12?",
            "Your account has been compromised. Click here to reset your password immediately.",
            "Can you send me the report by EOD? Thanks."
        ]
        
        for sample in samples:
            st.button(sample[:50] + "...", on_click=set_sample, args=(sample,), help=sample, use_container_width=True)

    # Results Section
    should_scan = st.session_state.auto_scan and st.session_state.text_input_area
    
    if should_scan:
        with st.spinner(f"Analyzing with {selected_model}..."):
            try:
                # Direct prediction
                start_time = time.time()
                predictor = get_predictor(selected_model)
                result = predictor.predict_with_details(st.session_state.text_input_area)
                latency = (time.time() - start_time) * 1000
                
                # Display results
                st.markdown("---")
                
                score = result['confidence']
                is_spam = result['prediction'] == 'spam'
                
                # Big Alert Box
                if is_spam:
                    st.markdown(f"""
                    <div class="spam-alert">
                        <h1 style='color: white; margin:0;'>🚨 SPAM DETECTED 🚨</h1>
                        <p style='font-size: 20px; margin:0;'>Confidence: {score:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="ham-alert">
                        <h1 style='color: black; margin:0;'>✅ SAFE MESSAGE</h1>
                        <p style='font-size: 20px; margin:0;'>Confidence: {score:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Prediction", result['prediction'].upper())
                m2.metric("Confidence", f"{score:.1%}")
                m3.metric("Spam Probability", f"{result.get('probability_spam', 0):.1%}")
                m4.metric("Latency", f"{latency:.0f}ms")
                
                # Probability Gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = result.get('probability_spam', 0) * 100,
                    title = {'text': "Spam Probability (%)"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "red" if is_spam else "green"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgreen"},
                            {'range': [50, 100], 'color': "lightpink"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                logger.error(f"Error loading model {selected_model}: {e}")
                st.error(f"Error loading model: {e}")
                st.info("Please rerun `python train_all_models.py` to ensure models are trained.")

# Tab 2: Analytics
with tabs[1]:
    st.header("📊 Performance Analytics")
    
    try:
        # Load comparison data
        df = pd.read_csv("reports/model_comparison.csv", index_col=0)
        st.dataframe(df.style.highlight_max(axis=1), use_container_width=True)
        
        # Charts
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Accuracy Comparison")
            fig = px.bar(
                x=df.columns, 
                y=df.loc['Accuracy'],
                labels={'x': 'Model', 'y': 'Accuracy'},
                color=df.columns,
                title="Model Accuracy"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.markdown("### F1 Score Comparison")
            fig = px.bar(
                x=df.columns, 
                y=df.loc['F1 Score'],
                labels={'x': 'Model', 'y': 'F1 Score'},
                color=df.columns,
                title="F1 Score (Balanced Metric)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.warning("Run `python train_all_models.py` to generate analytics data.")

# Tab 3: About
with tabs[2]:
    st.markdown("""
    ### 🛡️ About Spam Shield Pro
    
    This system uses advanced Machine Learning and NLP to detect spam/fraud messages with high accuracy.
    
    #### 🤖 Models Used:
    - **Naive Bayes**: Statistical baseline, extremely fast.
    - **SVM (Support Vector Machine)**: Balanced performance.
    - **XGBoost**: Gradient boosting decision trees.
    - **DistilBERT**: Transformer-based deep learning.
    
    #### 🚀 Features:
    - Real-time scanning
    - Multi-model support
    - Probability analysis
    - Interactive dashboard
    """)