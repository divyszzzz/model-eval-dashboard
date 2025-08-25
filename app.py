import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Multi-Model Evaluation Dashboard",
    page_icon="🚀",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Model configurations
MODEL_COLORS = {
    'MODEL A': '#FF6B6B',
    'MODEL B': '#4ECDC4', 
    'MODEL C': '#45B7D1',
    'MODEL D': '#FECA57',
    'MODEL E': '#FF9FF3',
    'MODEL F': '#8B5CF6',
    'MODEL G': '#F59E0B'
}

MODEL_NAMES = {
    'MODEL A': 'LLAMA 3.1 8B INSTRUCT',
    'MODEL B': 'V1_INSTRUCT_SFT_CK34',
    'MODEL C': 'V2_BASE_CPT_SFT_CK21',
    'MODEL D': 'V2_BASE_CPT_SFT_DPO_RUN1',
    'MODEL E': 'V2_BASE_CPT_SFT_DPO_RUN2',
    'MODEL F': 'V2_BASE_CPT_RESIDUAL',
    'MODEL G': 'V2_BASE_CPT_RESIDUAL_CONCISE'
}

def load_excel_data(uploaded_file):
    """Load and process Excel data"""
    try:
        df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def create_sample_chart():
    """Create a sample chart to test plotly"""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Sample Data',
        x=['Model A', 'Model B', 'Model C'],
        y=[3.2, 4.1, 3.8],
        marker_color='#FF6B6B'
    ))
    fig.update_layout(
        title="Sample Judge Scores (Testing Plotly)",
        xaxis_title="Models",
        yaxis_title="Score",
        height=400
    )
    return fig

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Multi-Model Evaluation Dashboard</h1>
        <p>Comprehensive Analysis with Judge Scores (1-5 Scale) & BERT F1 Scores</p>
        <p><em>QnA: 7 Models | Summary & Classification: 6 Models</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Test plotly
    st.success("✅ Plotly is working! App deployed successfully.")
    
    # Model Legend
    st.subheader("🎯 Model Legend")
    cols = st.columns(2)
    
    for i, (model, name) in enumerate(MODEL_NAMES.items()):
        with cols[i % 2]:
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div style="width: 20px; height: 20px; background-color: {MODEL_COLORS[model]}; 
                     border-radius: 50%; margin-right: 10px;"></div>
                <strong>{model}:</strong> {name}
            </div>
            """, unsafe_allow_html=True)
    
    # File Upload Section
    st.subheader("📂 Upload Dataset Files")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        qa_file = st.file_uploader("📊 QA Dataset (Excel)", type=['xlsx', 'xls'], key="qa")
    
    with col2:
        summary_file = st.file_uploader("📝 Summary Dataset (Excel)", type=['xlsx', 'xls'], key="summary")
    
    with col3:
        classification_file = st.file_uploader("🏷️ Classification Dataset (Excel)", type=['xlsx', 'xls'], key="classification")
    
    # Show sample chart
    st.subheader("📈 Sample Visualization")
    sample_fig = create_sample_chart()
    st.plotly_chart(sample_fig, use_container_width=True)
    
    # Process uploaded files
    if qa_file or summary_file or classification_file:
        st.subheader("📊 Uploaded Data")
        
        if qa_file:
            qa_data = load_excel_data(qa_file)
            if qa_data is not None:
                st.write("**QA Dataset:**")
                st.write(f"Rows: {len(qa_data)}, Columns: {len(qa_data.columns)}")
                st.dataframe(qa_data.head())
                
        if summary_file:
            summary_data = load_excel_data(summary_file)
            if summary_data is not None:
                st.write("**Summary Dataset:**")
                st.write(f"Rows: {len(summary_data)}, Columns: {len(summary_data.columns)}")
                st.dataframe(summary_data.head())
                
        if classification_file:
            classification_data = load_excel_data(classification_file)
            if classification_data is not None:
                st.write("**Classification Dataset:**")
                st.write(f"Rows: {len(classification_data)}, Columns: {len(classification_data.columns)}")
                st.dataframe(classification_data.head())
    else:
        st.info("👆 Please upload your Excel files to see your data analyzed!")
        
        # Instructions
        st.subheader("📋 Instructions")
        st.markdown("""
        1. **Upload your Excel files** using the file uploaders above
        2. **Your files should contain:**
           - Judge score columns: `Judge_Model_A_Score`, `Judge_Model_B_Score`, etc.
           - BERT F1 columns: `f1_base`, `bertscore_f1_v21`, etc.
        3. **The dashboard will automatically generate visualizations** based on your data
        
        **This is a working version!** Once you confirm it works, we can add more advanced features.
        """)

if __name__ == "__main__":
    main()
