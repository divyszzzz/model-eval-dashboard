# Save this as: dashboard_updated.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Multi-Model Evaluation Dashboard (1-5 Scale) - Updated",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    .main-header {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 5px;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    
    .upload-section {
        background: rgba(108, 99, 255, 0.1);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid rgba(108, 99, 255, 0.2);
        margin-bottom: 2rem;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 15px;
        background: rgba(255,255,255,0.8);
        border-radius: 20px;
        border: 1px solid #ddd;
        font-size: 0.9rem;
        margin-bottom: 10px;
    }
    
    .legend-color {
        width: 18px;
        height: 18px;
        border-radius: 50%;
        border: 2px solid white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .update-indicator {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    
    /* Hide streamlit menu */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
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

# Column mappings
COLUMN_MAPPINGS = {
    'qa': {
        'judgeColumns': [
            'Judge_Model_A_Score',
            'Judge_Model_B_Score',
            'Judge_Model_C_Score',
            'Judge_Model_F_Score',
            'Judge_Model_G_Score',
            'Judge_Model_H_Score',
            'Judge_Model_I_Score'
        ],
        'bertColumns': [
            'f1_base',
            'f1_V34',
            'bertscore_f1_v21',
            'bertscore_f1_v2_dpo_run1',
            'bertscore_f1_v2_dpo_run2',
            'bertscore_f1_v2_cpt_residual',
            'bertscore_f1_V2_BASE_CPT_RESIDUAL_CONCISE_qa'
        ]
    },
    'summary': {
        'judgeColumns': [
            'Judge_Model_A_Score',
            'Judge_Model_B_Score',
            'Judge_Model_C_Score',
            'Judge_Model_F_Score',
            'Judge_Model_G_Score',
            'Judge_Model_H_Score'
        ],
        'bertColumns': [
            'instruct_bertscore_f1',
            'finetune_bertscore_f1',
            'sft_v21_bertscore_f1',
            'bertscore_f1_v2_dpo_run1',
            'bertscore_f1_v2_dpo_run2',
            'bertscore_f1_v2_cpt_residual'
        ]
    },
    'classification': {
        'judgeColumns': [
            'Judge_Model_A_Score',
            'Judge_Model_B_Score',
            'Judge_Model_C_Score',
            'Judge_Model_F_Score',
            'Judge_Model_G_Score',
            'Judge_Model_H_Score'
        ],
        'bertColumns': [
            'instruct_bertscore_f1',
            'finetune_bertscore_f1',
            'sft_v21_bertscore_f1',
            'bertscore_f1_v2_dpo_run1',
            'bertscore_f1_v2_dpo_run2',
            'bertscore_f1_v2_cpt_residual'
        ]
    }
}

# Initialize session state
if 'datasets' not in st.session_state:
    st.session_state.datasets = {
        'qa': None,
        'summary': None,
        'classification': None
    }

def load_excel_data(uploaded_file):
    """Load and process Excel data"""
    try:
        df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def calculate_averages(df, columns, score_range=(1, 5)):
    """Calculate average scores with proper filtering"""
    averages = []
    for col in columns:
        if not col:
            averages.append(0)
            continue
            
        if col in df.columns:
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            if score_range == (1, 5):
                valid_scores = numeric_series[(numeric_series >= 1) & (numeric_series <= 5)].dropna()
            else:
                valid_scores = numeric_series[(numeric_series >= 0) & (numeric_series <= 1)].dropna()
            
            if len(valid_scores) > 0:
                averages.append(valid_scores.mean())
            else:
                averages.append(0)
        else:
            averages.append(0)
    
    return averages

def calculate_best_overall_model(datasets):
    """Calculate the best overall model"""
    model_scores = {model: [] for model in MODEL_COLORS.keys()}
    
    for task, data in datasets.items():
        if data is None or data.empty:
            continue
            
        mapping = COLUMN_MAPPINGS.get(task)
        if not mapping:
            continue

        for index, col in enumerate(mapping['judgeColumns']):
            if not col or index >= len(MODEL_COLORS):
                continue
                
            model_key = list(MODEL_COLORS.keys())[index]
            
            if col in data.columns:
                numeric_series = pd.to_numeric(data[col], errors='coerce')
                valid_scores = numeric_series[(numeric_series >= 1) & (numeric_series <= 5)].dropna()
                
                if len(valid_scores) > 0:
                    model_scores[model_key].extend(valid_scores.tolist())

    best_model = 'MODEL A'
    best_score = 0

    for model, scores in model_scores.items():
        if len(scores) > 0:
            avg_score = sum(scores) / len(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_model = model

    return {'model': best_model, 'score': best_score}

def create_judge_comparison_chart(datasets, specific_task=None):
    """Create judge scores comparison chart with FULL model names"""
    fig = go.Figure()
    
    tasks_to_process = [specific_task] if specific_task else [k for k, v in datasets.items() if v is not None]
    
    max_models = 0
    for task in tasks_to_process:
        if task in COLUMN_MAPPINGS:
            max_models = max(max_models, len(COLUMN_MAPPINGS[task]['judgeColumns']))
    
    if max_models == 0:
        return fig
    
    # UPDATED: Create model labels with FULL model names (not abbreviated)
    model_labels = []
    for i in range(max_models):
        model_key = list(MODEL_COLORS.keys())[i]
        model_labels.append(MODEL_NAMES[model_key])  # FULL NAMES
    
    for task in tasks_to_process:
        if datasets[task] is None or datasets[task].empty or task not in COLUMN_MAPPINGS:
            continue
        
        data = datasets[task]
        mapping = COLUMN_MAPPINGS[task]
        
        averages = calculate_averages(data, mapping['judgeColumns'], (1, 5))
        
        while len(averages) < max_models:
            averages.append(0)
        
        task_colors = {
            'qa': 'rgba(255, 107, 107, 0.8)',
            'summary': 'rgba(78, 205, 196, 0.8)',
            'classification': 'rgba(69, 183, 209, 0.8)'
        }
        
        border_colors = {
            'qa': '#FF6B6B',
            'summary': '#4ECDC4',
            'classification': '#45B7D1'
        }
        
        fig.add_trace(go.Bar(
            name=task.capitalize(),
            x=model_labels,
            y=averages,
            marker_color=task_colors.get(task, 'rgba(128, 128, 128, 0.8)'),
            marker_line_color=border_colors.get(task, '#808080'),
            marker_line_width=2
        ))
    
    fig.update_layout(
        title="🏆 Judge Scores Comparison (1-5 Scale) - UPDATED WITH FULL NAMES",
        xaxis_title="Models",
        yaxis_title="Judge Score (1-5 Scale)",
        yaxis=dict(range=[0, 5], dtick=0.5),
        showlegend=not specific_task,
        height=400,
        template="plotly_white",
        font=dict(size=9),
        xaxis=dict(tickangle=45)  # UPDATED: Rotate for better readability
    )
    
    return fig

def create_bert_comparison_chart(datasets, specific_task=None):
    """Create BERT F1 scores comparison chart with FULL model names"""
    fig = go.Figure()
    
    tasks_to_process = [specific_task] if specific_task else [k for k, v in datasets.items() if v is not None]
    
    max_models = 0
    for task in tasks_to_process:
        if task in COLUMN_MAPPINGS:
            max_models = max(max_models, len(COLUMN_MAPPINGS[task]['bertColumns']))
    
    if max_models == 0:
        return fig
    
    # UPDATED: Create model labels with FULL model names (not abbreviated)
    model_labels = []
    for i in range(max_models):
        model_key = list(MODEL_COLORS.keys())[i]
        model_labels.append(MODEL_NAMES[model_key])  # FULL NAMES
    
    for task in tasks_to_process:
        if datasets[task] is None or datasets[task].empty or task not in COLUMN_MAPPINGS:
            continue
        
        data = datasets[task]
        mapping = COLUMN_MAPPINGS[task]
        
        averages = calculate_averages(data, mapping['bertColumns'], (0, 1))
        
        while len(averages) < max_models:
            averages.append(0)
        
        line_colors = {
            'qa': '#96CEB4',
            'summary': '#FF9F43',
            'classification': '#9966FF'
        }
        
        fig.add_trace(go.Scatter(
            name=task.capitalize(),
            x=model_labels,
            y=averages,
            mode='lines+markers',
            line=dict(
                color=line_colors.get(task, '#808080'),
                width=3
            ),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title="📊 BERT F1 Scores - UPDATED WITH FULL NAMES",
        xaxis_title="Models",
        yaxis_title="BERT F1 Score",
        yaxis=dict(range=[0, 1], dtick=0.2),
        showlegend=not specific_task,
        height=400,
        template="plotly_white",
        font=dict(size=9),
        xaxis=dict(tickangle=45)  # UPDATED: Rotate for better readability
    )
    
    return fig

def create_task_comparison_chart(datasets):
    """UPDATED: Create task performance comparison chart - ONLY FOR OVERVIEW"""
    fig = go.Figure()
    
    valid_tasks = [task for task, data in datasets.items() if data is not None and not data.empty]
    
    if not valid_tasks:
        return fig
    
    max_models = 0
    for task in valid_tasks:
        if task in COLUMN_MAPPINGS:
            max_models = max(max_models, len(COLUMN_MAPPINGS[task]['judgeColumns']))

    models = list(MODEL_COLORS.keys())[:max_models]
    
    for index, model in enumerate(models):
        task_scores = []
        
        for task in valid_tasks:
            data = datasets[task]
            if task in COLUMN_MAPPINGS:
                mapping = COLUMN_MAPPINGS[task]
                
                if index < len(mapping['judgeColumns']) and mapping['judgeColumns'][index]:
                    col = mapping['judgeColumns'][index]
                    if col in data.columns:
                        numeric_series = pd.to_numeric(data[col], errors='coerce')
                        valid_scores = numeric_series[(numeric_series >= 1) & (numeric_series <= 5)].dropna()
                        avg_score = valid_scores.mean() if len(valid_scores) > 0 else 0
                    else:
                        avg_score = 0
                else:
                    avg_score = 0
            else:
                avg_score = 0
                
            task_scores.append(avg_score)
        
        fig.add_trace(go.Bar(
            name=MODEL_NAMES[model],  # UPDATED: Use FULL model names
            x=[task.capitalize() for task in valid_tasks],
            y=task_scores,
            marker_color=MODEL_COLORS[model],
            opacity=0.8,
            marker_line_color=MODEL_COLORS[model],
            marker_line_width=2
        ))
    
    fig.update_layout(
        title="🎯 Task Performance Comparison - OVERVIEW ONLY",
        xaxis_title="Tasks",
        yaxis_title="Average Judge Score (1-5 Scale)",
        yaxis=dict(range=[0, 5]),
        height=500,
        template="plotly_white",
        barmode='group',
        showlegend=True,
        legend=dict(
            orientation="v", 
            x=1.02, 
            y=1,
            font=dict(size=9)
        ),
        font=dict(size=9)
    )
    
    return fig

def main():
    # UPDATED: Header with update indicator
    st.markdown("""
    <div class="update-indicator">
        ✅ UPDATED VERSION - Full Model Names & Removed Charts as Requested
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Multi-Model Evaluation Dashboard - UPDATED</h1>
        <p>Comprehensive Analysis with Judge Scores (1-5 Scale) & BERT F1 Scores</p>
        <p><em>QnA: 7 Models | Summary & Classification: 6 Models</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Legend with full names
    st.markdown("### Model Legend - FULL NAMES")
    
    col1, col2 = st.columns(2)
    models_list = list(MODEL_NAMES.items())
    
    for i, (model, name) in enumerate(models_list):
        with col1 if i % 2 == 0 else col2:
            extra_text = " (QnA only)" if model == 'MODEL G' else ""
            st.markdown(f"""
            <div class="legend-item">
                <div class="legend-color" style="background-color: {MODEL_COLORS[model]};"></div>
                <span><strong>{model}:</strong> {name}{extra_text}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # File Upload Section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📂 Upload Dataset Files")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        qa_file = st.file_uploader("📊 QA Dataset (Excel)", type=['xlsx', 'xls'], key="qa")
    
    with col2:
        summary_file = st.file_uploader("📝 Summary Dataset (Excel)", type=['xlsx', 'xls'], key="summary")
    
    with col3:
        classification_file = st.file_uploader("🏷️ Classification Dataset (Excel)", type=['xlsx', 'xls'], key="classification")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Load datasets
    if qa_file:
        st.session_state.datasets['qa'] = load_excel_data(qa_file)
        if st.session_state.datasets['qa'] is not None:
            st.success(f"QA data loaded successfully ({len(st.session_state.datasets['qa'])} rows)")
    
    if summary_file:
        st.session_state.datasets['summary'] = load_excel_data(summary_file)
        if st.session_state.datasets['summary'] is not None:
            st.success(f"Summary data loaded successfully ({len(st.session_state.datasets['summary'])} rows)")
    
    if classification_file:
        st.session_state.datasets['classification'] = load_excel_data(classification_file)
        if st.session_state.datasets['classification'] is not None:
            st.success(f"Classification data loaded successfully ({len(st.session_state.datasets['classification'])} rows)")
    
    # Summary Statistics
    if any(df is not None for df in st.session_state.datasets.values()):
        st.markdown("### Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        tasks_loaded = sum(1 for df in st.session_state.datasets.values() if df is not None)
        total_samples = sum(len(df) for df in st.session_state.datasets.values() if df is not None)
        best_model_info = calculate_best_overall_model(st.session_state.datasets)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Total Samples</h4>
                <div class="metric-value">{total_samples:,}</div>
                <div class="metric-label">Across All Tasks</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Tasks Loaded</h4>
                <div class="metric-value">{tasks_loaded}</div>
                <div class="metric-label">Out of 3</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Best Overall Model</h4>
                <div class="metric-value">{best_model_info['model']}</div>
                <div class="metric-label">Avg Score: {best_model_info['score']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Task Selector
        st.markdown("### Analysis Controls")
        
        available_tasks = ['overview'] + [task for task, df in st.session_state.datasets.items() if df is not None]
        task_options = {
            'overview': '📈 Overview',
            'qa': '🤔 QA Analysis', 
            'summary': '📝 Summary Analysis',
            'classification': '🏷️ Classification Analysis'
        }
        
        selected_task = st.selectbox(
            "Select Task for Analysis",
            options=available_tasks,
            format_func=lambda x: task_options[x]
        )
        
        task_filter = None if selected_task == 'overview' else selected_task
        
        # UPDATED: Charts Section with new logic
        st.markdown("### 📈 Visualization Dashboard - UPDATED")
        
        if task_filter:
            st.info(f"Showing analysis for: **{task_filter.title()}** task")
            st.warning("📊 Task Performance Comparison chart is ONLY shown on Overview page")
        else:
            st.info("Showing overview across all loaded tasks")
        
        # UPDATED: For overview (main page), show Task Performance + Judge + BERT
        if not task_filter:
            # Task Performance Comparison (full-width, ONLY on overview)
            st.markdown("#### Task Performance Comparison (Overview Only)")
            st.plotly_chart(create_task_comparison_chart(st.session_state.datasets), use_container_width=True)
            
            # Judge and BERT comparison charts side by side
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = create_judge_comparison_chart(st.session_state.datasets, task_filter)
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = create_bert_comparison_chart(st.session_state.datasets, task_filter)
                st.plotly_chart(fig2, use_container_width=True)
        
        # UPDATED: For individual task pages, ONLY show Judge and BERT comparison charts
        else:
            st.markdown("#### Individual Task Analysis (No Task Performance Chart)")
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = create_judge_comparison_chart(st.session_state.datasets, task_filter)
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = create_bert_comparison_chart(st.session_state.datasets, task_filter)
                st.plotly_chart(fig2, use_container_width=True)
    
    else:
        st.info("👆 Please upload at least one Excel file to start the analysis.")

if __name__ == "__main__":
    main()
