import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Multi-Model Evaluation Dashboard (1-5 Scale)",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to match your HTML exactly
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
    
    .stSelectbox > div > div {
        background-color: rgba(108, 99, 255, 0.1);
        border-radius: 10px;
    }
    
    /* Hide streamlit menu */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Model configurations - EXACTLY as in your HTML
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

# Column mappings - EXACTLY as in your HTML JavaScript
COLUMN_MAPPINGS = {
    'qa': {
        'judgeColumns': [
            'Judge_Model_A_Score',  # MODEL A: LLAMA 3.1 8B INSTRUCT
            'Judge_Model_B_Score',  # MODEL B: V1_INSTRUCT_SFT_CK34
            'Judge_Model_C_Score',  # MODEL C: V2_BASE_CPT_SFT_CK21
            'Judge_Model_F_Score',  # MODEL D: V2_BASE_CPT_SFT_DPO_RUN1
            'Judge_Model_G_Score',  # MODEL E: V2_BASE_CPT_SFT_DPO_RUN2
            'Judge_Model_H_Score',  # MODEL F: V2_BASE_CPT_RESIDUAL
            'Judge_Model_I_Score'   # MODEL G: V2_BASE_CPT_RESIDUAL_CONCISE (QnA only)
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
            'Judge_Model_A_Score',  # MODEL A: LLAMA 3.1 8B INSTRUCT
            'Judge_Model_B_Score',  # MODEL B: V1_INSTRUCT_SFT_CK34
            'Judge_Model_C_Score',  # MODEL C: V2_BASE_CPT_SFT_CK21
            'Judge_Model_F_Score',  # MODEL D: V2_BASE_CPT_SFT_DPO_RUN1
            'Judge_Model_G_Score',  # MODEL E: V2_BASE_CPT_SFT_DPO_RUN2
            'Judge_Model_H_Score'   # MODEL F: V2_BASE_CPT_RESIDUAL
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
            'Judge_Model_A_Score',  # MODEL A: LLAMA 3.1 8B INSTRUCT
            'Judge_Model_B_Score',  # MODEL B: V1_INSTRUCT_SFT_CK34
            'Judge_Model_C_Score',  # MODEL C: V2_BASE_CPT_SFT_CK21
            'Judge_Model_F_Score',  # MODEL D: V2_BASE_CPT_SFT_DPO_RUN1
            'Judge_Model_G_Score',  # MODEL E: V2_BASE_CPT_SFT_DPO_RUN2
            'Judge_Model_H_Score'   # MODEL F: V2_BASE_CPT_RESIDUAL
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

def calculate_averages(df, columns):
    """Calculate average scores for given columns"""
    averages = []
    for col in columns:
        if col in df.columns:
            valid_scores = pd.to_numeric(df[col], errors='coerce').dropna()
            valid_scores = valid_scores[(valid_scores >= 1) & (valid_scores <= 5)]
            if not valid_scores.empty:
                averages.append(valid_scores.mean())
            else:
                averages.append(0)
        else:
            averages.append(0)
    return averages

def calculate_bert_averages(df, columns):
    """Calculate average BERT scores for given columns"""
    averages = []
    for col in columns:
        if col in df.columns:
            valid_scores = pd.to_numeric(df[col], errors='coerce').dropna()
            valid_scores = valid_scores[(valid_scores >= 0) & (valid_scores <= 1)]
            if not valid_scores.empty:
                averages.append(valid_scores.mean())
            else:
                averages.append(0)
        else:
            averages.append(0)
    return averages

def calculate_best_overall_model(datasets):
    """Calculate the best overall model - EXACTLY as in HTML"""
    model_scores = {}
    for model in MODEL_COLORS.keys():
        model_scores[model] = []
    
    for task, data in datasets.items():
        if data is None or data.empty:
            continue
            
        mapping = COLUMN_MAPPINGS.get(task)
        if not mapping:
            continue

        for index, col in enumerate(mapping['judgeColumns']):
            if not col:
                continue
                
            model_key = list(MODEL_COLORS.keys())[index]
            scores = data.apply(lambda row: pd.to_numeric(row.get(col, 0), errors='coerce'), axis=1).dropna()
            scores = scores[(scores >= 1) & (scores <= 5)]
            
            if len(scores) > 0:
                model_scores[model_key].extend(scores.tolist())

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
    """Create judge scores comparison chart - EXACTLY as in HTML"""
    fig = go.Figure()
    
    tasks_to_process = [specific_task] if specific_task else list(datasets.keys())
    
    # Determine max models across tasks
    max_models = 0
    for task in tasks_to_process:
        if datasets[task] is not None and not datasets[task].empty and task in COLUMN_MAPPINGS:
            max_models = max(max_models, len(COLUMN_MAPPINGS[task]['judgeColumns']))
    
    if max_models == 0:
        return fig
    
    # Model labels exactly as in HTML
    model_labels = []
    task_models = list(MODEL_COLORS.keys())[:max_models]
    for model in task_models:
        model_names = {
            'MODEL A': 'A (LLAMA 3.1 8B)',
            'MODEL B': 'B (V1_INSTRUCT_SFT)',
            'MODEL C': 'C (V2_BASE_CPT_SFT_CK21)',
            'MODEL D': 'D (V2_DPO_RUN1)',
            'MODEL E': 'E (V2_DPO_RUN2)',
            'MODEL F': 'F (V2_CPT_RESIDUAL)',
            'MODEL G': 'G (V2_CPT_RESIDUAL_CONCISE)'
        }
        model_labels.append(model_names[model])
    
    for task in tasks_to_process:
        if datasets[task] is None or datasets[task].empty or task not in COLUMN_MAPPINGS:
            continue
        
        data = datasets[task]
        mapping = COLUMN_MAPPINGS[task]
            averages = calculate_averages(data, mapping['judgeColumns'])
            
            # Pad averages with zeros if needed
            while len(averages) < max_models:
                averages.append(0)
            
            # Colors exactly as in HTML
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
                marker_color=task_colors[task],
                marker_line_color=border_colors[task],
                marker_line_width=2
            ))
    
    fig.update_layout(
        title="🏆 Judge Scores Comparison (1-5 Scale)",
        xaxis_title="Models",
        yaxis_title="Judge Score (1-5 Scale)",
        yaxis=dict(range=[1, 5], dtick=0.5),
        showlegend=not specific_task,
        height=400,
        template="plotly_white",
        font=dict(size=12)
    )
    
    return fig

def create_bert_comparison_chart(datasets, specific_task=None):
    """Create BERT F1 scores comparison chart - EXACTLY as in HTML"""
    fig = go.Figure()
    
    tasks_to_process = [specific_task] if specific_task else list(datasets.keys())
    
    # Determine max models across tasks
    max_models = 0
    for task in tasks_to_process:
        if datasets[task] is not None and not datasets[task].empty and task in COLUMN_MAPPINGS:
            max_models = max(max_models, len(COLUMN_MAPPINGS[task]['bertColumns']))
    
    if max_models == 0:
        return fig
    
    # Model labels exactly as in HTML
    model_labels = []
    task_models = list(MODEL_COLORS.keys())[:max_models]
    for model in task_models:
        model_names = {
            'MODEL A': 'A (LLAMA 3.1 8B)',
            'MODEL B': 'B (V1_INSTRUCT_SFT)',
            'MODEL C': 'C (V2_BASE_CPT_SFT_CK21)',
            'MODEL D': 'D (V2_DPO_RUN1)',
            'MODEL E': 'E (V2_DPO_RUN2)',
            'MODEL F': 'F (V2_CPT_RESIDUAL)',
            'MODEL G': 'G (V2_CPT_RESIDUAL_CONCISE)'
        }
        model_labels.append(model_names[model])
    
    for task in tasks_to_process:
        if datasets[task] is None or datasets[task].empty or task not in COLUMN_MAPPINGS:
            continue
        
        data = datasets[task]
        mapping = COLUMN_MAPPINGS[task]
        averages = calculate_bert_averages(data, mapping['bertColumns'])
            
            # Pad averages with zeros if needed
            while len(averages) < max_models:
                averages.append(0)
            
            # Colors exactly as in HTML
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
                    color=line_colors[task],
                    width=3
                ),
                marker=dict(size=6),
                fill=False
            ))
    
    fig.update_layout(
        title="📊 BERT F1 Scores",
        xaxis_title="Models",
        yaxis_title="BERT F1 Score",
        yaxis=dict(range=[0, 1], dtick=0.2),
        showlegend=not specific_task,
        height=400,
        template="plotly_white",
        font=dict(size=12)
    )
    
    return fig

def create_distribution_chart(datasets, specific_task=None):
    """Create judge score distribution chart - EXACTLY as in HTML"""
    tasks_to_process = [specific_task] if specific_task else list(datasets.keys())
    all_scores = []
    
    for task in tasks_to_process:
        if datasets[task] is None or datasets[task].empty or task not in COLUMN_MAPPINGS:
            continue
        
        data = datasets[task]
        mapping = COLUMN_MAPPINGS[task]
            
            for col in mapping['judgeColumns']:
                if not col:
                    continue
                if col in data.columns:
                    scores = pd.to_numeric(data[col], errors='coerce').dropna()
                    scores = scores[(scores >= 1) & (scores <= 5)]
                    all_scores.extend(scores.tolist())

    if len(all_scores) == 0:
        return go.Figure()

    # Calculate distribution exactly as in HTML
    distribution = []
    for score in [1, 2, 3, 4, 5]:
        count = sum(1 for s in all_scores if round(s) == score)
        distribution.append(count)

    # Colors exactly as in HTML
    colors = [
        'rgba(255, 99, 132, 0.8)',
        'rgba(255, 159, 64, 0.8)',
        'rgba(255, 205, 86, 0.8)',
        'rgba(75, 192, 192, 0.8)',
        'rgba(54, 162, 235, 0.8)'
    ]
    
    border_colors = [
        '#FF6384',
        '#FF9F40',
        '#FFCD56',
        '#4BC0C0',
        '#36A2EB'
    ]

    fig = go.Figure(data=[go.Pie(
        labels=['Score 1 (Poor)', 'Score 2 (Below Avg)', 'Score 3 (Average)', 'Score 4 (Good)', 'Score 5 (Excellent)'],
        values=distribution,
        hole=.3,
        marker=dict(
            colors=colors,
            line=dict(color=border_colors, width=2)
        )
    )])
    
    fig.update_layout(
        title="📈 Judge Score Distribution",
        height=400,
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="v", x=1.05)
    )
    
    return fig

def create_correlation_chart(datasets, specific_task=None):
    """Create Judge vs BERT correlation scatter plot - EXACTLY as in HTML"""
    fig = go.Figure()
    
    tasks_to_process = [specific_task] if specific_task else list(datasets.keys())
    
    for task in tasks_to_process:
        if datasets[task] is None or datasets[task].empty or task not in COLUMN_MAPPINGS:
            continue
        
        data = datasets[task]
        mapping = COLUMN_MAPPINGS[task]
            
            for model_index, (judge_col, bert_col) in enumerate(zip(mapping['judgeColumns'], mapping['bertColumns'])):
                if not judge_col or not bert_col:
                    continue
                    
                if judge_col in data.columns and bert_col in data.columns:
                    judge_scores = pd.to_numeric(data[judge_col], errors='coerce')
                    bert_scores = pd.to_numeric(data[bert_col], errors='coerce')
                    
                    valid_mask = (~judge_scores.isna()) & (~bert_scores.isna()) & (judge_scores >= 1) & (judge_scores <= 5) & (bert_scores >= 0)
                    
                    if valid_mask.sum() > 0:
                        model_key = list(MODEL_COLORS.keys())[model_index]
                        fig.add_trace(go.Scatter(
                            x=bert_scores[valid_mask],
                            y=judge_scores[valid_mask],
                            mode='markers',
                            name=f"{task} - {model_key}",
                            marker=dict(
                                color=MODEL_COLORS[model_key],
                                size=8,
                                opacity=0.7
                            )
                        ))
    
    fig.update_layout(
        title="⚖️ Judge vs BERT Correlation",
        xaxis_title="BERT F1 Score",
        yaxis_title="Judge Score (1-5)",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[1, 5]),
        height=400,
        template="plotly_white",
        showlegend=not specific_task,
        font=dict(size=12)
    )
    
    return fig

def create_task_comparison_chart(datasets, specific_task=None):
    """Create task performance comparison chart - EXACTLY as in HTML"""
    fig = go.Figure()
    
    tasks_to_process = [specific_task] if specific_task else list(datasets.keys())
    valid_tasks = [task for task in tasks_to_process if datasets[task] is not None and not datasets[task].empty]
    
    if not valid_tasks:
        return fig
    
    # Determine the maximum number of models across all tasks
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
                
                if index < len(mapping['judgeColumns']) and mapping['judgeColumns'][index] in data.columns:
                    scores = pd.to_numeric(data[mapping['judgeColumns'][index]], errors='coerce').dropna()
                    scores = scores[(scores >= 1) & (scores <= 5)]
                    avg_score = scores.mean() if not scores.empty else 0
                else:
                    avg_score = 0
            else:
                avg_score = 0
                
            task_scores.append(avg_score)
        
        fig.add_trace(go.Bar(
            name=model,
            x=[task.capitalize() for task in valid_tasks],
            y=task_scores,
            marker_color=MODEL_COLORS[model],
            opacity=0.8,
            marker_line_color=MODEL_COLORS[model],
            marker_line_width=2
        ))
    
    fig.update_layout(
        title="🎯 Task Performance Comparison" if not specific_task else "Average Performance Across Tasks",
        xaxis_title="Tasks",
        yaxis_title="Average Judge Score (1-5 Scale)",
        yaxis=dict(range=[1, 5]),
        height=400,
        template="plotly_white",
        barmode='group',
        showlegend=True,
        legend=dict(orientation="h", x=0, y=1.02)
    )
    
    return fig

def main():
    # Header - EXACTLY as in HTML
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Multi-Model Evaluation Dashboard</h1>
        <p>Comprehensive Analysis with Judge Scores (1-5 Scale) & BERT F1 Scores</p>
        <p><em>QnA: 7 Models | Summary & Classification: 6 Models</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Legend - EXACTLY as in HTML
    st.markdown("### Model Legend")
    
    # Create legend in 2 columns like HTML
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
    
    # File Upload Section - EXACTLY as in HTML layout
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
    
    # Summary Statistics - EXACTLY as in HTML
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
                <div class="metric-label">By Judge Score</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Task Selector - EXACTLY as in HTML
    if any(df is not None for df in st.session_state.datasets.values()):
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
        
        # Charts Section - EXACTLY as in HTML layout
        st.markdown("### 📈 Visualization Dashboard")
        
        if task_filter:
            st.info(f"Showing analysis for: **{task_filter.title()}** task")
        else:
            st.info("Showing overview across all loaded tasks")
        
        # Create two columns for charts - EXACTLY as in HTML
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = create_judge_comparison_chart(st.session_state.datasets, task_filter)
            st.plotly_chart(fig1, use_container_width=True)
            
            fig3 = create_distribution_chart(st.session_state.datasets, task_filter)
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            fig2 = create_bert_comparison_chart(st.session_state.datasets, task_filter)
            st.plotly_chart(fig2, use_container_width=True)
            
            fig4 = create_correlation_chart(st.session_state.datasets, task_filter)
            st.plotly_chart(fig4, use_container_width=True)
        
        # Full-width task comparison (only for overview) - EXACTLY as in HTML
        if not task_filter:
            fig5 = create_task_comparison_chart(st.session_state.datasets)
            st.plotly_chart(fig5, use_container_width=True)
        
        # Data Preview Section
        if task_filter and st.session_state.datasets[task_filter] is not None:
            st.markdown(f"### 📋 {task_filter.title()} Data Preview")
            df = st.session_state.datasets[task_filter]
            
            # Show basic info
            st.write(f"**Rows:** {len(df)} | **Columns:** {len(df.columns)}")
            
            # Show relevant columns if they exist
            if task_filter in COLUMN_MAPPINGS:
                judge_cols = COLUMN_MAPPINGS[task_filter]['judgeColumns']
                bert_cols = COLUMN_MAPPINGS[task_filter]['bertColumns']
                
                available_judge_cols = [col for col in judge_cols if col and col in df.columns]
                available_bert_cols = [col for col in bert_cols if col and col in df.columns]
                
                if available_judge_cols or available_bert_cols:
                    preview_cols = available_judge_cols[:5] + available_bert_cols[:5]  # Show first 5 of each
                    if preview_cols:
                        st.dataframe(df[preview_cols].head(10), use_container_width=True)
                    else:
                        st.warning("Expected columns not found. Showing all columns:")
                        st.dataframe(df.head(10), use_container_width=True)
                else:
                    st.warning("Expected columns not found. Showing all columns:")
                    st.dataframe(df.head(10), use_container_width=True)
            else:
                st.dataframe(df.head(10), use_container_width=True)
    
    else:
        st.info("👆 Please upload at least one Excel file to start the analysis.")
        
        # Show sample data format - EXACTLY as in HTML
        with st.expander("📋 Expected Data Format"):
            st.markdown("""
            **For each task, your Excel file should contain:**
            
            **QA Task:**
            - Judge score columns: `Judge_Model_A_Score`, `Judge_Model_B_Score`, etc.
            - BERT F1 columns: `f1_base`, `f1_V34`, `bertscore_f1_v21`, etc.
            
            **Summary Task:**
            - Judge score columns: `Judge_Model_A_Score`, `Judge_Model_B_Score`, etc.
            - BERT F1 columns: `instruct_bertscore_f1`, `finetune_bertscore_f1`, etc.
            
            **Classification Task:**
            - Same structure as Summary task
            
            All judge scores should be on a 1-5 scale, and BERT F1 scores should be between 0-1.
            """)

if __name__ == "__main__":
    main()
