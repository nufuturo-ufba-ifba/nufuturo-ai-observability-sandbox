import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
import datetime
import io
import re

st.set_page_config(
    layout="wide", 
    page_title="Analisador de Logs Alexandria",
    page_icon="🧬",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #ffffff;
    }
    
    .main-container {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(149, 117, 205, 0.1);
        border: 1px solid #F0EBFF;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e293b;
        text-align: center;
        margin-bottom: 1.5rem;
        background: linear-gradient(135deg, #9575CD 0%, #B39DDB 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #5E35B1;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #F3E5F5;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #B39DDB;
        box-shadow: 0 2px 6px rgba(149, 117, 205, 0.1);
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid #F3E5F5;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(149, 117, 205, 0.15);
        border-color: #D1C4E9;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #5E35B1;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #7E57C2;
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 12px 12px 0 0;
        padding: 1rem 2rem;
        font-weight: 500;
        border: 1px solid #F0EBFF;
        border-bottom: none;
        margin: 0 0.2rem;
        color: #7E57C2;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #B39DDB 0%, #D1C4E9 100%);
        color: white;
        border-color: #B39DDB;
        box-shadow: 0 2px 8px rgba(149, 117, 205, 0.2);
    }
    
    .upload-section {
        border: 2px dashed #D1C4E9;
        border-radius: 16px;
        padding: 3rem;
        text-align: center;
        background: #F3E5F5;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #9575CD;
        background: #EDE7F6;
        box-shadow: 0 4px 12px rgba(149, 117, 205, 0.1);
    }
    
    .stButton button {
        background: linear-gradient(135deg, #9575CD 0%, #B39DDB 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 500;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 2px 6px rgba(149, 117, 205, 0.2);
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #8561C8 0%, #A08BD6 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(149, 117, 205, 0.3);
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    
    .badge-success { background: #E8F5E8; color: #2E7D32; border: 1px solid #C8E6C9; }
    .badge-warning { background: #FFF8E1; color: #F57C00; border: 1px solid #FFECB3; }
    .badge-error { background: #FFEBEE; color: #C62828; border: 1px solid #FFCDD2; }
    .badge-info { background: #E3F2FD; color: #1565C0; border: 1px solid #BBDEFB; }
    
    .dataframe {
        border-radius: 12px;
        border: 1px solid #F0EBFF;
    }
    
    .stSelectbox, .stMultiselect, .stTextInput {
        border-radius: 8px;
    }
    
    .stSelectbox div, .stMultiselect div {
        border-radius: 8px;
        border: 1px solid #D1C4E9;
    }
    
    .streamlit-expanderHeader {
        background: #ffffff;
        border-radius: 8px;
        border: 1px solid #F0EBFF;
        color: #5E35B1;
        font-weight: 500;
    }
    
    .stMultiSelect div[data-baseweb="select"] span {
        color: #1e293b !important;
    }
    .stMultiSelect div[data-baseweb="tag"] span {
        color: #1e293b !important;
    }
</style>
""", unsafe_allow_html=True)

def create_metric_card(value, label, icon="📊"):
    return f"""
    <div class="metric-card">
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """

def detect_separator(line):
    if ';' in line and line.count(';') > line.count(','):
        return ';'
    return ','

def extract_first_uuid(text):
    if not isinstance(text, str):
        return None
    match = re.search(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', text, re.IGNORECASE)
    return match.group(0) if match else None

def find_key_recursive(data, targets):
    if isinstance(data, dict):
        for k, v in data.items():
            if k in targets and pd.notna(v) and str(v).strip() != "":
                return str(v)
        
        for v in data.values():
            res = find_key_recursive(v, targets)
            if res:
                return res
                
    elif isinstance(data, list):
        for item in data:
            res = find_key_recursive(item, targets)
            if res:
                return res
    
    return None

def normalize_user_id(row):
    user_keys = ['user-id', 'userId', 'user_id', 'sub', 'Subject', 'User', 'id', 'customer_id', 'customerId', 'client_id', 'client-id', 'clientId', 'uid', 'uuid']
    
    row_dict = {k: v for k, v in row.items() if pd.notna(v)}
    
    for field in user_keys:
        if field in row_dict:
             return str(row_dict[field])
             
    parsed_objects = []
    for k, v in row_dict.items():
        if isinstance(v, str) and v.strip().startswith('{'):
            try:
                parsed = json.loads(v)
                parsed_objects.append(parsed)
                res = find_key_recursive(parsed, user_keys)
                if res: return res
            except:
                pass
        elif isinstance(v, (dict, list)):
            parsed_objects.append(v)
            res = find_key_recursive(v, user_keys)
            if res: return res
            
    path_keys = ['path', 'url', 'uri', 'location', 'request_uri']
    for k in path_keys:
        val = None
        if k in row_dict:
            val = row_dict[k]
        elif 'log' in row_dict and isinstance(row_dict['log'], dict) and k in row_dict['log']:
            val = row_dict['log'][k]
            
        if val and isinstance(val, str):
            uuid = extract_first_uuid(val)
            if uuid:
                return uuid

    cid_keys = ['cid', 'correlation_id', 'correlationId', 'request_id', 'requestId', 'trace_id']
    for k in cid_keys:
        res = find_key_recursive(row_dict, [k])
        if res:
            return res
            
    return None

def normalize_timestamp(row):
    possible_fields = ['timestamp', 'created_at', 'createdAt', 'date', 'time', 'occurred_at']
    
    for field in possible_fields:
        if field in row and pd.notna(row[field]):
            try:
                return pd.to_datetime(row[field])
            except:
                pass
    return None

@st.cache_data
def load_and_process_files(files):
    all_data = []
    
    for file in files:
        try:
            filename = file.name
            df_temp = None
            
            if filename.lower().endswith('.csv'):
                try:
                    content = file.read().decode('utf-8', errors='ignore')
                    file.seek(0)
                    first_line = content.split('\n')[0]
                    sep = detect_separator(first_line)
                    
                    df_temp = pd.read_csv(file, sep=sep, on_bad_lines='skip')
                except Exception as e:
                    st.error(f"Erro ao ler CSV {filename}: {e}")
                    
            elif filename.lower().endswith('.json'):
                try:
                    content = json.load(file)
                    if isinstance(content, list):
                        df_temp = pd.DataFrame(content)
                    elif isinstance(content, dict):
                        df_temp = pd.DataFrame([content])
                except Exception as e:
                    file.seek(0)
                    try:
                        df_temp = pd.read_json(file, lines=True)
                    except Exception as e2:
                        st.error(f"Erro ao ler JSON {filename}: {e2}")

            if df_temp is not None:
                df_temp['source_file'] = filename
                df_temp['normalized_user_id'] = df_temp.apply(normalize_user_id, axis=1)
                df_temp['normalized_timestamp'] = df_temp.apply(normalize_timestamp, axis=1)
                
                all_data.append(df_temp)
                
        except Exception as e:
            st.error(f"Falha ao processar arquivo {file.name}: {str(e)}")

    if not all_data:
        return pd.DataFrame()

    full_df = pd.concat(all_data, ignore_index=True)
    
    if 'normalized_timestamp' in full_df.columns:
        full_df.sort_values('normalized_timestamp', inplace=True)
        
    return full_df

def main():
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: #5E35B1; font-weight: 700; margin-bottom: 1rem;">🧬 Analisador de Logs Alexandria</h1>
        <p style="color: #7E57C2; font-size: 1.1rem; max-width: 600px; margin: 0 auto;">
            Consolidação e análise de logs multi-origem (Backend & Mobile)
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="main-container">
        <div class="upload-section">
            <h3 style="color: #5E35B1; margin-bottom: 1rem;">📁 Upload de Arquivos de Log</h3>
            <p style="color: #7E57C2; margin-bottom: 2rem;">Arraste e solte múltiplos arquivos (JSON, CSV)</p>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        " ",
        type=['csv', 'json', 'txt'],
        accept_multiple_files=True,
        help="Selecione um ou mais arquivos de log para análise conjunta",
        label_visibility="collapsed"
    )
    
    st.markdown("</div></div>", unsafe_allow_html=True)

    if uploaded_files:
        with st.spinner('Processando e unificando logs...'):
            df = load_and_process_files(uploaded_files)
        
        if not df.empty:
            total_events = len(df)
            unique_users = df['normalized_user_id'].nunique()
            sources_count = df['source_file'].nunique()
            
            st.markdown('<div class="main-container">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(create_metric_card(total_events, "Total de Eventos", "📝"), unsafe_allow_html=True)
            with col2:
                st.markdown(create_metric_card(unique_users, "Usuários Identificados", "👥"), unsafe_allow_html=True)
            with col3:
                st.markdown(create_metric_card(sources_count, "Fontes de Dados", "📂"), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="main-container">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">🔍 Rastreabilidade Unificada & Timeline</div>', unsafe_allow_html=True)
            
            user_counts = df['normalized_user_id'].value_counts()
            users_list = ["Todos"] + user_counts.index.tolist()
            
            selected_user = st.selectbox(
                "Selecione uma Entidade (Usuário, Cliente ou CID) para rastrear:",
                options=users_list,
                format_func=lambda x: f"{x} ({user_counts[x]} eventos)" if x != "Todos" and pd.notna(x) else ("Todos" if x == "Todos" else "Desconhecido")
            )
            
            filtered_df = df.copy()
            if selected_user != "Todos":
                filtered_df = df[df['normalized_user_id'] == selected_user]
            
            col_chart1, col_chart2 = st.columns([2, 1])
            
            with col_chart1:
                st.subheader("Linha do Tempo de Eventos")
                if 'normalized_timestamp' in filtered_df.columns and filtered_df['normalized_timestamp'].notna().any():
                    fig_timeline = px.scatter(
                        filtered_df, 
                        x='normalized_timestamp', 
                        y='source_file',
                        color='source_file',
                        hover_data=filtered_df.columns,
                        title=f"Atividade Temporal de {selected_user}",
                        height=400
                    )
                    fig_timeline.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        xaxis=dict(showgrid=True, gridcolor='#F0F0F0'),
                        yaxis=dict(showgrid=True, gridcolor='#F0F0F0')
                    )
                    st.plotly_chart(fig_timeline, use_container_width=True)
                else:
                    st.info("Não foi possível gerar timeline (timestamps não identificados)")
            
            with col_chart2:
                st.subheader("Distribuição por Fonte")
                source_counts = filtered_df['source_file'].value_counts().reset_index()
                source_counts.columns = ['source', 'count']
                fig_pie = px.pie(source_counts, values='count', names='source', hole=0.4)
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="main-container">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">📋 Detalhes dos Eventos</div>', unsafe_allow_html=True)
            
            all_cols = filtered_df.columns.tolist()
            
            priority_cols = ['normalized_timestamp', 'level', 'service', 'cid', 'normalized_user_id', 'log']
            
            default_cols = []
            for c in priority_cols:
                if c in all_cols:
                    default_cols.append(c)
            
           
            secondary_cols = ['host', 'message', 'event', 'action', 'screen']
            for c in secondary_cols:
                if c in all_cols and c not in default_cols:
                    default_cols.append(c)
            
            if len(default_cols) < 3:
                 if 'source_file' in all_cols: default_cols.append('source_file')
            
            cols_to_show = st.multiselect(
                "Colunas Visíveis",
                options=all_cols,
                default=default_cols
            )
            
            if cols_to_show:
                if 'normalized_timestamp' in filtered_df.columns:
                     df_display = filtered_df.sort_values('normalized_timestamp', ascending=False)
                     df_display = df_display[cols_to_show]
                else:
                     df_display = filtered_df[cols_to_show]

                column_config = {
                    "log": st.column_config.JsonColumn(
                        "log",
                        help="Visualização estruturada do objeto de log",
                        width="large"
                    ),
                    "normalized_timestamp": st.column_config.DatetimeColumn(
                        "Timestamp",
                        format="D MMM YYYY, HH:mm:ss.SS"
                    ),
                    "normalized_user_id": st.column_config.TextColumn(
                        "UserId",
                        width="medium"
                    ),
                     "cid": st.column_config.TextColumn(
                        "Correlation ID",
                        width="medium",
                        help="ID único para rastrear a requisição entre serviços"
                    ),
                }

                st.dataframe(
                    df_display,
                    column_config=column_config,
                    use_container_width=True,
                    height=600
                )
            else:
                st.warning("Selecione pelo menos uma coluna para visualizar.")
                
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.info("Nenhum dado válido encontrado nos arquivos enviados.")

if __name__ == "__main__":
    main()
