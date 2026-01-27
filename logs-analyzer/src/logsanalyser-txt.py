import streamlit as st
import pandas as pd
import re
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    layout="wide", 
    page_title="Analisador de Logs",
    page_icon="📊",
    initial_sidebar_state="collapsed"
)

# CSS moderno e limpo com roxo claro
st.markdown("""
<style>
    /* Reset e base */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #ffffff;
    }
    
    /* Containers principais */
    .main-container {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(149, 117, 205, 0.1);
        border: 1px solid #F0EBFF;
    }
    
    /* Headers */
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
    
    /* Cards de métricas */
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
    
    /* Tabs estilizadas */
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
    
    /* Upload area */
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
    
    /* Botões */
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
    
    /* Log entries */
    .log-entry {
        font-family: 'Monaco', 'Menlo', monospace;
        font-size: 0.85rem;
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid;
        border-color: #94a3b8;
    }
    
    .error-log {
        background: #FFEBEE;
        border-color: #F44336;
    }
    
    .warning-log {
        background: #FFF3E0;
        border-color: #FF9800;
    }
    
    .info-log {
        background: #E3F2FD;
        border-color: #2196F3;
    }
    
    .debug-log {
        background: #F3E5F5;
        border-color: #9C27B0;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    
    .badge-error { background: #FFEBEE; color: #C62828; border: 1px solid #FFCDD2; }
    .badge-warning { background: #FFF3E0; color: #EF6C00; border: 1px solid #FFE0B2; }
    .badge-info { background: #E3F2FD; color: #1565C0; border: 1px solid #BBDEFB; }
    .badge-success { background: #E8F5E8; color: #2E7D32; border: 1px solid #C8E6C9; }
    
    /* Dataframes */
    .dataframe {
        border-radius: 12px;
        border: 1px solid #F0EBFF;
    }
    
    /* Sliders e inputs */
    .stSlider [data-baseweb="slider"] {
        color: #9575CD;
    }
    
    .stSelectbox, .stMultiselect {
        border-radius: 8px;
    }
    
    .stSelectbox div, .stMultiselect div {
        border-radius: 8px;
        border: 1px solid #D1C4E9;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: #ffffff;
        border-radius: 8px;
        border: 1px solid #F0EBFF;
        color: #5E35B1;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Funções de análise (mantidas as mesmas)
@st.cache_data
def parse_log_file(uploaded_file):
    log_entries = []
    
    if uploaded_file is None:
        return pd.DataFrame()

    content = uploaded_file.read().decode('utf-8', errors='ignore')
    lines = content.split('\n')

    log_patterns = [
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[.,]\d+Z?)\s*(.*)",
        r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s*(.*)",
        r"(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s*(.*)"
    ]

    current_entry = None

    for line in lines:
        match = None
        for pattern in log_patterns:
            match = re.match(pattern, line)
            if match:
                break
        
        if match:
            if current_entry:
                log_entries.append(current_entry)

            timestamp, message = match.groups()
            
            level = "INFO"
            message_upper = message.upper()
            if any(word in message_upper for word in ["ERROR", "EXCEPTION", "FAILED", "CRITICAL", "FATAL"]):
                level = "ERROR"
            elif any(word in message_upper for word in ["WARNING", "WARN", "DEPRECATED"]):
                level = "WARNING"
            elif any(word in message_upper for word in ["DEBUG", "TRACE"]):
                level = "DEBUG"

            current_entry = {
                "timestamp": timestamp,
                "level": level,
                "message": message.strip(),
                "full_log": line,
                "length": len(line),
                "has_exception": bool(re.search(r'Exception|Error', message, re.IGNORECASE))
            }
        elif current_entry:
            current_entry["full_log"] += "\n" + line
            current_entry["length"] += len(line)

    if current_entry:
        log_entries.append(current_entry)

    df = pd.DataFrame(log_entries)
    if not df.empty:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute
            df['time_seconds'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
        except:
            df['timestamp'] = pd.to_datetime('now')
            df['hour'] = 0
            df['minute'] = 0
            df['time_seconds'] = 0
    
    return df

def group_similar_errors(error_df):
    if error_df.empty:
        return pd.DataFrame()

    def get_error_template(msg):
        template = re.sub(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', '<UUID>', msg)
        template = re.sub(r'\[\d{3}\]', '[STATUS_CODE]', template)
        template = re.sub(r'\d+', '<NUM>', template)
        template = re.sub(r'\/[^\s]*\/[^\s]*', '<PATH>', template)
        template = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '<IP>', template)
        template = template.split(':')[0].split('[')[0].strip()
        return template

    error_df['template'] = error_df['message'].apply(get_error_template)
    grouped = error_df.groupby('template').agg(
        count=('template', 'size'),
        examples=('message', lambda x: list(x.unique()[:3])),
        first_occurrence=('timestamp', 'min'),
        last_occurrence=('timestamp', 'max')
    ).sort_values('count', ascending=False).reset_index()
    
    grouped['duration_hours'] = (grouped['last_occurrence'] - grouped['first_occurrence']).dt.total_seconds() / 3600
    
    return grouped

def analyze_failing_components(df):
    component_pattern = re.compile(r'([\w_-]+\.(?:clj|cljc|dart|java|py|js|ts|cpp|c|h|go|rs))')
    
    all_components = []
    error_logs = df[df['level'] == 'ERROR']['full_log']
    
    for log in error_logs:
        found_components = component_pattern.findall(log)
        all_components.extend(found_components)
        
    if not all_components:
        return None
        
    component_counts = pd.Series(all_components).value_counts()
    return component_counts

def perform_advanced_analysis(df):
    analysis = {}
    
    analysis['total_entries'] = len(df)
    analysis['unique_messages'] = df['message'].nunique()
    analysis['avg_message_length'] = df['message'].str.len().mean()
    
    if 'timestamp' in df.columns and not df.empty:
        time_range = df['timestamp'].max() - df['timestamp'].min()
        analysis['time_range_hours'] = time_range.total_seconds() / 3600
        analysis['logs_per_hour'] = len(df) / max(analysis['time_range_hours'], 1)
    
    level_stats = df['level'].value_counts()
    analysis['error_ratio'] = level_stats.get('ERROR', 0) / len(df)
    analysis['warning_ratio'] = level_stats.get('WARNING', 0) / len(df)
    
    common_words = Counter(" ".join(df['message']).split()).most_common(20)
    analysis['common_words'] = common_words
    
    return analysis

def extract_features_for_clustering(df):
    features = []
    
    for _, row in df.iterrows():
        feature_vector = [
            len(row['message']),
            int(row['level'] == 'ERROR'),
            int(row['level'] == 'WARNING'),
            int(row['has_exception']),
            row.get('hour', 0),
            len(row['message'].split()),
            int(any(char.isdigit() for char in row['message'])),
            int('http' in row['message'].lower()),
        ]
        features.append(feature_vector)
    
    return np.array(features)

def analyze_lda_topics(df, lda_model, vectorizer, n_top_words=10):
    """Analisa os tópicos encontrados pelo LDA"""
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    
    for topic_idx, topic in enumerate(lda_model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        topic_words = ", ".join(top_features)
        topics.append({
            'topic_id': topic_idx,
            'top_words': topic_words,
            'word_weights': topic[top_features_ind],
            'top_features': top_features
        })
    
    return topics

def find_optimal_clusters_kmeans(data, max_clusters=10):
    """Encontra o número ótimo de clusters usando o método do cotovelo para KMeans"""
    wcss = []  # Within-Cluster Sum of Square
    silhouette_scores = []
    
    for i in range(2, min(max_clusters + 1, len(data) - 1)):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
        
        if i < len(data):
            silhouette_scores.append(silhouette_score(data, kmeans.labels_))
        else:
            silhouette_scores.append(0)
    
    # Encontrar o ponto do cotovelo
    if len(wcss) > 1:
        deltas = np.diff(wcss)
        second_deltas = np.diff(deltas)
        if len(second_deltas) > 0:
            optimal_idx = np.argmax(second_deltas) + 2
            optimal_clusters = min(optimal_idx + 2, len(wcss) + 1)
        else:
            optimal_clusters = 3
    else:
        optimal_clusters = 2
    
    return wcss, silhouette_scores, optimal_clusters

def find_optimal_topics_lda(text_data, max_topics=10):
    """Encontra o número ótimo de tópicos para LDA usando perplexidade e coerência"""
    vectorizer = CountVectorizer(max_features=100, stop_words='english')
    X = vectorizer.fit_transform(text_data)
    
    perplexities = []
    log_likelihoods = []
    
    for n_components in range(2, min(max_topics + 1, len(text_data))):
        lda = LatentDirichletAllocation(
            n_components=n_components,
            random_state=42,
            max_iter=10
        )
        lda.fit(X)
        
        # Perplexidade (quanto menor, melhor)
        perplexity = lda.perplexity(X)
        perplexities.append(perplexity)
        
        # Log likelihood (quanto maior, melhor)
        log_likelihood = lda.score(X)
        log_likelihoods.append(log_likelihood)
    
    # Encontrar ponto ótimo baseado na taxa de mudança da perplexidade
    if len(perplexities) > 1:
        perplexity_deltas = np.diff(perplexities)
        # Normalizar as mudanças
        normalized_deltas = np.abs(perplexity_deltas) / np.max(np.abs(perplexity_deltas))
        
        # Encontrar onde a taxa de melhoria diminui significativamente
        optimal_idx = np.argmax(normalized_deltas < 0.1) + 2 if np.any(normalized_deltas < 0.1) else 2
        optimal_topics = min(optimal_idx + 1, len(perplexities) + 1)
    else:
        optimal_topics = 3
    
    return perplexities, log_likelihoods, optimal_topics

def perform_clustering(df, n_clusters=3, method='kmeans'):
    """Realiza clusterização dos logs"""
    if len(df) < n_clusters:
        st.warning(f"Poucas amostras para clusterização. Necessário pelo menos {n_clusters} amostras.")
        return df, None, None, None, None
    
    # Extrair features
    features = extract_features_for_clustering(df)
    
    # Normalizar features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Aplicar clusterização
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = clusterer.fit_predict(features_scaled)
        vectorizer = None
        
    elif method == 'dbscan':
        # DBSCAN não precisa de número de clusters - ele determina automaticamente
        clusterer = DBSCAN(eps=0.5, min_samples=2)
        clusters = clusterer.fit_predict(features_scaled)
        vectorizer = None
        
    else:  # lda
        # Para LDA, usamos o texto das mensagens
        vectorizer = CountVectorizer(max_features=100, stop_words='english')
        text_data = df['message'].fillna('').astype(str)
        X = vectorizer.fit_transform(text_data)
        
        lda = LatentDirichletAllocation(
            n_components=n_clusters, 
            random_state=42,
            max_iter=10
        )
        lda_features = lda.fit_transform(X)
        clusters = np.argmax(lda_features, axis=1)
        clusterer = lda
    
    # Calcular métricas de qualidade (apenas para KMeans)
    if method == 'kmeans' and len(set(clusters)) > 1:
        silhouette_avg = silhouette_score(features_scaled, clusters)
    else:
        silhouette_avg = -1
    
    # Adicionar clusters ao DataFrame
    df_clustered = df.copy()
    df_clustered['cluster'] = clusters
    
    # Redução de dimensionalidade para visualização
    if method == 'lda':
        # Para LDA, usamos t-SNE que funciona melhor com distribuições de tópicos
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(lda_features)
    else:
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features_scaled)
    
    return df_clustered, features_2d, silhouette_avg, clusterer, vectorizer

def analyze_cluster_patterns(df_clustered):
    cluster_analysis = []
    
    for cluster_id in sorted(df_clustered['cluster'].unique()):
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
        
        analysis = {
            'cluster_id': cluster_id,
            'size': len(cluster_data),
            'error_ratio': len(cluster_data[cluster_data['level'] == 'ERROR']) / len(cluster_data),
            'avg_length': cluster_data['message'].str.len().mean(),
            'common_level': cluster_data['level'].mode().iloc[0] if not cluster_data['level'].mode().empty else 'UNKNOWN',
            'sample_messages': cluster_data['message'].head(3).tolist()
        }
        
        cluster_analysis.append(analysis)
    
    return pd.DataFrame(cluster_analysis)

def create_metric_card(value, label, icon="📊"):
    """Cria um card de métrica estilizado"""
    return f"""
    <div class="metric-card">
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """

def main():
    # Header principal
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: #5E35B1; font-weight: 700; margin-bottom: 1rem;">📊 Analisador de Logs</h1>
        <p style="color: #7E57C2; font-size: 1.1rem; max-width: 600px; margin: 0 auto;">
            Analise e visualize logs de forma inteligente com clusterização automática
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Área de upload
    st.markdown("""
    <div class="main-container">
        <div class="upload-section">
            <h3 style="color: #5E35B1; margin-bottom: 1rem;">📁 Faça upload do arquivo de log</h3>
            <p style="color: #7E57C2; margin-bottom: 2rem;">Arquivos .txt ou .log</p>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        " ",
        type=['txt', 'log'],
        label_visibility="collapsed"
    )
    
    st.markdown("</div></div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        with st.spinner('Analisando arquivo...'):
            df = parse_log_file(uploaded_file)

        if df.empty:
            st.error("Não foi possível processar o arquivo. Verifique o formato.")
            return

        # Status do upload
        st.markdown(f"""
        <div class="main-container" style="background: #F3E5F5; border-left-color: #7E57C2;">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <span style="font-size: 1.5rem; color: #7E57C2;">✅</span>
                <div>
                    <h4 style="margin: 0; color: #5E35B1;">Arquivo processado com sucesso!</h4>
                    <p style="margin: 0; color: #7E57C2;">{len(df)} entradas de log encontradas</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Tabs principais
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Visão Geral", 
            "🔍 Análise de Erros", 
            "🤖 Clusterização", 
            "📊 Estatísticas"
        ])

        with tab1:
            st.markdown('<div class="main-container">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Resumo do Sistema</div>', unsafe_allow_html=True)
            
            # Métricas principais
            total_logs = len(df)
            error_count = len(df[df['level'] == 'ERROR'])
            warning_count = len(df[df['level'] == 'WARNING'])
            debug_count = len(df[df['level'] == 'DEBUG'])
            info_count = len(df[df['level'] == 'INFO'])

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.markdown(create_metric_card(total_logs, "Total de Logs", "📄"), unsafe_allow_html=True)
            with col2:
                st.markdown(create_metric_card(error_count, "Erros", "🔥"), unsafe_allow_html=True)
            with col3:
                st.markdown(create_metric_card(warning_count, "Warnings", "⚠️"), unsafe_allow_html=True)
            with col4:
                st.markdown(create_metric_card(info_count, "Info", "💡"), unsafe_allow_html=True)
            with col5:
                st.markdown(create_metric_card(debug_count, "Debug", "🐛"), unsafe_allow_html=True)

            # Gráficos
            st.markdown('<div class="section-header">Visualizações</div>', unsafe_allow_html=True)
            
            level_counts = df['level'].value_counts()
            fig_pie = px.pie(
                values=level_counts.values, 
                names=level_counts.index, 
                title='Distribuição por Nível',
                color=level_counts.index,
                color_discrete_map={'ERROR':'#F44336', 'WARNING':'#FF9800', 'INFO':'#2196F3', 'DEBUG':'#9C27B0'}
            )
            
            if not df.empty and 'timestamp' in df.columns:
                logs_over_time = df.set_index('timestamp').resample('5min').size().reset_index(name='count')
                fig_line = px.line(
                    logs_over_time, 
                    x='timestamp', 
                    y='count', 
                    title='Volume de Logs ao Longo do Tempo',
                    color_discrete_sequence=['#9575CD']
                )
                
                col1_chart, col2_chart = st.columns(2)
                with col1_chart:
                    st.plotly_chart(fig_pie, use_container_width=True)
                with col2_chart:
                    st.plotly_chart(fig_line, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

        with tab2:
            st.markdown('<div class="main-container">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Análise de Erros</div>', unsafe_allow_html=True)
            
            error_df = df[df['level'] == 'ERROR'].copy()
            
            if not error_df.empty:
                # Status dos erros
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(create_metric_card(len(error_df), "Total de Erros", "🔥"), unsafe_allow_html=True)
                with col2:
                    grouped_errors = group_similar_errors(error_df)
                    st.markdown(create_metric_card(len(grouped_errors), "Grupos de Erros", "🎯"), unsafe_allow_html=True)
                with col3:
                    unique_errors = error_df['message'].nunique()
                    st.markdown(create_metric_card(unique_errors, "Erros Únicos", "🔍"), unsafe_allow_html=True)

                st.markdown('<div class="section-header">Grupos de Erros Similares</div>', unsafe_allow_html=True)
                st.dataframe(grouped_errors.head(10), use_container_width=True)

                # Gráfico de grupos
                fig_grouped_bar = px.bar(
                    grouped_errors.head(10),
                    x='count',
                    y='template',
                    title="Top 10 Grupos de Erros",
                    orientation='h',
                    color='count',
                    color_continuous_scale='reds'
                )
                st.plotly_chart(fig_grouped_bar, use_container_width=True)

            else:
                st.markdown("""
                <div style="text-align: center; padding: 3rem; color: #7E57C2;">
                    <span style="font-size: 3rem;">🎉</span>
                    <h3>Nenhum erro encontrado!</h3>
                    <p>O sistema está funcionando perfeitamente.</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

        with tab3:
            st.markdown('<div class="main-container">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">🤖 Clusterização de Logs</div>', unsafe_allow_html=True)
            
            st.markdown("#### ⚙️ Configuração da Clusterização")
            cluster_method = st.selectbox(
                "Método de Clusterização", 
                options=['kmeans', 'dbscan', 'lda'],
                help="KMeans: grupos por features, DBSCAN: detecção automática de outliers, LDA: análise de tópicos em texto"
            )
            
            if cluster_method == 'kmeans':
                # Para KMeans, calcular número ótimo de clusters
                with st.spinner("Calculando número ótimo de clusters..."):
                    features = extract_features_for_clustering(df)
                    scaler = StandardScaler()
                    features_scaled = scaler.fit_transform(features)
                    
                    wcss, silhouette_scores, optimal_clusters = find_optimal_clusters_kmeans(features_scaled)
                
                st.markdown("#### 📊 Gráfico de Cotovelo - KMeans")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Gráfico de Cotovelo (WCSS)
                    fig_elbow = go.Figure()
                    fig_elbow.add_trace(go.Scatter(
                        x=list(range(2, len(wcss) + 2)),
                        y=wcss,
                        mode='lines+markers',
                        name='WCSS',
                        line=dict(color='#9575CD', width=3),
                        marker=dict(size=8, color='#7E57C2')
                    ))
                    fig_elbow.add_vline(x=optimal_clusters, line_dash="dash", 
                                      line_color="#FF6B6B", annotation_text=f"Ótimo: {optimal_clusters}")
                    fig_elbow.update_layout(
                        title='Método do Cotovelo - WCSS vs Número de Clusters',
                        xaxis_title='Número de Clusters',
                        yaxis_title='Within-Cluster Sum of Squares (WCSS)',
                        showlegend=False
                    )
                    st.plotly_chart(fig_elbow, use_container_width=True)
                
                with col2:
                    # Gráfico de Silhouette Score
                    fig_silhouette = go.Figure()
                    fig_silhouette.add_trace(go.Scatter(
                        x=list(range(2, len(silhouette_scores) + 2)),
                        y=silhouette_scores,
                        mode='lines+markers',
                        name='Silhouette Score',
                        line=dict(color='#7E57C2', width=3),
                        marker=dict(size=8, color='#5E35B1')
                    ))
                    fig_silhouette.update_layout(
                        title='Silhouette Score vs Número de Clusters',
                        xaxis_title='Número de Clusters',
                        yaxis_title='Silhouette Score',
                        showlegend=False
                    )
                    st.plotly_chart(fig_silhouette, use_container_width=True)
                
                st.success(f"🎯 **Número ótimo de clusters sugerido:** {optimal_clusters}")
                n_clusters = st.slider("Número de Clusters", min_value=2, max_value=8, value=optimal_clusters)
                
            elif cluster_method == 'dbscan':
                st.info("""
                🔍 **DBSCAN (Density-Based Spatial Clustering)**: 
                Este método detecta automaticamente o número de clusters baseado na densidade dos dados.
                Parâmetros principais:
                - **eps**: Distância máxima entre pontos para serem considerados vizinhos
                - **min_samples**: Número mínimo de pontos para formar um cluster denso
                """)
                col1, col2 = st.columns(2)
                with col1:
                    eps = st.slider("EPS (Distância)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
                with col2:
                    min_samples = st.slider("Min Samples", min_value=2, max_value=10, value=2)
                n_clusters = None  # DBSCAN determina automaticamente
                
            else:  # LDA
                # Para LDA, calcular número ótimo de tópicos
                with st.spinner("Calculando número ótimo de tópicos..."):
                    text_data = df['message'].fillna('').astype(str)
                    perplexities, log_likelihoods, optimal_topics = find_optimal_topics_lda(text_data)
                
                st.markdown("#### 📊 Métricas de Qualidade - LDA")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Gráfico de Perplexidade
                    fig_perplexity = go.Figure()
                    fig_perplexity.add_trace(go.Scatter(
                        x=list(range(2, len(perplexities) + 2)),
                        y=perplexities,
                        mode='lines+markers',
                        name='Perplexidade',
                        line=dict(color='#9575CD', width=3),
                        marker=dict(size=8, color='#7E57C2')
                    ))
                    fig_perplexity.add_vline(x=optimal_topics, line_dash="dash", 
                                           line_color="#FF6B6B", annotation_text=f"Ótimo: {optimal_topics}")
                    fig_perplexity.update_layout(
                        title='Perplexidade vs Número de Tópicos',
                        xaxis_title='Número de Tópicos',
                        yaxis_title='Perplexidade (menor é melhor)',
                        showlegend=False
                    )
                    st.plotly_chart(fig_perplexity, use_container_width=True)
                
                with col2:
                    # Gráfico de Log Likelihood
                    fig_likelihood = go.Figure()
                    fig_likelihood.add_trace(go.Scatter(
                        x=list(range(2, len(log_likelihoods) + 2)),
                        y=log_likelihoods,
                        mode='lines+markers',
                        name='Log Likelihood',
                        line=dict(color='#7E57C2', width=3),
                        marker=dict(size=8, color='#5E35B1')
                    ))
                    fig_likelihood.update_layout(
                        title='Log Likelihood vs Número de Tópicos',
                        xaxis_title='Número de Tópicos',
                        yaxis_title='Log Likelihood (maior é melhor)',
                        showlegend=False
                    )
                    st.plotly_chart(fig_likelihood, use_container_width=True)
                
                st.success(f"🎯 **Número ótimo de tópicos sugerido:** {optimal_topics}")
                n_clusters = st.slider("Número de Tópicos", min_value=2, max_value=8, value=optimal_topics)
            
            if st.button("🚀 Executar Clusterização", use_container_width=True):
                with st.spinner("Realizando clusterização..."):
                    df_clustered, features_2d, silhouette_score, clusterer, vectorizer = perform_clustering(
                        df, n_clusters, cluster_method
                    )
                    
                    if df_clustered is not None:
                        if cluster_method == 'kmeans':
                            st.success(f"Clusterização concluída! Score de Silhueta: {silhouette_score:.3f}")
                        elif cluster_method == 'dbscan':
                            n_clusters_found = len(set(df_clustered['cluster'])) - (1 if -1 in df_clustered['cluster'].values else 0)
                            n_outliers = len(df_clustered[df_clustered['cluster'] == -1])
                            st.success(f"Clusterização concluída! {n_clusters_found} clusters encontrados, {n_outliers} outliers")
                        else:  # LDA
                            st.success(f"Análise de tópicos concluída! {n_clusters} tópicos identificados")
                        
                        # Visualização dos clusters
                        st.markdown("#### 📊 Visualização dos Clusters")
                        
                        if features_2d is not None:
                            viz_df = pd.DataFrame({
                                'x': features_2d[:, 0],
                                'y': features_2d[:, 1],
                                'cluster': df_clustered['cluster'],
                                'level': df_clustered['level'],
                                'message': df_clustered['message'].str[:50] + '...'
                            })
                            
                            if cluster_method == 'lda':
                                title = 'Visualização 2D dos Tópicos LDA (t-SNE)'
                            elif cluster_method == 'dbscan':
                                title = 'Visualização 2D dos Clusters DBSCAN'
                            else:
                                title = 'Visualização 2D dos Clusters KMeans'
                            
                            fig_clusters = px.scatter(
                                viz_df, x='x', y='y', color='cluster',
                                hover_data=['level', 'message'],
                                title=title,
                                color_continuous_scale='purples'
                            )
                            st.plotly_chart(fig_clusters, use_container_width=True)
                        
                        # Análise específica para LDA
                        if cluster_method == 'lda' and hasattr(clusterer, 'components_'):
                            st.markdown("#### 📝 Análise de Tópicos LDA")
                            
                            topics = analyze_lda_topics(df, clusterer, vectorizer)
                            
                            for topic in topics:
                                with st.expander(f"Tópico {topic['topic_id']} - Palavras Principais"):
                                    st.write(f"**Palavras-chave:** {topic['top_words']}")
                                    
                                    # Gráfico de barras das palavras mais importantes
                                    fig_words = px.bar(
                                        x=topic['word_weights'],
                                        y=[word for word in topic['top_words'].split(", ")],
                                        orientation='h',
                                        title=f"Palavras Mais Importantes - Tópico {topic['topic_id']}",
                                        labels={'x': 'Importância', 'y': 'Palavras'},
                                        color_discrete_sequence=['#9575CD']
                                    )
                                    st.plotly_chart(fig_words, use_container_width=True)
                                    
                                    # Mostrar exemplos de logs deste tópico
                                    topic_logs = df_clustered[df_clustered['cluster'] == topic['topic_id']].head(3)
                                    if not topic_logs.empty:
                                        st.write("**Exemplos de logs deste tópico:**")
                                        for _, log in topic_logs.iterrows():
                                            level_class = f"{log['level'].lower()}-log"
                                            st.markdown(f"""
                                            <div class='log-entry {level_class}'>
                                                <strong>[{log['level']}]</strong> {log['message']}
                                            </div>
                                            """, unsafe_allow_html=True)
                        
                        # Análise dos clusters (para todos os métodos)
                        st.markdown("#### 📈 Análise dos Clusters")
                        cluster_analysis = analyze_cluster_patterns(df_clustered)
                        st.dataframe(cluster_analysis, use_container_width=True)
                        
                        # Exemplos por cluster
                        st.markdown("#### 🔍 Exemplos por Cluster")
                        for cluster_id in sorted(df_clustered['cluster'].unique()):
                            cluster_size = len(df_clustered[df_clustered['cluster'] == cluster_id])
                            if cluster_method == 'dbscan' and cluster_id == -1:
                                with st.expander(f"Outliers - {cluster_size} amostras"):
                                    cluster_samples = df_clustered[df_clustered['cluster'] == cluster_id].head(3)
                                    for _, sample in cluster_samples.iterrows():
                                        level_class = f"{sample['level'].lower()}-log"
                                        st.markdown(f"""
                                        <div class='log-entry {level_class}'>
                                            <strong>[{sample['level']}]</strong> {sample['message']}
                                        </div>
                                        """, unsafe_allow_html=True)
                            else:
                                with st.expander(f"Cluster {cluster_id} - {cluster_size} amostras"):
                                    cluster_samples = df_clustered[df_clustered['cluster'] == cluster_id].head(3)
                                    for _, sample in cluster_samples.iterrows():
                                        level_class = f"{sample['level'].lower()}-log"
                                        st.markdown(f"""
                                        <div class='log-entry {level_class}'>
                                            <strong>[{sample['level']}]</strong> {sample['message']}
                                        </div>
                                        """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

        with tab4:
            st.markdown('<div class="main-container">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Estatísticas Detalhadas</div>', unsafe_allow_html=True)
            
            analysis = perform_advanced_analysis(df)
            
            # Métricas de qualidade
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mensagens Únicas", analysis['unique_messages'])
            with col2:
                st.metric("Comprimento Médio", f"{analysis['avg_message_length']:.0f} chars")
            with col3:
                st.metric("Taxa de Erro", f"{analysis['error_ratio']:.1%}")
            with col4:
                st.metric("Logs por Hora", f"{analysis.get('logs_per_hour', 0):.1f}")

            # Visualizações
            col1, col2 = st.columns(2)
            
            with col1:
                common_words_df = pd.DataFrame(analysis['common_words'], columns=['Palavra', 'Frequência'])
                fig_words = px.bar(
                    common_words_df.head(10), 
                    x='Frequência', 
                    y='Palavra', 
                    orientation='h',
                    title='Palavras Mais Frequentes',
                    color='Frequência',
                    color_continuous_scale='purples'
                )
                st.plotly_chart(fig_words, use_container_width=True)
            
            with col2:
                if 'hour' in df.columns:
                    hour_dist = df['hour'].value_counts().sort_index()
                    hour_df = pd.DataFrame({
                        'hora': hour_dist.index,
                        'quantidade': hour_dist.values
                    })
                    fig_hour = px.bar(
                        hour_df, 
                        x='hora', 
                        y='quantidade',
                        title='Logs por Hora do Dia',
                        color_discrete_sequence=['#B39DDB']
                    )
                    st.plotly_chart(fig_hour, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

        # Explorador de Logs
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Explorador de Logs</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            log_levels = df['level'].unique().tolist()
            selected_levels = st.multiselect(
                "Níveis:",
                options=log_levels,
                default=log_levels
            )
        with col2:
            search_term = st.text_input(
                "Buscar:",
                placeholder="Digite um termo para buscar..."
            )

        filtered_df = df[df['level'].isin(selected_levels)]
        if search_term:
            filtered_df = filtered_df[filtered_df['message'].str.contains(search_term, case=False, na=False)]

        st.write(f"**Mostrando {len(filtered_df)} de {total_logs} entradas**")
        
        # Paginação
        page_size = st.slider("Logs por página:", 10, 50, 20)
        total_pages = max(1, len(filtered_df) // page_size + 1)
        page = st.number_input("Página", min_value=1, max_value=total_pages, value=1)
        
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, len(filtered_df))
        
        for _, row in filtered_df.iloc[start_idx:end_idx].iterrows():
            level_class = f"{row['level'].lower()}-log"
            timestamp_str = row.get('timestamp', 'N/A')
            if hasattr(timestamp_str, 'strftime'):
                timestamp_str = timestamp_str.strftime('%Y-%m-%d %H:%M:%S')
                
            with st.expander(f"{timestamp_str} - {row['message'][:100]}..."):
                st.markdown(f"<div class='log-entry {level_class}'>{row['full_log']}</div>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Tela inicial com componentes nativos do Streamlit
        with st.container():
            st.markdown('<div class="main-container">', unsafe_allow_html=True)
            
            st.markdown("""
            <div style="text-align: center;">
                <h2 style="color: #5E35B1; margin-bottom: 2rem;">Como Usar o Analisador de logs do Pipeline</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Grid de passos usando columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h4>1. Exportar Dados</h4>
                    <p>Exporte eventos do Pipeline em TXT</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h4>2. Fazer Upload</h4>
                    <p>Carregue os arquivos TXT</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <h4>3. Analisar</h4>
                    <p>Explore métricas e insights</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                <div class="metric-card">
                    <h4>4. Clusterizar</h4>
                    <p>Agrupe eventos similares</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Funcionalidades principais
            st.markdown("""
            <div style="background: #F3E5F5; padding: 2rem; border-radius: 12px; margin-top: 2rem;">
                <h4 style="color: #5E35B1; margin-bottom: 1rem;">📋 Funcionalidades Principais</h4>
                <p style="color: #7E57C2; margin: 0.5rem 0;">• Análise de severidade automática</p>
                <p style="color: #7E57C2; margin: 0.5rem 0;">• Clusterização inteligente de eventos</p>
                <p style="color: #7E57C2; margin: 0.5rem 0;">• Análise temporal e de contexto</p>
                <p style="color: #7E57C2; margin: 0.5rem 0;">• Explorador detalhado com breadcrumbs</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
if __name__ == "__main__":
    main()