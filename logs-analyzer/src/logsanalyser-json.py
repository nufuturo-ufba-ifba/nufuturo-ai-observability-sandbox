import streamlit as st
import pandas as pd
import json
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import warnings
from datetime import datetime, timedelta
import re

warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    layout="wide", 
    page_title="Analisador de Eventos Sentry",
    page_icon="🐛",
    initial_sidebar_state="collapsed"
)

# CSS moderno e limpo com roxo claro
st.markdown("""
<style>
    /* Reset e base */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #F8F6FF;
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
    
    /* Event entries */
    .event-entry {
        font-family: 'Monaco', 'Menlo', monospace;
        font-size: 0.85rem;
        background: #F8F6FF;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    
    .critical-event {
        background: #FFEBEE;
        border-color: #F44336;
    }
    
    .high-event {
        background: #FFF3E0;
        border-color: #FF9800;
    }
    
    .medium-event {
        background: #FFF8E1;
        border-color: #FFC107;
    }
    
    .low-event {
        background: #E3F2FD;
        border-color: #2196F3;
    }
    
    /* Clusters */
    .cluster-0 { background: #F3E5F5; border-left: 4px solid #9C27B0; padding: 1rem; margin: 0.5rem 0; border-radius: 8px; }
    .cluster-1 { background: #E8EAF6; border-left: 4px solid #3F51B5; padding: 1rem; margin: 0.5rem 0; border-radius: 8px; }
    .cluster-2 { background: #E0F2F1; border-left: 4px solid #009688; padding: 1rem; margin: 0.5rem 0; border-radius: 8px; }
    .cluster-3 { background: #FFF3E0; border-left: 4px solid #FF9800; padding: 1rem; margin: 0.5rem 0; border-radius: 8px; }
    .cluster-4 { background: #FCE4EC; border-left: 4px solid #E91E63; padding: 1rem; margin: 0.5rem 0; border-radius: 8px; }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    
    .badge-critical { background: #FFEBEE; color: #C62828; border: 1px solid #FFCDD2; }
    .badge-high { background: #FFF3E0; color: #EF6C00; border: 1px solid #FFE0B2; }
    .badge-medium { background: #FFF8E1; color: #F57C00; border: 1px solid #FFECB3; }
    .badge-low { background: #E3F2FD; color: #1565C0; border: 1px solid #BBDEFB; }
    
    /* Seletores e inputs */
    .stSelectbox, .stMultiselect, .stTextInput {
        border-radius: 8px;
    }
    
    .stSelectbox div, .stMultiselect div {
        border-radius: 8px;
        border: 1px solid #D1C4E9;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: #F8F6FF;
        border-radius: 8px;
        border: 1px solid #F0EBFF;
        color: #5E35B1;
        font-weight: 500;
    }
    
    /* Sliders */
    .stSlider [data-baseweb="slider"] {
        color: #9575CD;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #B39DDB 0%, #D1C4E9 100%);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_json_files(uploaded_files):
    """
    Carrega múltiplos arquivos JSON e extrai os dados relevantes para um DataFrame.
    """
    all_data = []
    for file in uploaded_files:
        try:
            file.seek(0)
            data = json.load(file)
            
            exception_info = data.get('exception', {}).get('values', [{}])[0]
            contexts = data.get('contexts', {})
            device_info = contexts.get('device', {})
            os_info = contexts.get('os', {})
            app_info = contexts.get('app', {})

            tags_dict = {tag[0]: tag[1] for tag in data.get('tags', [])}

            # Extrair informações de breadcrumbs
            breadcrumbs = data.get('breadcrumbs', {}).get('values', [])
            
            # Análise de severidade baseada em múltiplos fatores
            severity_score = calculate_severity_score(data, exception_info, breadcrumbs)
            
            entry = {
                'event_id': data.get('event_id'),
                'timestamp': pd.to_datetime(data.get('datetime')),
                'release': data.get('release'),
                'platform': data.get('platform'),
                'error_type': exception_info.get('type'),
                'error_value': exception_info.get('value'),
                'os_name': os_info.get('name'),
                'os_version': os_info.get('version'),
                'device_model': device_info.get('model'),
                'device_family': device_info.get('family'),
                'nu_origin': tags_dict.get('nu.origin'),
                'nu_component': tags_dict.get('nu.component'),
                'nu_current_screen': tags_dict.get('nu.current_screen'),
                'breadcrumbs': breadcrumbs,
                'breadcrumbs_count': len(breadcrumbs),
                'severity_score': severity_score,
                'severity_level': get_severity_level(severity_score),
                'has_user_interaction': any('click' in str(bc.get('message', '')).lower() or 
                                          'tap' in str(bc.get('message', '')).lower() 
                                          for bc in breadcrumbs),
                'full_json': data
            }
            all_data.append(entry)
        except Exception as e:
            st.warning(f"Não foi possível ler o arquivo {file.name}: {e}")
            
    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    
    # Adicionar features derivadas para análise avançada
    if not df.empty:
        df['error_value_length'] = df['error_value'].fillna('').str.len()
        df['error_value_words'] = df['error_value'].fillna('').str.split().str.len()
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['is_weekend'] = df['timestamp'].dt.dayofweek >= 5
        df['error_complexity'] = df['error_value_length'] * df['breadcrumbs_count']
    
    return df

def calculate_severity_score(data, exception_info, breadcrumbs):
    """Calcula um score de severidade baseado em múltiplos fatores"""
    score = 0
    
    # Fator 1: Tipo de exceção
    error_type = exception_info.get('type', '').lower()
    critical_errors = ['nullpointer', 'outofmemory', 'stackoverflow', 'fatal', 'crash']
    if any(critical in error_type for critical in critical_errors):
        score += 3
    elif 'exception' in error_type:
        score += 1
    
    # Fator 2: Quantidade de breadcrumbs
    score += min(len(breadcrumbs) // 5, 3)
    
    # Fator 3: Presença de interações do usuário
    user_interactions = sum(1 for bc in breadcrumbs if 'click' in str(bc.get('message', '')).lower() or 
                          'tap' in str(bc.get('message', '')).lower())
    score += min(user_interactions, 2)
    
    # Fator 4: Comprimento da mensagem de erro
    error_value = exception_info.get('value', '')
    if len(error_value) > 100:
        score += 1
    
    return score

def get_severity_level(score):
    """Converte score de severidade para nível categorizado"""
    if score >= 5:
        return 'CRITICAL'
    elif score >= 3:
        return 'HIGH'
    elif score >= 1:
        return 'MEDIUM'
    else:
        return 'LOW'

def extract_features_for_clustering(df):
    """Extrai features para clusterização de eventos Sentry"""
    features = []
    
    # Codificar variáveis categóricas
    le_origin = LabelEncoder()
    le_component = LabelEncoder()
    
    origins_encoded = le_origin.fit_transform(df['nu_origin'].fillna('unknown'))
    components_encoded = le_component.fit_transform(df['nu_component'].fillna('unknown'))
    
    for idx, row in df.iterrows():
        feature_vector = [
            row['severity_score'],
            row['breadcrumbs_count'],
            int(row['has_user_interaction']),
            row['error_value_length'],
            row['error_value_words'],
            row['hour_of_day'],
            len(str(row['error_type'] or '')),
            origins_encoded[idx],
            components_encoded[idx],
            row['error_complexity'] / 1000 if row['error_complexity'] > 0 else 0
        ]
        features.append(feature_vector)
    
    return np.array(features)

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
    vectorizer = CountVectorizer(max_features=50, stop_words='english')
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

def analyze_lda_topics_sentry(lda_model, vectorizer, n_top_words=15):
    """Analisa os tópicos encontrados pelo LDA para eventos Sentry"""
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

def perform_clustering(df, n_clusters=4, method='kmeans'):
    """Realiza clusterização dos eventos Sentry"""
    if len(df) < n_clusters:
        st.warning(f"Poucas amostras para clusterização. Necessário pelo menos {n_clusters} amostras.")
        return df, None, None, None, None
    
    if method == 'lda':
        # Para LDA, usamos as mensagens de erro
        vectorizer = CountVectorizer(
            max_features=50, 
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        text_data = df['error_value'].fillna('').astype(str)
        X = vectorizer.fit_transform(text_data)
        
        lda = LatentDirichletAllocation(
            n_components=n_clusters, 
            random_state=42,
            max_iter=10,
            learning_method='online'
        )
        lda_features = lda.fit_transform(X)
        clusters = np.argmax(lda_features, axis=1)
        
        # Adicionar clusters ao DataFrame
        df_clustered = df.copy()
        df_clustered['cluster'] = clusters
        
        # Usar t-SNE para visualização
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(lda_features)
        
        silhouette_avg = -1  # LDA não tem silhouette score direto
        return df_clustered, features_2d, silhouette_avg, lda, vectorizer
        
    else:
        features = extract_features_for_clustering(df)
        
        # Normalizar features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Aplicar clusterização
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        else:  # dbscan
            clusterer = DBSCAN(eps=0.5, min_samples=2)
        
        clusters = clusterer.fit_predict(features_scaled)
        
        # Calcular métricas de qualidade
        if method == 'kmeans' and len(set(clusters)) > 1:
            silhouette_avg = silhouette_score(features_scaled, clusters)
        else:
            silhouette_avg = -1
        
        # Adicionar clusters ao DataFrame
        df_clustered = df.copy()
        df_clustered['cluster'] = clusters
        
        # Redução de dimensionalidade para visualização
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features_scaled)
        
        return df_clustered, features_2d, silhouette_avg, clusterer, None
    
def analyze_cluster_patterns(df_clustered):
    """Analisa padrões em cada cluster"""
    cluster_analysis = []
    
    for cluster_id in sorted(df_clustered['cluster'].unique()):
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
        
        analysis = {
            'cluster_id': cluster_id,
            'size': len(cluster_data),
            'avg_severity': cluster_data['severity_score'].mean(),
            'common_error_type': cluster_data['error_type'].mode().iloc[0] if not cluster_data['error_type'].mode().empty else 'N/A',
            'common_origin': cluster_data['nu_origin'].mode().iloc[0] if not cluster_data['nu_origin'].mode().empty else 'N/A',
            'avg_breadcrumbs': cluster_data['breadcrumbs_count'].mean(),
            'user_interaction_rate': cluster_data['has_user_interaction'].mean(),
            'sample_messages': cluster_data['error_value'].head(3).tolist()
        }
        
        cluster_analysis.append(analysis)
    
    return pd.DataFrame(cluster_analysis)

def perform_temporal_analysis(df):
    """Realiza análise temporal dos eventos"""
    temporal_analysis = {}
    
    # Análise por hora do dia
    hourly_dist = df.groupby('hour_of_day').size()
    temporal_analysis['peak_hour'] = hourly_dist.idxmax() if not hourly_dist.empty else 0
    temporal_analysis['peak_hour_count'] = hourly_dist.max() if not hourly_dist.empty else 0
    
    # Análise por dia da semana
    daily_dist = df.groupby('day_of_week').size()
    temporal_analysis['peak_day'] = daily_dist.idxmax() if not daily_dist.empty else 'N/A'
    
    # Tendência temporal
    df_sorted = df.sort_values('timestamp')
    if len(df_sorted) > 1:
        time_range = (df_sorted['timestamp'].max() - df_sorted['timestamp'].min()).total_seconds() / 3600
        events_per_hour = len(df_sorted) / max(time_range, 1)
        temporal_analysis['events_per_hour'] = events_per_hour
    else:
        temporal_analysis['events_per_hour'] = 0
    
    return temporal_analysis

def analyze_error_patterns(df):
    """Analisa padrões específicos de erro"""
    patterns = {}
    
    # Erros mais frequentes por componente
    component_errors = df.groupby(['nu_component', 'error_type']).size().reset_index(name='count')
    patterns['component_errors'] = component_errors.sort_values('count', ascending=False).head(10)
    
    # Relação entre tela e erro
    screen_errors = df.groupby(['nu_current_screen', 'error_type']).size().reset_index(name='count')
    patterns['screen_errors'] = screen_errors.sort_values('count', ascending=False).head(10)
    
    # Análise de severidade por origem
    severity_by_origin = df.groupby('nu_origin')['severity_score'].agg(['mean', 'count']).reset_index()
    patterns['severity_by_origin'] = severity_by_origin.sort_values('mean', ascending=False)
    
    # Correlação entre complexidade e severidade
    patterns['complexity_severity_corr'] = df['error_complexity'].corr(df['severity_score'])
    
    return patterns

def perform_text_analysis(df):
    """Análise de texto das mensagens de erro"""
    text_analysis = {}
    
    # Palavras mais comuns nas mensagens de erro
    all_messages = ' '.join(df['error_value'].fillna('').astype(str))
    words = re.findall(r'\b[a-zA-Z]{4,}\b', all_messages.lower())
    word_freq = Counter(words)
    text_analysis['common_words'] = word_freq.most_common(20)
    
    # Análise de sentenças complexas (erros longos)
    long_errors = df[df['error_value_length'] > 100]
    text_analysis['long_errors_count'] = len(long_errors)
    text_analysis['avg_error_length'] = df['error_value_length'].mean()
    
    return text_analysis

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
        <h1 style="color: #5E35B1; font-weight: 700; margin-bottom: 1rem;">🐛 Analisador de Eventos Sentry</h1>
        <p style="color: #7E57C2; font-size: 1.1rem; max-width: 600px; margin: 0 auto;">
            Analise eventos de erro do Sentry com clusterização inteligente e métricas avançadas
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Área de upload
    st.markdown("""
    <div class="main-container">
        <div class="upload-section">
            <h3 style="color: #5E35B1; margin-bottom: 1rem;">📁 Faça upload dos arquivos JSON do Sentry</h3>
            <p style="color: #7E57C2; margin-bottom: 2rem;">Arquivos JSON exportados do Sentry</p>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        " ",
        type=['json'],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    st.markdown("</div></div>", unsafe_allow_html=True)

    if uploaded_files:
        with st.spinner('Analisando eventos do Sentry...'):
            df = load_json_files(uploaded_files)

        if df.empty:
            st.error("Nenhum dado válido encontrado nos arquivos JSON.")
            return

        # Status do upload
        st.markdown(f"""
        <div class="main-container" style="background: #F3E5F5; border-left-color: #7E57C2;">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <span style="font-size: 1.5rem; color: #7E57C2;">✅</span>
                <div>
                    <h4 style="margin: 0; color: #5E35B1;">Eventos processados com sucesso!</h4>
                    <p style="margin: 0; color: #7E57C2;">{len(df)} eventos do Sentry carregados</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- Filtros Dinâmicos ---
        st.sidebar.markdown('<div class="main-container" style="padding: 1.5rem;">', unsafe_allow_html=True)
        st.sidebar.header("🔧 Filtros")
        
        releases = sorted(df['release'].dropna().unique())
        selected_release = st.sidebar.multiselect("Versão (Release)", options=releases, default=releases)
        
        platforms = sorted(df['os_name'].dropna().unique())
        selected_platform = st.sidebar.multiselect("Plataforma", options=platforms, default=platforms)

        error_types = sorted(df['error_type'].dropna().unique())
        selected_error_type = st.sidebar.multiselect("Tipo de Erro", options=error_types, default=error_types)

        origins = sorted(df['nu_origin'].dropna().unique())
        selected_origin = st.sidebar.multiselect("Origem (nu.origin)", options=origins, default=origins)
        
        severity_levels = sorted(df['severity_level'].dropna().unique())
        selected_severity = st.sidebar.multiselect("Nível de Severidade", options=severity_levels, default=severity_levels)
        
        filtered_df = df[
            df['release'].isin(selected_release) &
            df['os_name'].isin(selected_platform) &
            df['error_type'].isin(selected_error_type) &
            df['nu_origin'].isin(selected_origin) &
            df['severity_level'].isin(selected_severity)
        ]
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        
        # --- Dashboard de Métricas ---
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📊 Resumo dos Eventos</div>', unsafe_allow_html=True)
        
        total_events = len(df)
        filtered_events = len(filtered_df)
        critical_events = len(filtered_df[filtered_df['severity_level'] == 'CRITICAL'])
        avg_severity = filtered_df['severity_score'].mean()
        unique_errors = filtered_df['error_type'].nunique()

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(create_metric_card(total_events, "Eventos Totais", "📊"), unsafe_allow_html=True)
        with col2:
            st.markdown(create_metric_card(filtered_events, "Eventos Filtrados", "✅"), unsafe_allow_html=True)
        with col3:
            st.markdown(create_metric_card(critical_events, "Eventos Críticos", "🔥"), unsafe_allow_html=True)
        with col4:
            st.markdown(create_metric_card(f"{avg_severity:.1f}", "Severidade Média", "⚠️"), unsafe_allow_html=True)
        with col5:
            st.markdown(create_metric_card(unique_errors, "Tipos de Erro", "🎯"), unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

        # --- Abas Principais ---
        if not filtered_df.empty:
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Análise Geral", 
                "🤖 Clusterização", 
                "📊 Estatísticas Avançadas", 
                "🗺️ Contexto"
            ])
            
            with tab1:
                st.markdown('<div class="main-container">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">📈 Análise Geral de Erros</div>', unsafe_allow_html=True)
                
                col1_chart, col2_chart = st.columns(2)
                
                with col1_chart:
                    error_counts = filtered_df['error_type'].value_counts().head(10)
                    fig_bar = px.bar(
                        x=error_counts.values, 
                        y=error_counts.index,
                        orientation='h',
                        title="Top 10 Tipos de Erro",
                        labels={'x': 'Contagem', 'y': 'Tipo de Erro'},
                        color=error_counts.values,
                        color_continuous_scale='purples'
                    )
                    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                with col2_chart:
                    severity_counts = filtered_df['severity_level'].value_counts()
                    fig_pie = px.pie(
                        values=severity_counts.values, 
                        names=severity_counts.index, 
                        title="Distribuição por Nível de Severidade",
                        color=severity_counts.index,
                        color_discrete_map={'CRITICAL':'#F44336', 'HIGH':'#FF9800', 'MEDIUM':'#FFC107', 'LOW':'#2196F3'}
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Timeline de eventos
                st.markdown('<div class="section-header">📅 Timeline de Eventos</div>', unsafe_allow_html=True)
                timeline_data = filtered_df.set_index('timestamp').resample('1H').size().reset_index(name='count')
                fig_timeline = px.line(
                    timeline_data, 
                    x='timestamp', 
                    y='count', 
                    title='Eventos por Hora',
                    color_discrete_sequence=['#9575CD']
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab2:
                st.markdown('<div class="main-container">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">🤖 Clusterização de Eventos</div>', unsafe_allow_html=True)
                
                cluster_method = st.selectbox(
                    "Método de Clusterização", 
                    options=['kmeans', 'dbscan', 'lda'],
                    help="KMeans: grupos por features, DBSCAN: detecção automática de outliers, LDA: análise de tópicos em mensagens de erro"
                )
                
                if cluster_method == 'kmeans':
                    # Para KMeans, calcular número ótimo de clusters
                    with st.spinner("Calculando número ótimo de clusters..."):
                        features = extract_features_for_clustering(filtered_df)
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
                        text_data = filtered_df['error_value'].fillna('').astype(str)
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
                        df_clustered, features_2d, silhouette_score, clusterer, vectorizer = perform_clustering(filtered_df, n_clusters, cluster_method)
                        
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
                                    'error_type': df_clustered['error_type'],
                                    'severity': df_clustered['severity_level'],
                                    'message': df_clustered['error_value'].str[:50] + '...'
                                })
                                
                                if cluster_method == 'lda':
                                    title = 'Visualização 2D dos Tópicos LDA (t-SNE)'
                                elif cluster_method == 'dbscan':
                                    title = 'Visualização 2D dos Clusters DBSCAN'
                                else:
                                    title = 'Visualização 2D dos Clusters KMeans'
                                
                                fig_clusters = px.scatter(
                                    viz_df, x='x', y='y', color='cluster',
                                    hover_data=['error_type', 'severity', 'message'],
                                    title=title,
                                    color_continuous_scale='purples'
                                )
                                st.plotly_chart(fig_clusters, use_container_width=True)
                            
                            # Análise específica para LDA
                            if cluster_method == 'lda' and hasattr(clusterer, 'components_'):
                                st.markdown("#### 📝 Análise de Tópicos LDA")
                                
                                topics = analyze_lda_topics_sentry(clusterer, vectorizer)
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    for topic in topics:
                                        with st.expander(f"Tópico {topic['topic_id']}"):
                                            st.write(f"**Palavras-chave:** {topic['top_words']}")
                                            
                                            # Mostrar exemplos de mensagens deste tópico
                                            topic_messages = df_clustered[df_clustered['cluster'] == topic['topic_id']]['error_value'].head(3)
                                            if len(topic_messages) > 0:
                                                st.write("**Exemplos de mensagens:**")
                                                for msg in topic_messages:
                                                    st.write(f"- {msg[:100]}...")
                                
                                with col2:
                                    # Gráfico de importância das palavras
                                    fig = go.Figure()
                                    for topic in topics:
                                        fig.add_trace(go.Bar(
                                            y=topic['top_features'][:10],  # Top 10 palavras
                                            x=topic['word_weights'][:10],
                                            name=f"Tópico {topic['topic_id']}",
                                            orientation='h'
                                        ))
                                    
                                    fig.update_layout(
                                        title="Top 10 Palavras por Tópico",
                                        xaxis_title="Importância",
                                        yaxis_title="Palavras",
                                        barmode='group',
                                        height=400
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            # Análise dos clusters (para todos os métodos)
                            st.markdown("#### 📈 Análise dos Clusters")
                            cluster_analysis = analyze_cluster_patterns(df_clustered)
                            st.dataframe(cluster_analysis, use_container_width=True)
                            
                            # Exemplos por cluster
                            st.markdown("#### 🔍 Exemplos por Cluster")
                            for cluster_id in sorted(df_clustered['cluster'].unique()):
                                cluster_size = len(df_clustered[df_clustered['cluster'] == cluster_id])
                                if cluster_method == 'dbscan' and cluster_id == -1:
                                    with st.expander(f"Outliers - {cluster_size} eventos"):
                                        cluster_samples = df_clustered[df_clustered['cluster'] == cluster_id].head(3)
                                        for _, sample in cluster_samples.iterrows():
                                            st.markdown(f"<div class='cluster-{cluster_id % 5}'>"
                                                    f"<strong>[{sample['severity_level']}] {sample['error_type']}</strong><br>"
                                                    f"{sample['error_value']}</div>", 
                                                    unsafe_allow_html=True)
                                else:
                                    with st.expander(f"Cluster {cluster_id} - {cluster_size} eventos"):
                                        cluster_samples = df_clustered[df_clustered['cluster'] == cluster_id].head(3)
                                        for _, sample in cluster_samples.iterrows():
                                            st.markdown(f"<div class='cluster-{cluster_id % 5}'>"
                                                    f"<strong>[{sample['severity_level']}] {sample['error_type']}</strong><br>"
                                                    f"{sample['error_value']}</div>", 
                                                    unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab3:
                st.markdown('<div class="main-container">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">📊 Estatísticas Avançadas</div>', unsafe_allow_html=True)
                
                # Análise temporal
                temporal_analysis = perform_temporal_analysis(filtered_df)
                
                col1_temp, col2_temp, col3_temp = st.columns(3)
                with col1_temp:
                    st.metric("🕒 Hora de Pico", f"{temporal_analysis['peak_hour']}:00")
                with col2_temp:
                    st.metric("📈 Eventos na Hora de Pico", temporal_analysis['peak_hour_count'])
                with col3_temp:
                    st.metric("📅 Dia com Mais Eventos", temporal_analysis['peak_day'])
                
                # Análise de padrões
                error_patterns = analyze_error_patterns(filtered_df)
                
                st.markdown('<div class="section-header">🔧 Padrões por Componente</div>', unsafe_allow_html=True)
                st.dataframe(error_patterns['component_errors'], use_container_width=True)
                
                st.markdown('<div class="section-header">📈 Severidade por Origem</div>', unsafe_allow_html=True)
                st.dataframe(error_patterns['severity_by_origin'], use_container_width=True)
                
                # Análise de texto
                text_analysis = perform_text_analysis(filtered_df)
                
                st.markdown('<div class="section-header">📝 Análise de Texto</div>', unsafe_allow_html=True)
                col_text1, col_text2 = st.columns(2)
                
                with col_text1:
                    common_words_df = pd.DataFrame(text_analysis['common_words'], columns=['Palavra', 'Frequência'])
                    fig_words = px.bar(
                        common_words_df.head(10), 
                        x='Frequência', 
                        y='Palavra', 
                        orientation='h',
                        title='Top 10 Palavras nas Mensagens de Erro',
                        color='Frequência',
                        color_continuous_scale='purples'
                    )
                    st.plotly_chart(fig_words, use_container_width=True)
                
                with col_text2:
                    st.metric("📝 Erros Longos (>100 chars)", text_analysis['long_errors_count'])
                    st.metric("📏 Comprimento Médio", f"{text_analysis['avg_error_length']:.1f} chars")
                    st.metric("🔗 Correlação Complexidade-Severidade", f"{error_patterns['complexity_severity_corr']:.3f}")
                
                # Distribuição por hora do dia
                st.markdown('<div class="section-header">🕒 Distribuição por Hora do Dia</div>', unsafe_allow_html=True)
                hourly_data = filtered_df['hour_of_day'].value_counts().sort_index()
                fig_hourly = px.bar(
                    x=hourly_data.index, 
                    y=hourly_data.values,
                    title='Distribuição de Eventos por Hora do Dia',
                    labels={'x': 'Hora do Dia', 'y': 'Número de Eventos'},
                    color_discrete_sequence=['#B39DDB']
                )
                st.plotly_chart(fig_hourly, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab4:
                st.markdown('<div class="main-container">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">🗺️ Análise de Contexto</div>', unsafe_allow_html=True)
                col1_context, col2_context = st.columns(2)
                
                with col1_context:
                    device_counts = filtered_df['device_family'].value_counts().head(10)
                    fig_device = px.bar(
                        y=device_counts.index, 
                        x=device_counts.values,
                        orientation='h', 
                        title="Top 10 Dispositivos com Erros",
                        labels={'y': 'Família do Dispositivo', 'x': 'Contagem'},
                        color=device_counts.values,
                        color_continuous_scale='purples'
                    )
                    fig_device.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_device, use_container_width=True)
                
                with col2_context:
                    screen_counts = filtered_df['nu_current_screen'].value_counts().head(10)
                    fig_screen = px.bar(
                        y=screen_counts.index, 
                        x=screen_counts.values,
                        orientation='h', 
                        title="Top 10 Telas com Erros",
                        labels={'y': 'Tela Atual', 'x': 'Contagem'},
                        color=screen_counts.values,
                        color_continuous_scale='purples'
                    )
                    fig_screen.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_screen, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Explorador de Eventos
            st.markdown('<div class="main-container">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">🔍 Explorador de Eventos Detalhados</div>', unsafe_allow_html=True)
            
            # Filtros adicionais para o explorador
            col_search1, col_search2 = st.columns(2)
            with col_search1:
                search_term = st.text_input("Buscar na mensagem de erro:")
            with col_search2:
                min_severity = st.slider("Severidade Mínima", 0, 10, 0)
            
            explorer_df = filtered_df[filtered_df['severity_score'] >= min_severity]
            if search_term:
                explorer_df = explorer_df[explorer_df['error_value'].str.contains(search_term, case=False, na=False)]
            
            st.write(f"**Mostrando {len(explorer_df)} de {len(filtered_df)} eventos**")
            
            # Paginação
            page_size = st.slider("Eventos por página:", 5, 20, 10)
            total_pages = max(1, len(explorer_df) // page_size + 1)
            page = st.number_input("Página", min_value=1, max_value=total_pages, value=1)
            
            start_idx = (page - 1) * page_size
            end_idx = min(start_idx + page_size, len(explorer_df))
            
            for _, row in explorer_df.iloc[start_idx:end_idx].iterrows():
                severity_class = f"{row['severity_level'].lower()}-event"
                timestamp_str = row.get('timestamp', 'N/A')
                if hasattr(timestamp_str, 'strftime'):
                    timestamp_str = timestamp_str.strftime('%Y-%m-%d %H:%M:%S')
                    
                with st.expander(f"**{timestamp_str}** | **[{row['severity_level']}]** | **{row['error_type']}** | {row['error_value'][:80]}..."):
                    st.markdown(f"**Origem:** `{row['nu_origin']}` | **Componente:** `{row['nu_component']}` | **Tela:** `{row['nu_current_screen']}`")
                    st.markdown(f"**Score de Severidade:** {row['severity_score']} | **Breadcrumbs:** {row['breadcrumbs_count']}")
                    
                    st.markdown("##### Breadcrumbs (Jornada do Usuário)")
                    if row['breadcrumbs']:
                        breadcrumbs_df = pd.DataFrame(row['breadcrumbs'])
                        required_cols = ['timestamp', 'category', 'message', 'level', 'data']
                        for col in required_cols:
                            if col not in breadcrumbs_df.columns:
                                breadcrumbs_df[col] = "N/A"
                        st.dataframe(breadcrumbs_df[required_cols].tail(15), use_container_width=True)
                    else:
                        st.write("Nenhum breadcrumb encontrado para este evento.")
                    
                    st.markdown("##### Detalhes do Evento (JSON Completo)")
                    st.json(row['full_json'])
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("Nenhum dado corresponde aos filtros selecionados.")

    else:
        # Tela inicial
        with st.container():
            st.markdown('<div class="main-container">', unsafe_allow_html=True)
            
            st.markdown("""
            <div style="text-align: center;">
                <h2 style="color: #5E35B1; margin-bottom: 2rem;">Como Usar o Analisador de logs do Sentry</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Grid de passos usando columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h4>1. Exportar Dados</h4>
                    <p>Exporte eventos do Sentry em JSON</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h4>2. Fazer Upload</h4>
                    <p>Carregue os arquivos CSV</p>
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