import streamlit as st
import pandas as pd
import json
import ast
from collections import Counter, defaultdict
import csv
import io
import re
import numpy as np
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

# Configuração da página
st.set_page_config(
    layout="wide", 
    page_title="Analisador de Logs CSV",
    page_icon="📊",
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
    
    /* Status badges */
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
    
    /* Dataframes */
    .dataframe {
        border-radius: 12px;
        border: 1px solid #F0EBFF;
    }
    
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

def parse_json_string(json_str):
    """Tenta parsear uma string JSON de várias formas"""
    if not json_str or not isinstance(json_str, str):
        return None
    
    json_str_clean = json_str.strip()
    if json_str_clean.startswith('"') and json_str_clean.endswith('"'):
        json_str_clean = json_str_clean[1:-1]
    
    try:
        return json.loads(json_str_clean)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(json_str_clean)
        except:
            try:
                json_str_fixed = re.sub(r"'([^']*)'", r'"\1"', json_str_clean)
                json_str_fixed = re.sub(r',\s*}', '}', json_str_fixed)
                json_str_fixed = re.sub(r',\s*]', ']', json_str_fixed)
                return json.loads(json_str_fixed)
            except:
                return None

def read_problematic_csv(uploaded_file):
    """Lê CSV com problemas de formatação usando métodos robustos"""
    content = uploaded_file.read().decode('utf-8', errors='ignore')
    lines = content.split('\n')
    
    # Tenta detectar o delimitador
    first_line = lines[0] if lines else ''
    delimiter = ','
    if ';' in first_line and first_line.count(';') > first_line.count(','):
        delimiter = ';'
    elif '\t' in first_line:
        delimiter = '\t'
    
    # Método 1: Pandas com engine python (mais tolerante)
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, delimiter=delimiter, engine='python', 
                        quoting=csv.QUOTE_ALL, on_bad_lines='warn')
        return df, "pandas"
    except Exception as e:
        st.warning(f"Pandas method failed: {e}")
    
    # Método 2: Leitura manual com csv.reader
    try:
        data = []
        for line in lines:
            if line.strip():
                try:
                    reader = csv.reader([line], delimiter=delimiter, quotechar='"')
                    for row in reader:
                        if row:
                            data.append(row)
                except:
                    # Fallback: split simples por delimitador
                    row = line.split(delimiter)
                    data.append(row)
        
        # Encontra número máximo de colunas
        max_cols = max(len(row) for row in data) if data else 0
        
        # Preenche linhas com menos colunas
        for row in data:
            while len(row) < max_cols:
                row.append(None)
            while len(row) > max_cols:
                row.pop()
        
        # Cria DataFrame
        if data:
            headers = data[0] if len(data[0]) == max_cols else [f'col_{i}' for i in range(max_cols)]
            df = pd.DataFrame(data[1:], columns=headers)
            return df, "manual"
    except Exception as e:
        st.warning(f"Manual method failed: {e}")
    
    return None, "error"

def extract_all_fields(data, parent_key='', separator='.'):
    """Extrai recursivamente todos os campos de um JSON e remove prefixos 'data.'"""
    fields = []
    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{parent_key}{separator}{key}" if parent_key else key
            # Remove prefixos 'data.' duplicados
            new_key = re.sub(r'^data\.data\.', 'data.', new_key)
            new_key = re.sub(r'^data\.', '', new_key)  # Remove todos os prefixos 'data.'
            
            if isinstance(value, (dict, list)):
                fields.extend(extract_all_fields(value, new_key, separator))
            else:
                fields.append(new_key)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, (dict, list)):
                fields.extend(extract_all_fields(item, parent_key, separator))
    else:
        if parent_key:
            # Remove prefixos 'data.' também para campos simples
            parent_key = re.sub(r'^data\.data\.', 'data.', parent_key)
            parent_key = re.sub(r'^data\.', '', parent_key)
            fields.append(parent_key)
    
    return list(set(fields))  # Remove duplicatas

def extract_field_value(data, field_path):
    """Extrai valor de campo aninhado, lidando com prefixos 'data.'"""
    try:
        # Adiciona prefixo 'data.' se necessário para campos comuns
        if not field_path.startswith('data.') and 'data.' in str(data):
            # Tenta com prefixo data. primeiro
            try:
                partes = f"data.{field_path}".split('.')
                current = data
                for parte in partes:
                    if isinstance(current, dict) and parte in current:
                        current = current[parte]
                    else:
                        current = None
                        break
                if current is not None:
                    return current
            except:
                pass
        
        # Tenta sem prefixo
        if '.' in field_path:
            partes = field_path.split('.')
            current = data
            for parte in partes:
                if isinstance(current, dict) and parte in current:
                    current = current[parte]
                else:
                    return None
            return current
        else:
            return data.get(field_path)
    except:
        return None

def discover_json_structure(df, coluna_json='log', sample_size=50):
    """Descobre automaticamente a estrutura dos JSONs"""
    all_fields = set()
    field_examples = defaultdict(set)
    field_types = defaultdict(set)
    
    sample_df = df.head(min(sample_size, len(df)))
    
    for idx, row in sample_df.iterrows():
        if pd.notna(row[coluna_json]):
            json_data = parse_json_string(str(row[coluna_json]))
            if json_data and isinstance(json_data, dict):
                fields = extract_all_fields(json_data)
                for field in fields:
                    all_fields.add(field)
                    value = extract_field_value(json_data, field)
                    if value is not None:
                        field_examples[field].add(str(value))
                        field_types[field].add(type(value).__name__)
    
    return sorted(all_fields), field_examples, field_types

def processar_csv_automatico(df, campos_selecionados, coluna_json='log'):
    """Processa CSV extraindo campos automaticamente"""
    try:
        associacoes = defaultdict(Counter)
        dados_completos = []
        linhas_processadas = len(df)
        linhas_com_json = 0
        
        for idx, row in df.iterrows():
            if pd.notna(row[coluna_json]):
                json_data = parse_json_string(str(row[coluna_json]))
                
                if json_data and isinstance(json_data, dict):
                    linhas_com_json += 1
                    linha_dados = {}
                    
                    for campo in campos_selecionados:
                        valor = extract_field_value(json_data, campo)
                        if valor is not None:
                            # Remove prefixos 'data.' do nome da coluna no DataFrame final
                            campo_clean = re.sub(r'^data\.', '', campo)
                            linha_dados[campo_clean] = valor
                            associacoes[campo_clean][str(valor)] += 1
                    
                    dados_completos.append(linha_dados)
        
        df_completo = pd.DataFrame(dados_completos)
        return associacoes, df_completo, linhas_processadas, linhas_com_json
        
    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
        return None, None, 0, 0

def create_metric_card(value, label, icon="📊"):
    """Cria um card de métrica estilizado"""
    return f"""
    <div class="metric-card">
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """

def prepare_data_for_clustering(df, selected_columns):
    """Prepara dados para clusterização"""
    if not selected_columns:
        return None, "Nenhuma coluna selecionada"
    
    clustering_df = df[selected_columns].copy()
    
    # Prepara dados numéricos
    numeric_cols = clustering_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Prepara dados categóricos (converte para numérico)
    categorical_cols = clustering_df.select_dtypes(include=['object']).columns.tolist()
    
    # Codifica variáveis categóricas
    for col in categorical_cols:
        if col in clustering_df.columns:
            le = LabelEncoder()
            try:
                clustering_df[col] = le.fit_transform(clustering_df[col].astype(str))
                numeric_cols.append(col)
            except:
                clustering_df = clustering_df.drop(columns=[col])
    
    if len(numeric_cols) == 0:
        return None, "Nenhuma coluna numérica disponível para clusterização"
    
    # Remove colunas com muitos valores faltantes
    clustering_df = clustering_df.dropna(thresh=len(clustering_df) * 0.5, axis=1)
    
    if clustering_df.empty:
        return None, "Dados insuficientes após limpeza"
    
    # Preenche valores faltantes
    for col in clustering_df.columns:
        if clustering_df[col].dtype in [np.float64, np.int64]:
            clustering_df[col] = clustering_df[col].fillna(clustering_df[col].median())
        else:
            clustering_df[col] = clustering_df[col].fillna(clustering_df[col].mode()[0] if not clustering_df[col].mode().empty else 0)
    
    # Normaliza os dados
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_df)
    
    return {
        'scaled_data': scaled_data,
        'original_df': clustering_df,
        'feature_names': clustering_df.columns.tolist(),
        'scaler': scaler
    }, None

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

def perform_advanced_clustering(data, n_clusters, method='kmeans', text_data=None, feature_names=None):
    """Executa clusterização com diferentes métodos"""
    
    if method == 'lda':
        # Para LDA, precisamos de dados textuais
        if text_data is None:
            return None, "LDA requer dados textuais para análise"
        
        vectorizer = CountVectorizer(
            max_features=100, 
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        X = vectorizer.fit_transform(text_data)
        
        lda = LatentDirichletAllocation(
            n_components=n_clusters, 
            random_state=42,
            max_iter=10,
            learning_method='online'
        )
        lda_features = lda.fit_transform(X)
        clusters = np.argmax(lda_features, axis=1)
        
        # Métricas para LDA são diferentes
        silhouette_avg = -1
        calinski_score = -1
        davies_score = -1
        
        return {
            'clusters': clusters,
            'model': lda,
            'metrics': {
                'silhouette': silhouette_avg,
                'calinski_harabasz': calinski_score,
                'davies_bouldin': davies_score
            },
            'vectorizer': vectorizer,
            'method': 'lda',
            'features_2d': TSNE(n_components=2, random_state=42).fit_transform(lda_features)
        }
        
    else:
        # KMeans e DBSCAN tradicionais
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        else:  # dbscan
            clusterer = DBSCAN(eps=0.5, min_samples=2)
        
        clusters = clusterer.fit_predict(data)
        
        # Calcula métricas de avaliação
        if len(set(clusters)) > 1 and method == 'kmeans':
            silhouette_avg = silhouette_score(data, clusters)
            calinski_score = calinski_harabasz_score(data, clusters)
            davies_score = davies_bouldin_score(data, clusters)
        else:
            silhouette_avg = -1
            calinski_score = -1
            davies_score = -1
        
        return {
            'clusters': clusters,
            'model': clusterer,
            'metrics': {
                'silhouette': silhouette_avg,
                'calinski_harabasz': calinski_score,
                'davies_bouldin': davies_score
            },
            'method': method,
            'features_2d': PCA(n_components=2).fit_transform(data)
        }

def prepare_text_data_for_lda(df, selected_columns):
    """Prepara dados textuais para LDA"""
    text_columns = df[selected_columns].select_dtypes(include=['object']).columns
    
    if len(text_columns) == 0:
        return None, "Nenhuma coluna textual encontrada para LDA"
    
    # Combina o texto de todas as colunas textuais
    text_data = df[text_columns].astype(str).agg(' '.join, axis=1)
    
    return text_data, None

def analyze_lda_topics(lda_model, vectorizer, n_top_words=10):
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

def create_cluster_analysis(df, clustering_result, feature_names):
    """Cria análise detalhada dos clusters"""
    df_analysis = df.copy()
    df_analysis['cluster'] = clustering_result['clusters']
    
    cluster_profiles = {}
    
    for cluster_id in sorted(df_analysis['cluster'].unique()):
        cluster_data = df_analysis[df_analysis['cluster'] == cluster_id]
        
        profile = {
            'size': len(cluster_data),
            'percentage': (len(cluster_data) / len(df_analysis)) * 100,
            'characteristics': {}
        }
        
        # Analisa cada característica do cluster
        for feature in feature_names:
            if feature in cluster_data.columns:
                if cluster_data[feature].dtype in ['object', 'category']:
                    value_counts = cluster_data[feature].value_counts()
                    if not value_counts.empty:
                        profile['characteristics'][feature] = {
                            'type': 'categorical',
                            'top_value': value_counts.index[0],
                            'top_percentage': (value_counts.iloc[0] / len(cluster_data)) * 100
                        }
                else:
                    profile['characteristics'][feature] = {
                        'type': 'numeric',
                        'mean': cluster_data[feature].mean(),
                        'median': cluster_data[feature].median(),
                        'std': cluster_data[feature].std()
                    }
        
        cluster_profiles[f'Cluster {cluster_id}'] = profile
    
    return cluster_profiles

def main():
    # Header principal
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: #5E35B1; font-weight: 700; margin-bottom: 1rem;">📊 Analisador de Logs CSV</h1>
        <p style="color: #7E57C2; font-size: 1.1rem; max-width: 600px; margin: 0 auto;">
            Analise e extraia insights de arquivos CSV contendo logs em formato JSON
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Área de upload
    st.markdown("""
    <div class="main-container">
        <div class="upload-section">
            <h3 style="color: #5E35B1; margin-bottom: 1rem;">📁 Faça upload do arquivo CSV</h3>
            <p style="color: #7E57C2; margin-bottom: 2rem;">Arquivos CSV que podem conter logs em formato JSON</p>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        " ",
        type=['csv'],
        help="Arquivo CSV que pode conter problemas de formatação",
        label_visibility="collapsed"
    )
    
    st.markdown("</div></div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        with st.spinner('Lendo arquivo CSV...'):
            df, method = read_problematic_csv(uploaded_file)
        
        if df is not None:
            # Status do upload
            st.markdown(f"""
            <div class="main-container" style="background: #F3E5F5; border-left-color: #7E57C2;">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <span style="font-size: 1.5rem; color: #7E57C2;">✅</span>
                    <div>
                        <h4 style="margin: 0; color: #5E35B1;">CSV carregado com sucesso!</h4>
                        <p style="margin: 0; color: #7E57C2;">Método utilizado: {method} | {len(df)} linhas × {len(df.columns)} colunas</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📋 Visualizar primeiras linhas"):
                st.dataframe(df.head(3), use_container_width=True)
            
            possible_json_cols = [col for col in df.columns if any(x in str(col).lower() for x in ['log', 'json', 'data', 'message'])]
            if not possible_json_cols and len(df.columns) > 0:
                possible_json_cols = [df.columns[0]]
            
            coluna_json = st.selectbox(
                "📝 Selecione a coluna que contém o JSON",
                options=df.columns.tolist(),
                index=df.columns.get_loc(possible_json_cols[0]) if possible_json_cols else 0
            )
            
            with st.spinner('Analisando estrutura dos JSONs...'):
                campos_disponiveis, exemplos_campos, tipos_campos = discover_json_structure(df, coluna_json)
            
            if campos_disponiveis:
                st.markdown('<div class="main-container">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">🎯 Campos Disponíveis para Análise</div>', unsafe_allow_html=True)
                
                st.success(f"Encontrados {len(campos_disponiveis)} campos nos JSONs!")
                
                filtro_campo = st.text_input("🔍 Filtrar campos", placeholder="Digite para filtrar...")
                campos_filtrados = [campo for campo in campos_disponiveis if filtro_campo.lower() in campo.lower()]
                
                st.markdown("#### Selecione os campos para análise:")
                
                col1, col2 = st.columns(2)
                campos_selecionados = []
                
                with col1:
                    for i, campo in enumerate(campos_filtrados[:len(campos_filtrados)//2]):
                        tipo_info = f" ({', '.join(tipos_campos[campo])})" if campo in tipos_campos else ""
                        if st.checkbox(f"`{campo}`{tipo_info}", key=f"campo_{i}", value=True):
                            campos_selecionados.append(campo)
                
                with col2:
                    for i, campo in enumerate(campos_filtrados[len(campos_filtrados)//2:]):
                        idx = i + len(campos_filtrados)//2
                        tipo_info = f" ({', '.join(tipos_campos[campo])})" if campo in tipos_campos else ""
                        if st.checkbox(f"`{campo}`{tipo_info}", key=f"campo_{idx}", value=True):
                            campos_selecionados.append(campo)
                
                if campos_selecionados and st.button("🚀 Iniciar Análise Completa", type="primary", use_container_width=True):
                    with st.spinner('Processando dados e gerando análises...'):
                        st.session_state.analysis_result = processar_csv_automatico(
                            df, campos_selecionados, coluna_json
                        )
                        st.session_state.analysis_started = True
                st.markdown('</div>', unsafe_allow_html=True)

                if st.session_state.get('analysis_started', False) and st.session_state.get('analysis_result') is not None:
                    associacoes, df_completo, total_linhas, linhas_json = st.session_state.analysis_result
                    
                    if associacoes and df_completo is not None and not df_completo.empty:
                        
                        # Tabs principais
                        tab1, tab2, tab3, tab4, tab5 = st.tabs([
                            "📈 Visão Geral", 
                            "📊 Estatísticas", 
                            "👥 Personas", 
                            "🔍 Análises Avançadas", 
                            "🤖 Clusterização"
                        ])

                        with tab1:
                            st.markdown('<div class="main-container">', unsafe_allow_html=True)
                            st.markdown('<div class="section-header">📈 Visão Geral dos Dados</div>', unsafe_allow_html=True)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.markdown(create_metric_card(total_linhas, "Total de Linhas", "📄"), unsafe_allow_html=True)
                            with col2:
                                st.markdown(create_metric_card(linhas_json, "Linhas com JSON", "✅"), unsafe_allow_html=True)
                            with col3:
                                st.markdown(create_metric_card(len(campos_selecionados), "Campos Analisados", "🎯"), unsafe_allow_html=True)
                            with col4:
                                st.markdown(create_metric_card(df_completo.nunique().sum(), "Valores Únicos", "🔤"), unsafe_allow_html=True)
                            
                            st.markdown("#### 📋 Dados Processados")
                            st.dataframe(df_completo.head(10), use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with tab2:
                            st.markdown('<div class="main-container">', unsafe_allow_html=True)
                            st.markdown('<div class="section-header">📊 Análises Estatísticas</div>', unsafe_allow_html=True)
                            
                            st.markdown("#### 📈 Estatísticas Descritivas")
                            st.dataframe(df_completo.describe(include='all'), use_container_width=True)
                            
                            numeric_cols = df_completo.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 1:
                                st.markdown("#### 🔗 Matriz de Correlação")
                                corr_matrix = df_completo[numeric_cols].corr()
                                fig = px.imshow(
                                    corr_matrix,
                                    title="Matriz de Correlação",
                                    color_continuous_scale='Purples',
                                    aspect="auto"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown("#### 📊 Distribuições")
                            for col in df_completo.columns:
                                with st.expander(f"Distribuição de {col}"):
                                    if df_completo[col].dtype == 'object':
                                        fig = px.histogram(df_completo, x=col, title=f"Distribuição de {col}",
                                                          color_discrete_sequence=['#B39DDB'])
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        fig = px.histogram(df_completo, x=col, title=f"Distribuição de {col}",
                                                          color_discrete_sequence=['#9575CD'])
                                        st.plotly_chart(fig, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)

                        with tab3:
                            st.markdown('<div class="main-container">', unsafe_allow_html=True)
                            st.markdown('<div class="section-header">👥 Análise de Personas</div>', unsafe_allow_html=True)
                            
                            def create_persona_analysis(df, clustering_fields):
                                """Cria análise de personas baseada em clustering"""
                                if not clustering_fields:
                                    return None, "Selecione campos para análise de personas"
                                
                                analysis_df = df[clustering_fields].copy()
                                
                                for col in analysis_df.columns:
                                    if analysis_df[col].dtype == 'object':
                                        le = LabelEncoder()
                                        analysis_df[col] = le.fit_transform(analysis_df[col].astype(str))
                                
                                analysis_df = analysis_df.fillna(analysis_df.mean())
                                
                                scaler = StandardScaler()
                                scaled_data = scaler.fit_transform(analysis_df)
                                
                                kmeans = KMeans(n_clusters=min(5, len(analysis_df)), random_state=42)
                                personas = kmeans.fit_predict(scaled_data)
                                
                                df_persona = df.copy()
                                df_persona['persona'] = personas
                                
                                persona_profiles = {}
                                for persona in range(len(set(personas))):
                                    persona_data = df_persona[df_persona['persona'] == persona]
                                    profile = {
                                        'size': len(persona_data),
                                        'characteristics': {}
                                    }
                                    
                                    for col in clustering_fields:
                                        if persona_data[col].dtype == 'object':
                                            profile['characteristics'][col] = persona_data[col].mode().iloc[0] if not persona_data[col].mode().empty else 'N/A'
                                        else:
                                            profile['characteristics'][col] = persona_data[col].mean()
                                    
                                    persona_profiles[f'Persona {persona + 1}'] = profile
                                
                                return persona_profiles, None

                            def perform_clustering(df, n_clusters=3):
                                """Realiza clustering K-means"""
                                numeric_df = df.select_dtypes(include=[np.number])
                                
                                if len(numeric_df.columns) < 2:
                                    return None, None, "Mínimo 2 colunas numéricas necessárias para clustering"
                                
                                numeric_df = numeric_df.fillna(numeric_df.mean())
                                scaler = StandardScaler()
                                scaled_data = scaler.fit_transform(numeric_df)
                                
                                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                                clusters = kmeans.fit_predict(scaled_data)
                                
                                return clusters, kmeans, None

                            campos_persona = st.multiselect(
                                "Selecione campos para análise de personas",
                                options=df_completo.columns.tolist(),
                                default=df_completo.columns.tolist()[:min(5, len(df_completo.columns))],
                                help="Escolha campos que representem características distintas para formação de personas"
                            )
                            
                            if campos_persona:
                                with st.spinner("Criando análise de personas..."):
                                    personas, error = create_persona_analysis(df_completo, campos_persona)
                                
                                if personas:
                                    st.markdown("#### 👤 Perfil das Personas")
                                    
                                    # Métricas das personas
                                    col1, col2, col3, col4 = st.columns(4)
                                    total_personas = len(personas)
                                    total_usuarios = sum(profile['size'] for profile in personas.values())
                                    
                                    with col1:
                                        st.metric("Número de Personas", total_personas)
                                    with col2:
                                        st.metric("Total de Registros", total_usuarios)
                                    with col3:
                                        maior_persona = max(personas.values(), key=lambda x: x['size'])
                                        st.metric("Persona Mais Comum", f"{maior_persona['size']} reg.")
                                    with col4:
                                        menor_persona = min(personas.values(), key=lambda x: x['size'])
                                        st.metric("Persona Menos Comum", f"{menor_persona['size']} reg.")
                                    
                                    # Detalhes de cada persona
                                    for persona_name, profile in personas.items():
                                        with st.expander(f"{persona_name} - {profile['size']} registros ({(profile['size']/total_usuarios)*100:.1f}%)"):
                                            st.write("**Características Principais:**")
                                            
                                            # Criar cards para cada característica
                                            cols = st.columns(2)
                                            char_items = list(profile['characteristics'].items())
                                            
                                            for idx, (campo, valor) in enumerate(char_items):
                                                with cols[idx % 2]:
                                                    if isinstance(valor, (int, float)):
                                                        st.metric(f"{campo}", f"{valor:.2f}")
                                                    else:
                                                        st.metric(f"{campo}", valor)
                                    
                                    # Visualização das personas
                                    if len(campos_persona) >= 2:
                                        st.markdown("#### 📊 Visualização das Personas")
                                        try:
                                            if 'persona' not in df_completo.columns:
                                                personas_result, _, _ = perform_clustering(df_completo[campos_persona], min(5, len(df_completo)))
                                                df_completo_temp = df_completo.copy()
                                                df_completo_temp['persona'] = personas_result
                                            else:
                                                df_completo_temp = df_completo
                                            
                                            # Scatter plot
                                            fig = px.scatter(
                                                df_completo_temp, 
                                                x=campos_persona[0], 
                                                y=campos_persona[1],
                                                color='persona',
                                                title="Visualização de Personas - Distribuição",
                                                color_continuous_scale='purples',
                                                hover_data=campos_persona[:3]
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                                            
                                            # Gráfico de barras com distribuição
                                            persona_counts = df_completo_temp['persona'].value_counts().sort_index()
                                            fig_bar = px.bar(
                                                x=[f"Persona {i+1}" for i in persona_counts.index],
                                                y=persona_counts.values,
                                                title="Distribuição de Registros por Persona",
                                                labels={'x': 'Persona', 'y': 'Número de Registros'},
                                                color=persona_counts.values,
                                                color_continuous_scale='purples'
                                            )
                                            st.plotly_chart(fig_bar, use_container_width=True)
                                            
                                        except Exception as e:
                                            st.info(f"Visualização não disponível: {e}")
                                else:
                                    st.error(f"Erro na análise de personas: {error}")
                            else:
                                st.info("👆 Selecione pelo menos um campo para análise de personas")
                            
                            st.markdown('</div>', unsafe_allow_html=True)

                        with tab4:
                            st.markdown('<div class="main-container">', unsafe_allow_html=True)
                            st.markdown('<div class="section-header">🔍 Análises Avançadas</div>', unsafe_allow_html=True)
                            
                            if df_completo.empty:
                                st.warning("Não há dados disponíveis para análises avançadas")
                            else:
                                st.markdown("#### 📋 Estrutura dos Dados")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Colunas", len(df_completo.columns))
                                with col2:
                                    st.metric("Registros", len(df_completo))
                                with col3:
                                    st.metric("Dados Faltantes", df_completo.isnull().sum().sum())
                                
                                st.markdown("#### 📊 Análise de Distribuição")
                                
                                analysis_column = st.selectbox(
                                    "Selecione a coluna para análise de distribuição",
                                    options=df_completo.columns.tolist(),
                                    key="dist_analysis_col"
                                )
                                
                                if analysis_column:
                                    try:
                                        if analysis_column not in df_completo.columns:
                                            st.error(f"Coluna '{analysis_column}' não encontrada")
                                        elif df_completo[analysis_column].isnull().all():
                                            st.warning(f"Coluna '{analysis_column}' está vazia")
                                        else:
                                            non_null_data = df_completo[analysis_column].dropna()
                                            if len(non_null_data) == 0:
                                                st.warning("Não há dados não-nulos para análise")
                                            else:
                                                value_counts = non_null_data.value_counts()
                                                value_percent = non_null_data.value_counts(normalize=True) * 100
                                                
                                                # Métricas da coluna
                                                col1, col2, col3, col4 = st.columns(4)
                                                
                                                with col1:
                                                    st.metric("Valores Únicos", len(value_counts))
                                                with col2:
                                                    st.metric("Total Registros", len(non_null_data))
                                                with col3:
                                                    st.metric("Valores Nulos", df_completo[analysis_column].isnull().sum())
                                                with col4:
                                                    if not value_counts.empty:
                                                        st.metric("Moda", value_counts.index[0])
                                                
                                                st.markdown("#### 🏆 Top 10 Valores")
                                                if len(value_counts) > 0:
                                                    top_10 = value_counts.head(10)
                                                    
                                                    fig = px.bar(
                                                        x=top_10.index.astype(str), 
                                                        y=top_10.values,
                                                        labels={'x': analysis_column, 'y': 'Frequência'},
                                                        title=f"Top 10 Valores - {analysis_column}",
                                                        color=top_10.values,
                                                        color_continuous_scale='purples'
                                                    )
                                                    fig.update_layout(xaxis_tickangle=-45)
                                                    st.plotly_chart(fig, use_container_width=True)
                                                    
                                                    st.markdown("#### 📋 Tabela de Frequências")
                                                    dist_df = pd.DataFrame({
                                                        'Valor': value_counts.index,
                                                        'Frequência': value_counts.values,
                                                        'Percentual (%)': value_percent.values.round(2)
                                                    })
                                                    st.dataframe(dist_df.head(15), use_container_width=True)
                                                    
                                    except Exception as e:
                                        st.error(f"Erro na análise da coluna {analysis_column}: {str(e)}")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with tab5:
                            st.markdown('<div class="main-container">', unsafe_allow_html=True)
                            st.markdown('<div class="section-header">🤖 Análise de Clusterização</div>', unsafe_allow_html=True)
                            
                            st.markdown("""
                            A **clusterização** é uma técnica de aprendizado não supervisionado que agrupa dados similares. 
                            Esta análise ajuda a descobrir padrões naturais nos seus dados.
                            """)
                            
                            if df_completo.empty:
                                st.warning("Não há dados disponíveis para clusterização")
                            else:
                                st.markdown("### 🔧 Configuração da Clusterização")
                                
                                available_columns = df_completo.columns.tolist()
                                clustering_columns = st.multiselect(
                                    "Selecione as colunas para clusterização:",
                                    options=available_columns,
                                    default=available_columns[:min(3, len(available_columns))],
                                    help="Selecione colunas numéricas ou categóricas para formação dos clusters"
                                )
                                
                                if clustering_columns:
                                    clustering_method = st.selectbox(
                                        "Método de Clusterização:",
                                        options=['kmeans', 'dbscan', 'lda'],
                                        help="KMeans: clusters numéricos, DBSCAN: detecção automática de outliers, LDA: análise de tópicos em texto"
                                    )
                                    
                                    if clustering_method == 'kmeans':
                                        # Para KMeans, calcular número ótimo de clusters
                                        with st.spinner("Calculando número ótimo de clusters..."):
                                            clustering_data, error = prepare_data_for_clustering(df_completo, clustering_columns)
                                            if error:
                                                st.error(f"Erro ao preparar dados: {error}")
                                            else:
                                                wcss, silhouette_scores, optimal_clusters = find_optimal_clusters_kmeans(clustering_data['scaled_data'])
                                        
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
                                        
                                    elif clustering_method == 'dbscan':
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
                                            text_data, error = prepare_text_data_for_lda(df_completo, clustering_columns)
                                            if error:
                                                st.error(f"Erro no LDA: {error}")
                                            else:
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
                                        with st.spinner("Executando clusterização..."):
                                            if clustering_method == 'lda':
                                                clustering_result = perform_advanced_clustering(
                                                    None, n_clusters, method='lda', text_data=text_data
                                                )
                                            else:
                                                clustering_data, error = prepare_data_for_clustering(df_completo, clustering_columns)
                                                if error:
                                                    st.error(f"Erro ao preparar dados: {error}")
                                                else:
                                                    if clustering_method == 'dbscan':
                                                        # Para DBSCAN, usar parâmetros específicos
                                                        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                                                        clusters = dbscan.fit_predict(clustering_data['scaled_data'])
                                                        clustering_result = {
                                                            'clusters': clusters,
                                                            'model': dbscan,
                                                            'metrics': {'silhouette': -1, 'calinski_harabasz': -1, 'davies_bouldin': -1},
                                                            'method': 'dbscan',
                                                            'features_2d': PCA(n_components=2).fit_transform(clustering_data['scaled_data'])
                                                        }
                                                    else:
                                                        clustering_result = perform_advanced_clustering(
                                                            clustering_data['scaled_data'], n_clusters, method=clustering_method
                                                        )
                                        
                                        if clustering_result is None:
                                            st.error("Erro na clusterização")
                                        else:
                                            if clustering_method == 'kmeans':
                                                st.success(f"Clusterização concluída! Score de Silhueta: {clustering_result['metrics']['silhouette']:.3f}")
                                            elif clustering_method == 'dbscan':
                                                n_clusters_found = len(set(clustering_result['clusters'])) - (1 if -1 in clustering_result['clusters'] else 0)
                                                n_outliers = len([x for x in clustering_result['clusters'] if x == -1])
                                                st.success(f"Clusterização concluída! {n_clusters_found} clusters encontrados, {n_outliers} outliers")
                                            else:  # LDA
                                                st.success(f"Análise de tópicos concluída! {n_clusters} tópicos identificados")
                                            
                                            # Visualização dos clusters
                                            st.markdown("#### 📊 Visualização dos Clusters")
                                            
                                            if 'features_2d' in clustering_result:
                                                viz_df = pd.DataFrame({
                                                    'x': clustering_result['features_2d'][:, 0],
                                                    'y': clustering_result['features_2d'][:, 1],
                                                    'cluster': clustering_result['clusters'],
                                                    'level': df_completo.iloc[:len(clustering_result['clusters'])]['level'] if 'level' in df_completo.columns else 'INFO',
                                                    'message': df_completo.iloc[:len(clustering_result['clusters'])]['message'].str[:50] + '...' if 'message' in df_completo.columns else ''
                                                })
                                                
                                                if clustering_method == 'lda':
                                                    title = 'Visualização 2D dos Tópicos LDA (t-SNE)'
                                                elif clustering_method == 'dbscan':
                                                    title = 'Visualização 2D dos Clusters DBSCAN'
                                                else:
                                                    title = 'Visualização 2D dos Clusters KMeans'
                                                
                                                fig_clusters = px.scatter(
                                                    viz_df, x='x', y='y', color='cluster',
                                                    hover_data=['level', 'message'] if 'level' in viz_df.columns else None,
                                                    title=title,
                                                    color_continuous_scale='purples'
                                                )
                                                st.plotly_chart(fig_clusters, use_container_width=True)
                                            
                                            # Análise específica para LDA
                                            if clustering_method == 'lda' and hasattr(clustering_result['model'], 'components_'):
                                                st.markdown("#### 📝 Análise de Tópicos LDA")
                                                
                                                topics = analyze_lda_topics(clustering_result['model'], clustering_result['vectorizer'])
                                                
                                                for topic in topics:
                                                    with st.expander(f"Tópico {topic['topic_id']}"):
                                                        st.write(f"**Palavras-chave:** {topic['top_words']}")
                                                        
                                                        # Gráfico de barras
                                                        fig = px.bar(
                                                            x=topic['word_weights'],
                                                            y=topic['top_features'],
                                                            orientation='h',
                                                            title=f"Top Palavras - Tópico {topic['topic_id']}",
                                                            labels={'x': 'Importância', 'y': 'Palavras'},
                                                            color=topic['word_weights'],
                                                            color_continuous_scale='purples'
                                                        )
                                                        st.plotly_chart(fig, use_container_width=True)
                                                        
                                                        # Mostrar exemplos deste tópico
                                                        topic_examples = df_completo[clustering_result['clusters'] == topic['topic_id']].head(3)
                                                        if not topic_examples.empty:
                                                            st.write("**Exemplos deste tópico:**")
                                                            for _, example in topic_examples.iterrows():
                                                                preview = " | ".join([str(example[col])[:50] for col in clustering_columns[:2] if col in example])
                                                                st.write(f"- {preview}...")
                                            
                                            # Análise de clusters tradicional (para todos os métodos)
                                            st.markdown("#### 🔍 Análise Detalhada dos Clusters")
                                            df_resultado = df_completo.copy()
                                            df_resultado['cluster'] = clustering_result['clusters']
                                            
                                            cluster_profiles = create_cluster_analysis(
                                                df_completo, clustering_result, clustering_columns
                                            )
                                            st.dataframe(cluster_profiles, use_container_width=True)
                                            
                                            # Exemplos por cluster
                                            st.markdown("#### 🔍 Exemplos por Cluster")
                                            for cluster_id in sorted(df_resultado['cluster'].unique()):
                                                cluster_size = len(df_resultado[df_resultado['cluster'] == cluster_id])
                                                if clustering_method == 'dbscan' and cluster_id == -1:
                                                    with st.expander(f"Outliers - {cluster_size} amostras"):
                                                        cluster_samples = df_resultado[df_resultado['cluster'] == cluster_id].head(3)
                                                        for _, sample in cluster_samples.iterrows():
                                                            st.write(f"**{sample.name}:** {sample.to_dict()}")
                                                else:
                                                    with st.expander(f"Cluster {cluster_id} - {cluster_size} amostras"):
                                                        cluster_samples = df_resultado[df_resultado['cluster'] == cluster_id].head(3)
                                                        for _, sample in cluster_samples.iterrows():
                                                            st.write(f"**{sample.name}:** {sample.to_dict()}")
                                            
                                            st.markdown("#### 💾 Exportar Resultados")
                                            csv = df_resultado.to_csv(index=False)
                                            st.download_button(
                                                label="📥 Download dos dados com clusters",
                                                data=csv,
                                                file_name="dados_clusterizados.csv",
                                                mime="text/csv",
                                                use_container_width=True
                                            )
                            
                                else:
                                    st.info("👆 Selecione pelo menos uma coluna para iniciar a análise de clusterização")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    else:
                        st.error("❌ Não foi possível processar os dados JSON ou DataFrame vazio.")
            
            else:
                st.warning("❌ Nenhum campo JSON encontrado na coluna selecionada.")
        
        else:
            st.error("❌ Não foi possível ler o arquivo CSV. Tente outro arquivo.")
    
    else:
        # Tela inicial - Como Usar
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
                    <p>Exporte eventos do Alexandria em CSV</p>
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
    if 'analysis_started' not in st.session_state:
        st.session_state.analysis_started = False
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    main()