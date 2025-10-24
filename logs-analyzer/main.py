import streamlit as st
import pandas as pd
import json
import ast
from collections import Counter, defaultdict
import csv
import io
import re
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .auto-field {
        background-color: #e8f5e8;
        padding: 0.5rem;
        margin: 0.2rem;
        border-radius: 5px;
        border-left: 4px solid #4caf50;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
        text-align: center;
    }
    .stat-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def parse_json_string(json_str):
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
    content = uploaded_file.read().decode('utf-8', errors='ignore')
    lines = content.split('\n')
    
    first_line = lines[0] if lines else ''
    delimiter = ','
    if ';' in first_line and first_line.count(';') > first_line.count(','):
        delimiter = ';'
    elif '\t' in first_line:
        delimiter = '\t'
    
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, delimiter=delimiter, engine='python', 
                        quoting=csv.QUOTE_ALL, on_bad_lines='warn')
        return df, "pandas"
    except Exception as e:
        st.warning(f"Pandas method failed: {e}")
    
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
                    row = line.split(delimiter)
                    data.append(row)
        
        max_cols = max(len(row) for row in data) if data else 0
        
        for row in data:
            while len(row) < max_cols:
                row.append(None)
            while len(row) > max_cols:
                row.pop()
        
        if data:
            headers = data[0] if len(data[0]) == max_cols else [f'col_{i}' for i in range(max_cols)]
            df = pd.DataFrame(data[1:], columns=headers)
            return df, "manual"
    except Exception as e:
        st.warning(f"Manual method failed: {e}")
    
    return None, "error"

def extract_all_fields(data, parent_key='', separator='.'):
    fields = []
    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{parent_key}{separator}{key}" if parent_key else key
            new_key = re.sub(r'^data\.data\.', 'data.', new_key)
            new_key = re.sub(r'^data\.', '', new_key)
            
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
            parent_key = re.sub(r'^data\.data\.', 'data.', parent_key)
            parent_key = re.sub(r'^data\.', '', parent_key)
            fields.append(parent_key)
    
    return list(set(fields))

def extract_field_value(data, field_path):
    try:
        if not field_path.startswith('data.') and 'data.' in str(data):
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
                            campo_clean = re.sub(r'^data\.', '', campo)
                            linha_dados[campo_clean] = valor
                            associacoes[campo_clean][str(valor)] += 1
                    
                    dados_completos.append(linha_dados)
        
        df_completo = pd.DataFrame(dados_completos)
        return associacoes, df_completo, linhas_processadas, linhas_com_json
        
    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
        return None, None, 0, 0

def analyze_statistics(df, campos_selecionados):
    analyses = {}
    
    analyses['descriptive'] = df.describe(include='all')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        analyses['correlation'] = df[numeric_cols].corr()
    
    analyses['distribution'] = {}
    for col in df.columns:
        if df[col].dtype in ['object', 'category']:
            analyses['distribution'][col] = df[col].value_counts()
        else:
            analyses['distribution'][col] = df[col].describe()
    
    return analyses

def perform_pca_analysis(df, n_components=2):
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 2:
        return None, None, "Mínimo 2 colunas numéricas necessárias para PCA"
    
    numeric_df = numeric_df.fillna(numeric_df.mean())
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    
    pca = PCA(n_components=min(n_components, len(numeric_df.columns)))
    pca_result = pca.fit_transform(scaled_data)
    
    return pca_result, pca, None

def perform_clustering(df, n_clusters=3):
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 2:
        return None, None, "Mínimo 2 colunas numéricas necessárias para clustering"
    
    numeric_df = numeric_df.fillna(numeric_df.mean())
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    
    return clusters, kmeans, None

def create_persona_analysis(df, clustering_fields):
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

def prepare_data_for_clustering(df, selected_columns):
    if not selected_columns:
        return None, "Nenhuma coluna selecionada"
    
    clustering_df = df[selected_columns].copy()
    
    numeric_cols = clustering_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = clustering_df.select_dtypes(include=['object']).columns.tolist()
    
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
    
    clustering_df = clustering_df.dropna(thresh=len(clustering_df) * 0.5, axis=1)
    
    if clustering_df.empty:
        return None, "Dados insuficientes após limpeza"
    
    for col in clustering_df.columns:
        if clustering_df[col].dtype in [np.float64, np.int64]:
            clustering_df[col] = clustering_df[col].fillna(clustering_df[col].median())
        else:
            clustering_df[col] = clustering_df[col].fillna(clustering_df[col].mode()[0] if not clustering_df[col].mode().empty else 0)
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_df)
    
    return {
        'scaled_data': scaled_data,
        'original_df': clustering_df,
        'feature_names': clustering_df.columns.tolist(),
        'scaler': scaler
    }, None

def find_optimal_clusters(data, max_clusters=10):
    wcss = []
    silhouette_scores = []
    
    for i in range(2, min(max_clusters + 1, len(data) - 1)):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
        
        if i < len(data):
            silhouette_scores.append(silhouette_score(data, kmeans.labels_))
        else:
            silhouette_scores.append(0)
    
    return wcss, silhouette_scores

def perform_advanced_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(data)
    
    silhouette_avg = silhouette_score(data, clusters)
    calinski_score = calinski_harabasz_score(data, clusters)
    davies_score = davies_bouldin_score(data, clusters)
    
    return {
        'clusters': clusters,
        'model': kmeans,
        'metrics': {
            'silhouette': silhouette_avg,
            'calinski_harabasz': calinski_score,
            'davies_bouldin': davies_score
        }
    }

def visualize_clusters_2d(data, clusters, feature_names):
    pca = PCA(n_components=2, random_state=42)
    reduced_data = pca.fit_transform(data)
    
    plot_df = pd.DataFrame({
        'PC1': reduced_data[:, 0],
        'PC2': reduced_data[:, 1],
        'cluster': clusters
    })
    
    fig = px.scatter(
        plot_df, x='PC1', y='PC2', color='cluster',
        title='Visualização de Clusters - PCA',
        labels={'PC1': f'Componente Principal 1 ({pca.explained_variance_ratio_[0]:.1%})',
               'PC2': f'Componente Principal 2 ({pca.explained_variance_ratio_[1]:.1%})'},
        color_continuous_scale='viridis'
    )
    
    return fig

def create_cluster_analysis(df, clustering_result, feature_names):
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
    st.markdown('<h1 class="main-header">Analisador LOGS (em CSV)</h1>', unsafe_allow_html=True)
    
    if "analysis_started" not in st.session_state:
        st.session_state.analysis_started = False
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None

    uploaded_file = st.file_uploader(
        "Faça upload do seu arquivo CSV",
        type=['csv'],
        help="Arquivo CSV que pode conter problemas de formatação"
    )
    
    if uploaded_file is not None:
        with st.spinner('Lendo arquivo CSV (pode levar alguns segundos)...'):
            df, method = read_problematic_csv(uploaded_file)
        
        if df is not None:
            st.success(f"✅ CSV lido com sucesso usando método: {method}")
            st.write(f"**Formato:** {len(df)} linhas × {len(df.columns)} colunas")
            
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
                st.success(f"Encontrados {len(campos_disponiveis)} campos nos JSONs!")
                st.subheader("Selecione os Campos para Análise")
                
                filtro_campo = st.text_input("🔍 Filtrar campos", placeholder="Digite para filtrar...")
                campos_filtrados = [campo for campo in campos_disponiveis if filtro_campo.lower() in campo.lower()]
                
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
                
                if campos_selecionados and st.button("Iniciar Análise Completa", type="primary"):
                    with st.spinner('Processando dados e gerando análises...'):
                        st.session_state.analysis_result = processar_csv_automatico(
                            df, campos_selecionados, coluna_json
                        )
                        st.session_state.analysis_started = True

                if st.session_state.analysis_started and st.session_state.analysis_result is not None:
                    associacoes, df_completo, total_linhas, linhas_json = st.session_state.analysis_result
                    
                    if associacoes and df_completo is not None and not df_completo.empty:
                        
                        tab_options = ["Visão Geral", "Estatísticas", "Personas", "Análises Avançadas", "Clusterização"]
                        
                        active_tab = st.radio(
                            "Navegar entre as análises:",
                            options=tab_options,
                            horizontal=True,
                            key='active_tab'
                        )
                        st.markdown("---")

                        if active_tab == "Visão Geral":
                            st.subheader("Visão Geral dos Dados")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("📄 Total de Linhas", total_linhas)
                            with col2:
                                st.metric("✅ Linhas com JSON", linhas_json)
                            with col3:
                                st.metric("🎯 Campos Analisados", len(campos_selecionados))
                            with col4:
                                st.metric("🔤 Valores Únicos", df_completo.nunique().sum())
                            
                            st.dataframe(df_completo.head(10), use_container_width=True)
                        
                        elif active_tab == "Estatísticas":
                            st.subheader("Análises Estatísticas")
                            
                            st.markdown("#### 📊 Estatísticas Descritivas")
                            st.dataframe(df_completo.describe(include='all'), use_container_width=True)
                            
                            numeric_cols = df_completo.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 1:
                                st.markdown("#### Matriz de Correlação")
                                corr_matrix = df_completo[numeric_cols].corr()
                                fig, ax = plt.subplots(figsize=(10, 8))
                                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                                st.pyplot(fig)
                            
                            st.markdown("#### Distribuições")
                            for col in df_completo.columns:
                                with st.expander(f"Distribuição de {col}"):
                                    if df_completo[col].dtype == 'object':
                                        fig = px.histogram(df_completo, x=col, title=f"Distribuição de {col}")
                                        st.plotly_chart(fig)
                                    else:
                                        fig = px.histogram(df_completo, x=col, title=f"Distribuição de {col}")
                                        st.plotly_chart(fig)
                        
                        elif active_tab == "Personas":
                            st.subheader("Análise de Personas")
                            
                            campos_persona = st.multiselect(
                                "Selecione campos para análise de personas",
                                options=df_completo.columns.tolist(),
                                default=df_completo.columns.tolist()[:min(5, len(df_completo.columns))]
                            )
                            
                            if campos_persona:
                                personas, error = create_persona_analysis(df_completo, campos_persona)
                                
                                if personas:
                                    st.markdown("#### Perfil das Personas")
                                    
                                    for persona_name, profile in personas.items():
                                        with st.expander(f"{persona_name} (n={profile['size']})"):
                                            st.write("**Características:**")
                                            for campo, valor in profile['characteristics'].items():
                                                st.write(f"- **{campo}**: {valor}")
                                    
                                    if len(campos_persona) >= 2:
                                        try:
                                            if 'persona' not in df_completo.columns:
                                                personas_result, _, _ = perform_clustering(df_completo[campos_persona], 3)
                                                df_completo['persona'] = personas_result
                                            
                                            fig = px.scatter(
                                                df_completo, 
                                                x=campos_persona[0], 
                                                y=campos_persona[1],
                                                color='persona',
                                                title="Visualização de Personas"
                                            )
                                            st.plotly_chart(fig)
                                        except Exception as e:
                                            st.info(f"Visualização não disponível: {e}")
                        
                        elif active_tab == "Análises Avançadas":
                            st.subheader("Análises Avançadas")
                            
                            if df_completo.empty:
                                st.warning("Não há dados disponíveis para análises avançadas")
                            else:
                                st.write("**Estrutura dos dados:**")
                                st.write(f"- Colunas disponíveis: {', '.join(df_completo.columns.tolist())}")
                                st.write(f"- Total de registros: {len(df_completo)}")
                                st.write(f"- Tipos de dados: {df_completo.dtypes.to_dict()}")
                                
                                st.subheader("Análise de Distribuição")
                                
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
                                                
                                                st.subheader("Top 10 Valores")
                                                if len(value_counts) > 0:
                                                    top_10 = value_counts.head(10)
                                                    
                                                    fig = px.bar(
                                                        x=top_10.index.astype(str), 
                                                        y=top_10.values,
                                                        labels={'x': analysis_column, 'y': 'Frequência'},
                                                        title=f"Top 10 Valores - {analysis_column}",
                                                        color=top_10.values,
                                                        color_continuous_scale='viridis'
                                                    )
                                                    fig.update_layout(xaxis_tickangle=-45)
                                                    st.plotly_chart(fig, use_container_width=True)
                                                    
                                                    st.subheader("📋 Tabela de Frequências")
                                                    dist_df = pd.DataFrame({
                                                        'Valor': value_counts.index,
                                                        'Frequência': value_counts.values,
                                                        'Percentual (%)': value_percent.values.round(2)
                                                    })
                                                    st.dataframe(dist_df.head(15), use_container_width=True)
                                                    
                                                    if len(df_completo.columns) > 1:
                                                        st.subheader("🔗 Análise de Associação")
                                                        
                                                        other_columns = [col for col in df_completo.columns if col != analysis_column]
                                                        col2 = st.selectbox(
                                                            "Selecione outra coluna para análise de associação",
                                                            options=other_columns,
                                                            key="assoc_col"
                                                        )
                                                        
                                                        if col2:
                                                            try:
                                                                assoc_data = df_completo[[analysis_column, col2]].dropna()
                                                                
                                                                if len(assoc_data) < 2:
                                                                    st.warning("Dados insuficientes para análise de associação")
                                                                else:
                                                                    contingency = pd.crosstab(assoc_data[analysis_column], assoc_data[col2])
                                                                    
                                                                    st.write("**Tabela de Contingência:**")
                                                                    st.dataframe(contingency, use_container_width=True)
                                                                    
                                                                    if not contingency.empty:
                                                                        fig_heatmap = px.imshow(
                                                                            contingency,
                                                                            title=f"Associação entre {analysis_column} e {col2}",
                                                                            labels=dict(x=col2, y=analysis_column, color="Frequência"),
                                                                            aspect="auto",
                                                                            color_continuous_scale='blues'
                                                                        )
                                                                        st.plotly_chart(fig_heatmap, use_container_width=True)
                                                                        
                                                                        try:
                                                                            chi2, p_value, dof, expected = chi2_contingency(contingency)
                                                                            n = contingency.sum().sum()
                                                                            cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1))) if n > 0 else 0
                                                                            
                                                                            st.write("**Estatísticas de Associação:**")
                                                                            st.write(f"- Chi-quadrado: {chi2:.3f}")
                                                                            st.write(f"- P-value: {p_value:.6f}")
                                                                            st.write(f"- Graus de liberdade: {dof}")
                                                                            st.write(f"- Cramér's V: {cramers_v:.3f}")
                                                                            
                                                                            if p_value < 0.05:
                                                                                st.success("✅ **Associação estatisticamente significativa** (p < 0.05)")
                                                                                st.info(f"Forte evidência de relação entre {analysis_column} e {col2}")
                                                                            else:
                                                                                st.warning("**Não há evidência de associação significativa**")
                                                                                
                                                                        except Exception as e:
                                                                            st.warning(f"Não foi possível calcular estatísticas de associação: {str(e)}")
                                                            except Exception as e:
                                                                st.error(f"Erro na análise de associação: {str(e)}")
                                    except Exception as e:
                                        st.error(f"Erro na análise da coluna {analysis_column}: {str(e)}")

                        elif active_tab == "📊 Clusterização":
                            st.subheader("📊 Análise de Clusterização")
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
                                    with st.spinner("Preparando dados para clusterização..."):
                                        clustering_data, error = prepare_data_for_clustering(df_completo, clustering_columns)
                                    
                                    if error:
                                        st.error(f"Erro ao preparar dados: {error}")
                                    else:
                                        st.success(f"✅ Dados preparados! {clustering_data['scaled_data'].shape[0]} amostras, {clustering_data['scaled_data'].shape[1]} características")
                                        
                                        st.markdown("### 🔍 Determinação do Número Ótimo de Clusters")
                                        
                                        with st.spinner("Calculando número ótimo de clusters..."):
                                            wcss, silhouette_scores = find_optimal_clusters(
                                                clustering_data['scaled_data'], 
                                                max_clusters=min(8, len(clustering_data['scaled_data']) - 1)
                                            )
                                        
                                        fig_elbow = go.Figure()
                                        fig_elbow.add_trace(go.Scatter(
                                            x=list(range(2, len(wcss) + 2)),
                                            y=wcss,
                                            mode='lines+markers',
                                            name='WCSS',
                                            line=dict(color='blue')
                                        ))
                                        fig_elbow.update_layout(
                                            title='Método do Cotovelo - WCSS vs Número de Clusters',
                                            xaxis_title='Número de Clusters',
                                            yaxis_title='Within-Cluster Sum of Squares (WCSS)'
                                        )
                                        st.plotly_chart(fig_elbow)
                                        
                                        fig_silhouette = go.Figure()
                                        fig_silhouette.add_trace(go.Scatter(
                                            x=list(range(2, len(silhouette_scores) + 2)),
                                            y=silhouette_scores,
                                            mode='lines+markers',
                                            name='Silhouette Score',
                                            line=dict(color='green')
                                        ))
                                        fig_silhouette.update_layout(
                                            title='Silhouette Score vs Número de Clusters',
                                            xaxis_title='Número de Clusters',
                                            yaxis_title='Silhouette Score'
                                        )
                                        st.plotly_chart(fig_silhouette)
                                        
                                        st.markdown("### ⚙️ Executar Clusterização")
                                        
                                        n_clusters = st.slider(
                                            "Selecione o número de clusters:",
                                            min_value=2,
                                            max_value=min(8, len(clustering_data['scaled_data']) - 1),
                                            value=3,
                                            help="Baseie-se nos gráficos acima para escolher o número ideal"
                                        )
                                        
                                        if st.button("🚀 Executar Clusterização K-Means"):
                                            with st.spinner("Executando clusterização K-Means..."):
                                                clustering_result = perform_advanced_clustering(
                                                    clustering_data['scaled_data'], n_clusters
                                                )
                                            
                                            st.markdown("### 📈 Resultados da Clusterização")
                                            
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric("Silhouette Score", f"{clustering_result['metrics']['silhouette']:.3f}")
                                            with col2:
                                                st.metric("Calinski-Harabasz", f"{clustering_result['metrics']['calinski_harabasz']:.1f}")
                                            with col3:
                                                st.metric("Davies-Bouldin", f"{clustering_result['metrics']['davies_bouldin']:.3f}")
                                            
                                            st.markdown("#### 🌐 Visualização dos Clusters")
                                            fig_clusters = visualize_clusters_2d(
                                                clustering_data['scaled_data'],
                                                clustering_result['clusters'],
                                                clustering_data['feature_names']
                                            )
                                            st.plotly_chart(fig_clusters, use_container_width=True)
                                            
                                            st.markdown("#### 🔍 Análise Detalhada dos Clusters")
                                            cluster_profiles = create_cluster_analysis(
                                                df_completo, clustering_result, clustering_data['feature_names']
                                            )
                                            
                                            for cluster_name, profile in cluster_profiles.items():
                                                with st.expander(f"{cluster_name} - {profile['size']} registros ({profile['percentage']:.1f}%)"):
                                                    st.write(f"**Tamanho:** {profile['size']} registros")
                                                    st.write(f"**Percentual:** {profile['percentage']:.1f}%")
                                                    st.write("**Características principais:**")
                                                    
                                                    for feature, stats in profile['characteristics'].items():
                                                        if stats['type'] == 'numeric':
                                                            st.write(f"- **{feature}:** Média = {stats['mean']:.2f}, Mediana = {stats['median']:.2f}")
                                                        else:
                                                            st.write(f"- **{feature}:** Valor mais comum = '{stats['top_value']}' ({stats['top_percentage']:.1f}%)")
                                            
                                            st.markdown("#### 💾 Exportar Resultados")
                                            df_resultado = df_completo.copy()
                                            df_resultado['cluster'] = clustering_result['clusters']
                                            
                                            csv = df_resultado.to_csv(index=False)
                                            st.download_button(
                                                label="📥 Download dos dados com clusters",
                                                data=csv,
                                                file_name="dados_clusterizados.csv",
                                                mime="text/csv"
                                            )
                            
                                else:
                                    st.info("Selecione pelo menos uma coluna para iniciar a análise de clusterização")
                    
                    else:
                        st.error("Não foi possível processar os dados JSON ou DataFrame vazio.")
            
            else:
                st.warning("Nenhum campo JSON encontrado na coluna selecionada.")
        
        else:
            st.error("Não foi possível ler o arquivo CSV. Tente outro arquivo.")
    
    else:
        st.info("👆 Faça upload de um arquivo CSV para começar a análise")

if __name__ == "__main__":
    main()