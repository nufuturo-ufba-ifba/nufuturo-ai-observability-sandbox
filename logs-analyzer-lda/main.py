import streamlit as st
import json
import pandas as pd
import re
from io import StringIO
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from typing import List, Dict, Any, Tuple, Optional

# Chaves que procuramos para extrair texto relevante do log
TARGET_TEXT_KEYS = {'message', 'msg', 'error', 'description', 'value', 'title', 'detail', 'type'}

def extract_text_from_log(log_obj: Dict[str, Any]) -> str:
    """
    Varre recursivamente um objeto JSON e extrai valores de string
    associados a chaves de interesse (TARGET_TEXT_KEYS).
    """
    texts = []
    
    def _recursive_extract(obj):
        if isinstance(obj, dict):
            for key, val in obj.items():
                if key.lower() in TARGET_TEXT_KEYS and isinstance(val, str) and val:
                    texts.append(val)
                elif isinstance(val, (dict, list)):
                    _recursive_extract(val)
        elif isinstance(obj, list):
            for item in obj:
                _recursive_extract(item)
                
    _recursive_extract(log_obj)
    return " ".join(texts)

def clean_text(text: str) -> str:
    """
    Limpa o texto para análise de LDA:
    - Converte para minúsculas
    - Remove números e pontuação
    - Remove palavras muito curtas (ruído)
    """
    text = text.lower()
    text = re.sub(r'[\W\d_]+', ' ', text) # Remove pontuação, números e underscores
    text = re.sub(r'\s+', ' ', text).strip() # Remove espaços extras
    words = [word for word in text.split() if len(word) > 2]
    return " ".join(words)

@st.cache_data(show_spinner="Processando todos os logs... Isso pode levar um tempo.")
def run_lda_analysis(logs: List[Dict[str, Any]], num_topics: int) -> Tuple[Optional[pd.DataFrame], Optional[LatentDirichletAllocation], Optional[CountVectorizer], Optional[pd.Series]]:
    """
    Executa o pipeline completo de LDA.
    Retorna (DataFrame, Modelo, Vetorizador, Contagem de Tópicos) ou (DataFrame, None, None, None) em modo de log único.
    """
    
    extracted_texts = [extract_text_from_log(log) for log in logs]
    cleaned_texts = [clean_text(text) for text in extracted_texts]
    source_files = [log.get('_source_file', 'unknown') for log in logs]
    
    df = pd.DataFrame({
        'raw_log': logs,
        'extracted_text': extracted_texts,
        'cleaned_text': cleaned_texts,
        '_source_file': source_files
    })
    
    df_valid = df[df['cleaned_text'] != ""].copy()
    n_docs = len(df_valid)
    
    if n_docs == 0:
        st.error("Nenhum texto relevante (baseado nas chaves-alvo) foi encontrado nos logs.")
        return None, None, None, None

    # --- CORREÇÃO PRINCIPAL: MODO DE LOG ÚNICO ---
    # Se houver apenas 1 log, não rode o LDA.
    # Apenas retorne o DataFrame com os dados extraídos.
    if n_docs == 1:
        st.info("Você carregou apenas 1 log válido. Exibindo análise individual.")
        # Retorna o DataFrame, mas None para os modelos LDA
        return df_valid, None, None, None
    
    # --- MODO LDA (Multi-Log) ---
    
    # Ajusta os parâmetros do Vetorizador para N pequenos (2 a 4 logs)
    if 2 <= n_docs < 5:
        current_min_df = 1
        current_max_df = 1.0
        st.warning(f"Com apenas {n_docs} logs, a análise pode ser menos precisa. Usando min_df=1, max_df=1.0 para evitar erros.")
    else:
        # Padrões para 5+ logs
        current_min_df = 2
        current_max_df = 0.9
    
    vectorizer = CountVectorizer(max_df=current_max_df, min_df=current_min_df, lowercase=True)
    
    try:
        tf_matrix = vectorizer.fit_transform(df_valid['cleaned_text'])
    except ValueError as e:
        if 'max_df corresponds to < documents than min_df' in str(e):
             st.error(f"Erro ao vetorizar: {e}. Isso pode acontecer se os logs forem muito pequenos ou idênticos.")
             return None, None, None, None
        else:
            raise e

    if tf_matrix.shape[1] == 0:
        st.error(f"Não foi possível construir um vocabulário. Isso pode acontecer se todos os logs forem idênticos ou não houver palavras em comum (após a limpeza e filtros min/max_df).")
        st.info(f"Tente carregar logs mais variados. (Parâmetros usados: min_df={current_min_df}, max_df={current_max_df})")
        return None, None, None, None

    # Ajusta o número de tópicos se for maior que o número de logs
    if num_topics >= n_docs:
        new_num_topics = n_docs - 1
        st.warning(f"O número de tópicos ({num_topics}) é muito alto para {n_docs} logs. Reduzindo o número de tópicos para {new_num_topics}.")
        num_topics = new_num_topics

    if num_topics < 1:
        st.error("Não é possível rodar a análise com menos de 1 tópico. (Nº de logs é muito baixo)")
        return None, None, None, None
        
    # 4. Treinar Modelo LDA
    lda = LatentDirichletAllocation(
        n_components=num_topics, 
        max_iter=10, 
        learning_method='online', 
        learning_offset=50., 
        random_state=42
    )
    lda.fit(tf_matrix)
    
    # 5. Atribuir tópicos a cada log
    topic_distribution = lda.transform(tf_matrix)
    df_valid['topic'] = topic_distribution.argmax(axis=1)
    df_valid['topic_probability'] = topic_distribution.max(axis=1)
    
    topic_counts = df_valid['topic'].value_counts().sort_index()
    
    return df_valid.sort_values('topic_probability', ascending=False), lda, vectorizer, topic_counts

def get_topic_words(topic_idx: int, lda_model: LatentDirichletAllocation, vectorizer: CountVectorizer, n_words: int) -> str:
    """Retorna as N palavras mais importantes para um determinado tópico."""
    try:
        feature_names = vectorizer.get_feature_names_out()
        topic_components = lda_model.components_[topic_idx]
        top_word_indices = topic_components.argsort()[:-n_words - 1:-1]
        top_words = [feature_names[i] for i in top_word_indices]
        return ", ".join(top_words)
    except Exception:
        return "N/A"

# --- Configuração da Página Streamlit ---

st.set_page_config(layout="wide", page_title="Analisador de Logs com LDA")

st.title("🔎 Analisador de Logs JSON com LDA")
st.write("Faça o upload de um ou mais arquivos JSON contendo logs. O sistema usará Latent Dirichlet Allocation (LDA) para encontrar e agrupar automaticamente os 'tópicos' (tipos de erro) mais comuns.")

# --- Sidebar de Configurações ---

st.sidebar.header("Configurações do Modelo LDA")
num_topics = st.sidebar.slider(
    "Número de Tópicos (Grupos)", 
    min_value=2, 
    max_value=20, 
    value=5, 
    step=1,
    help="Quantos grupos de logs você acha que existem? (Ex: 'Erros de Nulidade', 'Erros de Rede', 'Logs de Sucesso')"
)
num_top_words = st.sidebar.slider(
    "Palavras por Tópico", 
    min_value=3, 
    max_value=15, 
    value=7, 
    step=1,
    help="Quantas palavras-chave mostrar para descrever cada tópico."
)

# --- Upload de Arquivos ---

uploaded_files = st.file_uploader(
    "Carregue seus arquivos .json", 
    type="json", 
    accept_multiple_files=True
)

if uploaded_files:
    all_logs = []
    log_sources = []
    
    for file in uploaded_files:
        try:
            string_data = StringIO(file.getvalue().decode("utf-8")).read()
            data = json.loads(string_data)
            
            if isinstance(data, list):
                all_logs.extend(data)
                log_sources.extend([file.name] * len(data))
            elif isinstance(data, dict):
                all_logs.append(data)
                log_sources.append(file.name)
                
        except json.JSONDecodeError:
            st.error(f"Erro Crítico: O arquivo '{file.name}' não é um JSON válido e será ignorado.")
        except Exception as e:
            st.error(f"Erro ao ler '{file.name}': {e}")

    if not all_logs:
        st.warning("Nenhum log válido foi carregado.")
    else:
        st.success(f"Carregados {len(all_logs)} logs de {len(uploaded_files)} arquivos.")
        
        for i, log in enumerate(all_logs):
            if isinstance(log, dict):
                log['_source_file'] = log_sources[i]

        # --- Executa a Análise ---
        df_results, lda_model, vectorizer, topic_counts = run_lda_analysis(all_logs, num_topics)
        
        # --- ROTA DE EXIBIÇÃO ---

        # Caso 1: SUCESSO - MODO LDA (Multi-Log)
        if lda_model and vectorizer and (df_results is not None) and (topic_counts is not None):
            st.header("Resultados da Análise de Tópicos (LDA)")
            st.write("Logs foram agrupados nos seguintes tópicos.")
            
            valid_topic_indices = topic_counts[topic_counts > 0].index
            
            if len(valid_topic_indices) == 0:
                st.warning("A análise foi executada, mas nenhum tópico conclusivo foi formado.")
            
            cols = st.columns(2)
            for i in valid_topic_indices:
                col_index = valid_topic_indices.get_loc(i) % 2
                topic_words = get_topic_words(i, lda_model, vectorizer, num_top_words)
                count = topic_counts[i]
                cols[col_index].info(f"**Tópico {i}** ({count} logs)\n\n*Palavras-chave: {topic_words}*")

            st.markdown("---")
            st.header("Explore os Logs por Tópico")

            if len(valid_topic_indices) > 0:
                tab_titles = [f"Tópico {i} ({topic_counts[i]} logs)" for i in valid_topic_indices]
                topic_tabs = st.tabs(tab_titles)
                
                for i, tab in zip(valid_topic_indices, topic_tabs):
                    with tab:
                        topic_df = df_results[df_results['topic'] == i]
                        st.write(f"**Principais Palavras:** {get_topic_words(i, lda_model, vectorizer, num_top_words)}")
                        
                        st.dataframe(topic_df[['_source_file', 'extracted_text', 'topic_probability']], height=300)
                        
                        st.subheader("Ver Log Bruto (JSON)")
                        most_relevant_log = topic_df.iloc[0]['raw_log']
                        with st.expander("Clique para ver o JSON completo do log mais relevante deste tópico"):
                            st.json(most_relevant_log)
            else:
                st.write("Nenhum log para exibir nas abas.")

        # Caso 2: SUCESSO - MODO ANÁLISE INDIVIDUAL (1 Log)
        elif (df_results is not None) and (lda_model is None):
            st.header("Análise de Log Único")
            
            # Pega a única linha do dataframe
            log_row = df_results.iloc[0]
            
            st.subheader(f"Arquivo de Origem: `{log_row['_source_file']}`")
            
            st.subheader("Texto Extraído do Log")
            if log_row['extracted_text']:
                st.write(log_row['extracted_text'])
            else:
                st.warning("Nenhum texto relevante (com base nas chaves-alvo) foi encontrado neste log.")
                
            st.subheader("JSON Bruto")
            with st.expander("Clique para ver o JSON completo"):
                st.json(log_row['raw_log'])

    