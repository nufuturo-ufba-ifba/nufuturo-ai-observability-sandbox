import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import plotly.graph_objects as go
from prometheus_pandas import query
from dotenv import load_dotenv
import os

# Carregar variáveis de ambiente
load_dotenv()
BASE_URL = os.getenv("BASE_URL")

# Método para atualizar variáveis de estado ao submeter o formulário
def submitted_form():
    st.session_state.prom_query = st.session_state.get('query_input', '')
    st.session_state.begin = st.session_state.get('range_begin', (datetime.now() - timedelta(days=3)).isoformat())
    st.session_state.end = st.session_state.get('range_end', datetime.now().isoformat())
    st.session_state.delta = st.session_state.get('delta_input', 3)
    st.session_state.form_submitted = True

# Configurar título da página
st.title("Home")

# Inicializar variáveis de estado
if "prom_connection" not in st.session_state:
    st.session_state.prom_connection = query.Prometheus(BASE_URL)

if "prom_query" not in st.session_state:
    st.session_state.prom_query = ""

if "begin" not in st.session_state:
    st.session_state.begin = (datetime.now() - timedelta(days=3)).isoformat()

if "end" not in st.session_state:
    st.session_state.end = datetime.now().isoformat()

if "interval" not in st.session_state:
    st.session_state.interval = "5m"

if "delta" not in st.session_state:
    st.session_state.delta = 3

if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False

# --- FORMULÁRIO CORRIGIDO ---
with st.form("prometheus_query_params"):
    query_input = st.text_input(
        'Prometheus Query',
        value=st.session_state.prom_query,
        placeholder='Input prometheus query',
        key="query_input"
    )

    range_end = st.text_input(
        'End of query range',
        value=datetime.now().isoformat(),
        disabled=True,
        key="range_end"
    )

    delta_input = st.slider(
        label="Length of query range (in days)",
        min_value=1,
        max_value=7,
        value=st.session_state.delta,
        key="delta_input"
    )

    # Calcular range_begin com base no delta_input
    range_begin = (datetime.now() - timedelta(days=delta_input)).isoformat()
    st.session_state.range_begin = range_begin

    # O BOTÃO PRECISA ESTAR AQUI DENTRO DO "WITH"
    submitted = st.form_submit_button("Submit")
    if submitted:
        submitted_form()

# --- PROCESSAMENTO (FORA DO FORMULÁRIO) ---
if st.session_state.form_submitted:
    if all([st.session_state.prom_query.strip(), st.session_state.begin, st.session_state.end, st.session_state.interval]):
        with st.spinner('🔄 Carregando dados do Prometheus...'):
            try:
                inicio_iso = pd.to_datetime(st.session_state.begin).strftime('%Y-%m-%dT%H:%M:%SZ')
                fim_iso = pd.to_datetime(st.session_state.end).strftime('%Y-%m-%dT%H:%M:%SZ')

                prom_data = st.session_state.prom_connection.query_range(
                    st.session_state.prom_query, inicio_iso, fim_iso, st.session_state.interval
                )

                if prom_data is None or len(prom_data) == 0:
                    st.error("❌ Não foi possível obter os dados dessa métrica")
                    st.session_state.form_submitted = False
                else:
                    df = pd.DataFrame(columns=["time", "values"])
                    df["time"] = prom_data.index.to_numpy()
                    df["values"] = prom_data.values

                    st.subheader(f"Query: {st.session_state.prom_query}")

                    linha = go.Scatter(
                        x=df['time'],
                        y=df["values"],
                        mode='lines',
                        line=dict(color='#8A05BE', width=1),
                        name='Data'
                    )

                    fig = go.Figure(data=[linha])
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Erro ao processar dados: {str(e)}")
                st.session_state.form_submitted = False

        st.subheader("Escolha um método de análise:")
        st.page_link("models/isolation_forest.py", label="Isolation Forest", icon="🌲")
        st.page_link("models/z_score.py", label="Z-Score", icon="💤")
        st.page_link("models/prophet_analysis.py", label="Prophet", icon="🔮")
        st.page_link("models/matrix_profile.py", label="Matrix Profile", icon="🧮")
    else:
        st.subheader("⚠️ Configuração Necessária")
        st.info("Por favor, preencha todos os campos obrigatórios.")
        st.page_link("models/homepage.py", label="🏠 Atualizar Formulário")