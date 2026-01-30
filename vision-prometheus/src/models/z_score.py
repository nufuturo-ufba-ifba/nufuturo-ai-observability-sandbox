import pandas as pd
import streamlit as st
from prometheus_pandas import query
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.title("Z-Score")

# Inicializar as variáveis do session_state se não existirem
if 'prom_query' not in st.session_state:
    st.session_state.prom_query = None
if 'begin' not in st.session_state:
    st.session_state.begin = None
if 'end' not in st.session_state:
    st.session_state.end = None
if 'interval' not in st.session_state:
    st.session_state.interval = None
if 'prom_connection' not in st.session_state:
    st.session_state.prom_connection = None

if all([st.session_state.prom_query, st.session_state.begin, st.session_state.end, st.session_state.interval]):
    
    # Adicionar loading spinner com mensagem mais específica
    with st.spinner('🔄 Carregando dados e calculando Z-Score...'):
        try:
            # Converter datas para formato ISO
            inicio_iso = pd.to_datetime(st.session_state.begin).tz_localize('America/Sao_Paulo').strftime('%Y-%m-%dT%H:%M:%SZ')
            fim_iso = pd.to_datetime(st.session_state.end).tz_localize('America/Sao_Paulo').strftime('%Y-%m-%dT%H:%M:%SZ')
            
            # Progress bar para mostrar etapas
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text('Consultando dados do Prometheus...')
            progress_bar.progress(25)
            
            prom_data = st.session_state.prom_connection.query_range(
                st.session_state.prom_query, inicio_iso, fim_iso, st.session_state.interval
            )
            
            # Verificar se retornou dados
            if prom_data is None or len(prom_data) == 0:
                st.error("❌ Não foi possível obter os dados dessa métrica")
                progress_bar.empty()
                status_text.empty()
            else:
                status_text.text('Processando dados...')
                progress_bar.progress(50)
                
                # Criar DataFrame
                df = pd.DataFrame(columns=["time", "values"])
                df["time"] = prom_data.index.to_numpy()
                df["values"] = prom_data.values
                
                df['values'] = pd.to_numeric(df['values'], errors='coerce')
                df.dropna(subset=['values'], inplace=True)
                
                # Verificar se ainda há dados após limpeza
                if len(df) == 0:
                    st.error("❌ Não há dados válidos para processar")
                    progress_bar.empty()
                    status_text.empty()
                else:
                    status_text.text('Calculando Z-Score...')
                    progress_bar.progress(75)
                    
                    # Parâmetros na sidebar (mantendo apenas o window original)
                    st.sidebar.markdown("### ⚙️ Parâmetros")
                    window = st.sidebar.slider('Time window size for Z-Score calculation', min_value=2, max_value=12, value=7, step=1)
                    
                    # Adicionar informações sobre os dados
                    st.sidebar.markdown("### 📈 Informações dos Dados")
                    st.sidebar.metric("Total de pontos", len(df))
                    st.sidebar.metric("Período", f"{pd.to_datetime(df['time'].iloc[0]).strftime('%d/%m/%Y %H:%M')} - {pd.to_datetime(df['time'].iloc[-1]).strftime('%d/%m/%Y %H:%M')}")
                    
                    # Calculate mean and std deviation for Z-Score (mantendo lógica original)
                    df['mean_values'] = [stats.trim_mean(df.iloc[i:i+1]['values'].astype(float), 0.1) if len(df.iloc[i:i+1]['values']) > 0 else df.iloc[i]['values'] for i in range(len(df))]
                    df['std_values'] = [stats.mstats.trimmed_std(df.iloc[i:i+1]['values'].astype(float), limits=(0.1, 0.1)) if len(df.iloc[i:i+1]['values']) > 0 else 1.0 for i in range(len(df))]
                    
                    # Calculate Z-Score (mantendo lógica original)
                    zscore = (df['values'] - df['mean_values'].rolling(window, closed='left').mean()) / df['mean_values'].rolling(window, closed='left').std()
                    
                    status_text.text('Identificando anomalias...')
                    progress_bar.progress(90)
                    
                    # Create a DataFrame to store Z-Score results (mantendo lógica original)
                    zscore_df = pd.DataFrame(zscore)
                    zscore_df.columns = ['zscore']
                    zscore_df['time'] = df['time']
                    zscore_df['warning'] = (zscore_df['zscore'] > 4) | (zscore_df['zscore'] < -4)  # Trigger warning for anomalies
                    zscore_df['plot'] = [df['values'].iloc[i] if zscore_df['warning'].iloc[i] else None for i in range(len(df))]
                    
                    # Contar anomalias
                    anomalias_count = zscore_df['warning'].sum()
                    
                    progress_bar.progress(100)
                    status_text.text('Concluído!')
                    
                    # Limpar progress bar e status
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Título da análise
                    st.subheader(f"📊 Análise: {st.session_state.prom_query}")
                    
                    # Métricas resumo
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🔍 Anomalias Detectadas", anomalias_count)
                    with col2:
                        st.metric("📊 Taxa de Anomalia", f"{anomalias_count/len(df)*100:.2f}%")
                    with col3:
                        st.metric("📏 Janela Temporal", window)
                    with col4:
                        st.metric("⚡ Limiar Z-Score", "±4")
                    
                    # Criar subplots
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=('Série Temporal Original', 'Z-Score'),
                        vertical_spacing=0.08,
                        shared_xaxes=True
                    )
                    
                    # Gráfico 1: Série temporal original
                    fig.add_trace(
                        go.Scatter(
                            x=zscore_df['time'],
                            y=df['values'],
                            mode='lines',
                            line=dict(color='#8A05BE', width=1.5),
                            name='Data',
                            hovertemplate='<b>Tempo:</b> %{x}<br><b>Valor:</b> %{y:.2f}<extra></extra>'
                        ),
                        row=1, col=1
                    )
                    
                    # Anomalias na série original
                    if anomalias_count > 0:
                        anomalias_data = zscore_df.dropna(subset=['plot'])
                        fig.add_trace(
                            go.Scatter(
                                x=anomalias_data['time'],
                                y=anomalias_data['plot'],
                                mode='markers',
                                marker=dict(color='red', size=8, symbol='circle'),
                                name='Warning (Anomalies)',
                                hovertemplate='<b>Anomalia</b><br><b>Tempo:</b> %{x}<br><b>Valor:</b> %{y:.2f}<extra></extra>'
                            ),
                            row=1, col=1
                        )
                    
                    # Gráfico 2: Z-Score
                    fig.add_trace(
                        go.Scatter(
                            x=zscore_df['time'],
                            y=zscore_df['zscore'],
                            mode='lines',
                            line=dict(color='#1f77b4', width=1.5),
                            name='Z-Score',
                            hovertemplate='<b>Tempo:</b> %{x}<br><b>Z-Score:</b> %{y:.2f}<extra></extra>'
                        ),
                        row=2, col=1
                    )
                    
                    # Linhas de limiar no Z-Score
                    fig.add_hline(
                        y=4,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Limiar = +4",
                        row=2, col=1
                    )
                    
                    fig.add_hline(
                        y=-4,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Limiar = -4",
                        row=2, col=1
                    )
                    
                    # Anomalias no Z-Score
                    if anomalias_count > 0:
                        anomalias_zscore = zscore_df[zscore_df['warning']]
                        fig.add_trace(
                            go.Scatter(
                                x=anomalias_zscore['time'],
                                y=anomalias_zscore['zscore'],
                                mode='markers',
                                marker=dict(color='red', size=8, symbol='circle'),
                                name='Anomalias Z-Score',
                                showlegend=False,
                                hovertemplate='<b>Anomalia Z-Score</b><br><b>Tempo:</b> %{x}<br><b>Z-Score:</b> %{y:.2f}<extra></extra>'
                            ),
                            row=2, col=1
                        )
                    
                    # Layout do gráfico
                    fig.update_layout(
                        height=700,
                        title_text="Análise de Anomalias com Z-Score",
                        title_x=0.5,
                        showlegend=True,
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=1.01
                        )
                    )
                    
                    fig.update_xaxes(title_text="Tempo", row=2, col=1)
                    fig.update_yaxes(title_text="Valor", row=1, col=1)
                    fig.update_yaxes(title_text="Z-Score", row=2, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Mostrar estatísticas detalhadas se houver anomalias
                    if anomalias_count > 0:
                        st.subheader("🚨 Detalhes das Anomalias")
                        
                        anomalies_details = zscore_df[zscore_df['warning']].copy()
                        anomalies_details['time_formatted'] = pd.to_datetime(anomalies_details['time']).dt.strftime('%d/%m/%Y %H:%M:%S')
                        anomalies_details['values'] = [df['values'].iloc[i] for i in anomalies_details.index]
                        
                        st.dataframe(
                            anomalies_details[['time_formatted', 'values', 'zscore']].rename(columns={
                                'time_formatted': 'Tempo',
                                'values': 'Valor',
                                'zscore': 'Z-Score'
                            }),
                            use_container_width=True
                        )
                        
                        # Opção para download dos dados
                        csv = anomalies_details.to_csv(index=False)
                        st.download_button(
                            label="📥 Baixar Anomalias (CSV)",
                            data=csv,
                            file_name=f"anomalias_zscore_{st.session_state.prom_query.replace('/', '_')}.csv",
                            mime='text/csv'
                        )
                    else:
                        st.info("✅ Nenhuma anomalia foi detectada com os parâmetros atuais.")
                        
        except Exception as e:
            st.error(f"❌ Erro ao processar dados: {str(e)}")
            # Debug info (opcional - remover em produção)
            with st.expander("🔧 Informações de Debug"):
                st.exception(e)

else:
    st.subheader("⚠️ Configuração Necessária")
    st.info("Por favor, vá para a página inicial e preencha todos os campos obrigatórios.")
    st.page_link("./homepage.py", label="🏠 Ir para Home", use_container_width=True)