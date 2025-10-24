import pandas as pd
import numpy as np
import streamlit as st
import stumpy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prometheus_pandas import query

st.title("Matrix Profile")

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
    with st.spinner('🔄 Carregando dados e calculando Matrix Profile...'):
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
                    status_text.text('Calculando Matrix Profile...')
                    progress_bar.progress(75)
                    
                    # Parâmetros na sidebar
                    st.sidebar.markdown("### ⚙️ Parâmetros")
                    m = st.sidebar.slider('🪟 Tamanho da janela (m)', min_value=2, max_value=min(50, len(df)//2), value=min(6, len(df)//4), step=1)
                    std_multiplier = st.sidebar.slider('📊 Desvio Padrão para Limiar', min_value=1, max_value=5, value=3, step=1)
                    
                    # Adicionar informações sobre os dados
                    st.sidebar.markdown("### 📈 Informações dos Dados")
                    st.sidebar.metric("Total de pontos", len(df))
                    st.sidebar.metric("Período", f"{pd.to_datetime(df['time'].iloc[0]).strftime('%d/%m/%Y %H:%M')} - {pd.to_datetime(df['time'].iloc[-1]).strftime('%d/%m/%Y %H:%M')}")
                    
                    values_float = df['values'].astype(float)
                    
                    # Calcular Matrix Profile
                    mp = stumpy.stump(values_float, m=m, normalize=False)
                    
                    status_text.text('Identificando anomalias...')
                    progress_bar.progress(90)
                    
                    # Calcular limiar e anomalias
                    mean_mp = np.mean(mp[:, 0])
                    std_mp = np.std(mp[:, 0], dtype=np.float64)
                    limiar_mp = mean_mp + (std_multiplier * std_mp)
                    anomalias_idx = np.where(mp[:, 0] > limiar_mp)[0]
                    
                    # Criar DataFrame do Matrix Profile
                    matrix_profile_df = pd.DataFrame(mp[:, 0], columns=['matrix_profile'])
                    matrix_profile_df['time'] = df['time'][:len(mp)]
                    matrix_profile_df['anomaly'] = matrix_profile_df.index.isin(anomalias_idx)
                    
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
                        st.metric("🔍 Anomalias Detectadas", len(anomalias_idx))
                    with col2:
                        st.metric("📊 Taxa de Anomalia", f"{len(anomalias_idx)/len(df)*100:.2f}%")
                    with col3:
                        st.metric("📏 Tamanho da Janela", m)
                    with col4:
                        st.metric("⚡ Limiar (σ)", std_multiplier)
                    
                    # Criar subplots
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=('Série Temporal Original', 'Matrix Profile'),
                        vertical_spacing=0.08,
                        shared_xaxes=True
                    )
                    
                    # Gráfico 1: Série temporal original
                    fig.add_trace(
                        go.Scatter(
                            x=df['time'],
                            y=df['values'],
                            mode='lines',
                            line=dict(color='#8A05BE', width=1.5),
                            name='Dados Originais',
                            hovertemplate='<b>Tempo:</b> %{x}<br><b>Valor:</b> %{y:.2f}<extra></extra>'
                        ),
                        row=1, col=1
                    )
                    
                    # Anomalias na série original
                    if len(anomalias_idx) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=df.iloc[anomalias_idx]['time'],
                                y=df.iloc[anomalias_idx]['values'],
                                mode='markers',
                                marker=dict(color='red', size=8, symbol='circle'),
                                name='Anomalias',
                                hovertemplate='<b>Anomalia</b><br><b>Tempo:</b> %{x}<br><b>Valor:</b> %{y:.2f}<extra></extra>'
                            ),
                            row=1, col=1
                        )
                    
                    # Gráfico 2: Matrix Profile
                    fig.add_trace(
                        go.Scatter(
                            x=matrix_profile_df['time'],
                            y=matrix_profile_df['matrix_profile'],
                            mode='lines',
                            line=dict(color='#1f77b4', width=1.5),
                            name='Matrix Profile',
                            hovertemplate='<b>Tempo:</b> %{x}<br><b>Distância:</b> %{y:.4f}<extra></extra>'
                        ),
                        row=2, col=1
                    )
                    
                    # Linha do limiar
                    fig.add_hline(
                        y=limiar_mp,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Limiar = {limiar_mp:.4f}",
                        row=2, col=1
                    )
                    
                    # Anomalias no Matrix Profile
                    if len(anomalias_idx) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=matrix_profile_df.iloc[anomalias_idx]['time'],
                                y=matrix_profile_df.iloc[anomalias_idx]['matrix_profile'],
                                mode='markers',
                                marker=dict(color='red', size=8, symbol='circle'),
                                name='Anomalias MP',
                                showlegend=False,
                                hovertemplate='<b>Anomalia no MP</b><br><b>Tempo:</b> %{x}<br><b>Distância:</b> %{y:.4f}<extra></extra>'
                            ),
                            row=2, col=1
                        )
                    
                    # Layout do gráfico
                    fig.update_layout(
                        height=700,
                        title_text="Análise de Anomalias com Matrix Profile",
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
                    fig.update_yaxes(title_text="Distância", row=2, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Mostrar estatísticas detalhadas se houver anomalias
                    if len(anomalias_idx) > 0:
                        st.subheader("🚨 Detalhes das Anomalias")
                        
                        anomalies_df = df.iloc[anomalias_idx].copy()
                        anomalies_df['matrix_profile_score'] = mp[anomalias_idx, 0]
                        anomalies_df['time_formatted'] = pd.to_datetime(anomalies_df['time']).dt.strftime('%d/%m/%Y %H:%M:%S')
                        
                        st.dataframe(
                            anomalies_df[['time_formatted', 'values', 'matrix_profile_score']].rename(columns={
                                'time_formatted': 'Tempo',
                                'values': 'Valor',
                                'matrix_profile_score': 'Score MP'
                            }),
                            use_container_width=True
                        )
                        
                        # Opção para download dos dados
                        csv = anomalies_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Baixar Anomalias (CSV)",
                            data=csv,
                            file_name=f"anomalias_{st.session_state.prom_query.replace('/', '_')}.csv",
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