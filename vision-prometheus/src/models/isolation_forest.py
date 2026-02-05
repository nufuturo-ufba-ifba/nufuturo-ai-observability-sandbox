from prometheus_pandas import query
import pandas as pd
from sklearn.ensemble import IsolationForest
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.title("Isolation Forest")

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
    
    # Adicionar loading spinner com mensagem específica
    with st.spinner('🔄 Carregando dados e executando Isolation Forest...'):
        try:
            # Progress bar para mostrar etapas
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text('Consultando dados do Prometheus...')
            progress_bar.progress(20)
            
            # from datetime to ISO 8601
            inicio_iso = pd.to_datetime(st.session_state.begin).tz_localize('America/Sao_Paulo').strftime('%Y-%m-%dT%H:%M:%SZ')
            fim_iso = pd.to_datetime(st.session_state.end).tz_localize('America/Sao_Paulo').strftime('%Y-%m-%dT%H:%M:%SZ')

            # pull prometheus data
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
                progress_bar.progress(40)
                
                # store data in pandas dataframe
                df = pd.DataFrame(columns=["time", "values"])
                df["time"] = prom_data.index.to_numpy()
                df["values"] = prom_data.values

                # Verificar se ainda há dados após limpeza
                df['values'] = pd.to_numeric(df['values'], errors='coerce')
                df.dropna(subset=['values'], inplace=True)
                
                if len(df) == 0:
                    st.error("❌ Não há dados válidos para processar")
                    progress_bar.empty()
                    status_text.empty()
                else:
                    status_text.text('Configurando modelo...')
                    progress_bar.progress(50)
                    
                    # Parâmetros na sidebar
                    st.sidebar.markdown("### ⚙️ Parâmetros do Isolation Forest")
                    
                    contamination_level = st.sidebar.slider(
                        '🦠 Nível de Contaminação', 
                        min_value=0.001, 
                        max_value=0.1, 
                        value=0.01,
                        step=0.001, 
                        format="%.3f",
                        help="Proporção esperada de anomalias no dataset"
                    )
                    
                    days_train = st.sidebar.selectbox(
                        '📅 Dias de Treinamento', 
                        options=range(1, 8),
                        index=2,
                        help="Número de dias para usar no treinamento do modelo"
                    )
                    
                    n_estimators = st.sidebar.slider(
                        '🌲 Número de Árvores',
                        min_value=50,
                        max_value=200,
                        value=100,
                        step=10,
                        help="Número de árvores no ensemble"
                    )
                    
                    # Adicionar informações sobre os dados
                    st.sidebar.markdown("### 📈 Informações dos Dados")
                    st.sidebar.metric("Total de pontos", len(df))
                    st.sidebar.metric("Período", f"{pd.to_datetime(df['time'].iloc[0]).strftime('%d/%m/%Y %H:%M')} - {pd.to_datetime(df['time'].iloc[-1]).strftime('%d/%m/%Y %H:%M')}")
                    
                    ############
                    #
                    #       Isolation Forest Analysis
                    #
                    ###########
                    
                    status_text.text('Treinando modelo Isolation Forest...')
                    progress_bar.progress(60)
                    
                    model = IsolationForest(
                        contamination=contamination_level, 
                        n_estimators=n_estimators,
                        random_state=42
                    )

                    size = int((days_train/7) * len(df))
                    
                    # Garantir que temos dados suficientes para treinar
                    if size < 10:
                        size = min(len(df)//2, 50)  # Usar pelo menos 10 pontos ou metade dos dados
                    
                    X_train = df[:size]["values"].dropna().values.reshape(-1,1)
                    
                    if len(X_train) == 0:
                        st.error("❌ Não há dados suficientes para treinamento")
                        progress_bar.empty()
                        status_text.empty()
                    else:
                        model.fit(X_train)
                        
                        status_text.text('Detectando anomalias...')
                        progress_bar.progress(80)
                        
                        # Predições no conjunto de teste
                        X_test = df[size:]["values"].dropna().values.reshape(-1,1)
                        test_indices = df[size:]["values"].dropna().index
                        
                        if len(X_test) > 0:
                            preds = model.predict(X_test)
                            scores = model.decision_function(X_test)
                            
                            # Criar DataFrame com anomalias
                            test_df = df.iloc[test_indices].copy()
                            test_df['prediction'] = preds
                            test_df['isolation_score'] = scores
                            
                            anomalies_mask = preds == -1
                            anomalies_df = test_df[anomalies_mask].copy()
                            
                            progress_bar.progress(100)
                            status_text.text('Concluído!')
                            
                            # Limpar progress bar e status
                            progress_bar.empty()
                            status_text.empty()
                            
                            ############
                            #
                            #       Visualization
                            #
                            ###########
                            
                            # Título da análise
                            st.subheader(f"🤖 Análise: {st.session_state.prom_query}")
                            
                            # Métricas resumo
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("🔍 Anomalias Detectadas", len(anomalies_df))
                            with col2:
                                st.metric("📊 Taxa de Anomalia", f"{len(anomalies_df)/len(test_df)*100:.2f}%" if len(test_df) > 0 else "0%")
                            with col3:
                                st.metric("📅 Dias de Treino", days_train)
                            with col4:
                                st.metric("🦠 Contaminação", f"{contamination_level:.3f}")
                            
                            # Criar subplots
                            fig = make_subplots(
                                rows=2, cols=1,
                                subplot_titles=('Série Temporal com Anomalias', 'Isolation Scores'),
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
                            
                            # Destacar área de treinamento
                            fig.add_shape(
                                type="rect",
                                x0=df['time'].min(),
                                x1=df['time'].iloc[size-1] if size < len(df) else df['time'].max(),
                                y0=df['values'].min(),
                                y1=df['values'].max(),
                                fillcolor="lightblue",
                                opacity=0.2,
                                line=dict(width=0),
                                row=1, col=1
                            )
                            
                            # Adicionar anotação para área de treinamento
                            fig.add_annotation(
                                x=df['time'].iloc[size//2] if size < len(df) else df['time'].iloc[len(df)//4],
                                y=df['values'].max(),
                                text="Área de Treinamento",
                                showarrow=True,
                                arrowhead=2,
                                arrowcolor="blue",
                                bgcolor="lightblue",
                                row=1, col=1
                            )
                            
                            # Anomalias na série original
                            if len(anomalies_df) > 0:
                                fig.add_trace(
                                    go.Scatter(
                                        x=anomalies_df['time'],
                                        y=anomalies_df['values'],
                                        mode='markers',
                                        marker=dict(color='red', size=8, symbol='circle'),
                                        name='Anomalias',
                                        hovertemplate='<b>Anomalia</b><br><b>Tempo:</b> %{x}<br><b>Valor:</b> %{y:.2f}<extra></extra>'
                                    ),
                                    row=1, col=1
                                )
                            
                            # Gráfico 2: Isolation Scores
                            if len(test_df) > 0:
                                fig.add_trace(
                                    go.Scatter(
                                        x=test_df['time'],
                                        y=test_df['isolation_score'],
                                        mode='lines+markers',
                                        line=dict(color='#1f77b4', width=1.5),
                                        marker=dict(size=4),
                                        name='Isolation Score',
                                        hovertemplate='<b>Tempo:</b> %{x}<br><b>Score:</b> %{y:.4f}<extra></extra>'
                                    ),
                                    row=2, col=1
                                )
                                
                                # Linha de limiar (0 é o limiar padrão do Isolation Forest)
                                fig.add_hline(
                                    y=0,
                                    line_dash="dash",
                                    line_color="red",
                                    annotation_text="Limiar de Anomalia",
                                    row=2, col=1
                                )
                                
                                # Anomalias no gráfico de scores
                                if len(anomalies_df) > 0:
                                    fig.add_trace(
                                        go.Scatter(
                                            x=anomalies_df['time'],
                                            y=anomalies_df['isolation_score'],
                                            mode='markers',
                                            marker=dict(color='red', size=8, symbol='circle'),
                                            name='Anomalias Score',
                                            showlegend=False,
                                            hovertemplate='<b>Anomalia</b><br><b>Tempo:</b> %{x}<br><b>Score:</b> %{y:.4f}<extra></extra>'
                                        ),
                                        row=2, col=1
                                    )
                            
                            # Layout do gráfico
                            fig.update_layout(
                                height=700,
                                title_text="Análise de Anomalias com Isolation Forest",
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
                            fig.update_yaxes(title_text="Isolation Score", row=2, col=1)
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Mostrar estatísticas detalhadas se houver anomalias
                            if len(anomalies_df) > 0:
                                st.subheader("🚨 Detalhes das Anomalias")
                                
                                # Preparar dados para exibição
                                display_anomalies = anomalies_df.copy()
                                display_anomalies['time_formatted'] = pd.to_datetime(display_anomalies['time']).dt.strftime('%d/%m/%Y %H:%M:%S')
                                
                                st.dataframe(
                                    display_anomalies[['time_formatted', 'values', 'isolation_score']].rename(columns={
                                        'time_formatted': 'Tempo',
                                        'values': 'Valor',
                                        'isolation_score': 'Score de Isolação'
                                    }),
                                    use_container_width=True
                                )
                                
                                # Estatísticas das anomalias
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("🔢 Score Médio", f"{anomalies_df['isolation_score'].mean():.4f}")
                                    st.metric("📊 Score Mínimo", f"{anomalies_df['isolation_score'].min():.4f}")
                                with col2:
                                    st.metric("📈 Score Máximo", f"{anomalies_df['isolation_score'].max():.4f}")
                                    st.metric("📏 Desvio Padrão", f"{anomalies_df['isolation_score'].std():.4f}")
                                
                                # Opção para download dos dados
                                csv = display_anomalies.to_csv(index=False)
                                st.download_button(
                                    label="📥 Baixar Anomalias (CSV)",
                                    data=csv,
                                    file_name=f"anomalias_isolation_{st.session_state.prom_query.replace('/', '_')}.csv",
                                    mime='text/csv'
                                )
                                
                                # Distribuição dos scores
                                st.subheader("📊 Distribuição dos Isolation Scores")
                                fig_hist = go.Figure()
                                fig_hist.add_trace(go.Histogram(
                                    x=test_df['isolation_score'],
                                    nbinsx=30,
                                    name='Scores Normais',
                                    opacity=0.7,
                                    marker_color='blue'
                                ))
                                fig_hist.add_trace(go.Histogram(
                                    x=anomalies_df['isolation_score'],
                                    nbinsx=30,
                                    name='Scores Anômalos',
                                    opacity=0.7,
                                    marker_color='red'
                                ))
                                fig_hist.update_layout(
                                    title="Distribuição dos Isolation Scores",
                                    xaxis_title="Isolation Score",
                                    yaxis_title="Frequência",
                                    barmode='overlay'
                                )
                                st.plotly_chart(fig_hist, use_container_width=True)
                                
                            else:
                                st.info("✅ Nenhuma anomalia foi detectada com os parâmetros atuais.")
                                
                                # Mostrar distribuição mesmo sem anomalias
                                if len(test_df) > 0:
                                    st.subheader("📊 Distribuição dos Isolation Scores")
                                    fig_hist = go.Figure()
                                    fig_hist.add_trace(go.Histogram(
                                        x=test_df['isolation_score'],
                                        nbinsx=30,
                                        name='Todos os Scores',
                                        opacity=0.7,
                                        marker_color='blue'
                                    ))
                                    fig_hist.update_layout(
                                        title="Distribuição dos Isolation Scores",
                                        xaxis_title="Isolation Score",
                                        yaxis_title="Frequência"
                                    )
                                    st.plotly_chart(fig_hist, use_container_width=True)
                        else:
                            st.warning("⚠️ Não há dados suficientes para teste após o período de treinamento.")
                            progress_bar.empty()
                            status_text.empty()

        except Exception as e:
            st.error(f"❌ Erro ao processar dados: {str(e)}")
            # Debug info (opcional - remover em produção)
            with st.expander("🔧 Informações de Debug"):
                st.exception(e)

else:
    st.subheader("⚠️ Configuração Necessária")
    st.info("Por favor, vá para a página inicial e preencha todos os campos obrigatórios.")
    st.page_link("models/homepage.py", label="🏠 Ir para Home", use_container_width=True)