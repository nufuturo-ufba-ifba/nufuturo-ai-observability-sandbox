import pandas as pd
import streamlit as st
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prometheus_pandas import query
import numpy as np

st.title("Prophet")

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

# Only run if all fields are provided
if all([st.session_state.prom_query, st.session_state.begin, st.session_state.end, st.session_state.interval]):
    
    # Adicionar loading spinner com mensagem específica
    with st.spinner('🔄 Carregando dados e executando Prophet...'):
        try:
            # Progress bar para mostrar etapas
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text('Consultando dados do Prometheus...')
            progress_bar.progress(20)
            
            # Convert input datetime to ISO 8601 format
            inicio_iso = pd.to_datetime(st.session_state.begin).tz_localize('America/Sao_Paulo').strftime('%Y-%m-%dT%H:%M:%SZ')
            fim_iso = pd.to_datetime(st.session_state.end).tz_localize('America/Sao_Paulo').strftime('%Y-%m-%dT%H:%M:%SZ')

            # Pull data from Prometheus
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
                
                # Store Prometheus data in DataFrame
                df = pd.DataFrame(columns=["time", "values"])
                df["time"] = prom_data.index.to_numpy()
                df["values"] = prom_data.values

                # Certifique-se de que 'time' está em formato datetime
                df['time'] = pd.to_datetime(df['time'])
                df['values'] = pd.to_numeric(df['values'], errors='coerce')
                df.dropna(subset=['values'], inplace=True)

                # Verificar se ainda há dados após limpeza
                if len(df) == 0:
                    st.error("❌ Não há dados válidos para processar")
                    progress_bar.empty()
                    status_text.empty()
                else:
                    status_text.text('Configurando modelo Prophet...')
                    progress_bar.progress(50)
                    
                    # Parâmetros na sidebar
                    st.sidebar.markdown("### ⚙️ Parâmetros do Prophet")
                    
                    days_train = st.sidebar.selectbox(
                        '📅 Dias de Treinamento', 
                        options=range(1, 8),
                        index=2,
                        help="Número de dias para usar no treinamento do modelo"
                    )
                    
                    interval_width = st.sidebar.slider(
                        '📊 Largura do Intervalo de Confiança',
                        min_value=0.80,
                        max_value=0.999,
                        value=0.95,
                        step=0.01,
                        format="%.2f",
                        help="Largura do intervalo de predição (95% = intervalo de 95%)"
                    )
                    
                    growth_model = st.sidebar.selectbox(
                        '📈 Modelo de Crescimento',
                        options=['linear', 'flat'],
                        index=1,
                        help="Tipo de tendência: linear (crescimento/decrescimento) ou flat (constante)"
                    )
                    
                    weekly_seasonality = st.sidebar.checkbox(
                        '📅 Sazonalidade Semanal',
                        value=False,
                        help="Incluir padrões semanais na previsão"
                    )
                    
                    yearly_seasonality = st.sidebar.checkbox(
                        '🗓️ Sazonalidade Anual',
                        value=False,
                        help="Incluir padrões anuais na previsão"
                    )
                    
                    daily_seasonality = st.sidebar.checkbox(
                        '🌅 Sazonalidade Diária',
                        value=False,
                        help="Incluir padrões diários na previsão"
                    )
                    
                    # Adicionar informações sobre os dados
                    st.sidebar.markdown("### 📈 Informações dos Dados")
                    st.sidebar.metric("Total de pontos", len(df))
                    st.sidebar.metric("Período", f"{df['time'].min().strftime('%d/%m/%Y %H:%M')} - {df['time'].max().strftime('%d/%m/%Y %H:%M')}")
                    
                    ############
                    #
                    #       Prophet Analysis
                    #
                    ###########
                    
                    status_text.text('Dividindo dados em treino e teste...')
                    progress_bar.progress(60)
                    
                    size = int((days_train / 7) * len(df))
                    
                    # Garantir que temos dados suficientes para treinar e testar
                    if size < 10:
                        size = min(len(df)//2, max(10, len(df)//3))
                    
                    # Dividir dados em treino e teste
                    train = df[:size].copy()
                    test = df[size:].copy()
                    
                    if len(train) == 0 or len(test) == 0:
                        st.error("❌ Não há dados suficientes para divisão treino/teste")
                        progress_bar.empty()
                        status_text.empty()
                    else:
                        status_text.text('Treinando modelo Prophet...')
                        progress_bar.progress(70)
                        
                        # Preparar dados para o Prophet (necessita colunas 'ds' e 'y')
                        train_prophet = train.rename(columns={"time": "ds", "values": "y"})
                        test_prophet = test.rename(columns={"time": "ds", "values": "y"})
                        
                        # Criação e treino do modelo Prophet
                        model = Prophet(
                            interval_width=interval_width,
                            weekly_seasonality=weekly_seasonality,
                            yearly_seasonality=yearly_seasonality,
                            daily_seasonality=daily_seasonality,
                            growth=growth_model
                        )
                        
                        # Suprimir logs do Prophet
                        import logging
                        logging.getLogger('prophet').setLevel(logging.WARNING)
                        
                        model.fit(train_prophet)
                        
                        status_text.text('Gerando previsões...')
                        progress_bar.progress(85)
                        
                        # Previsão com Prophet
                        future = test_prophet[['ds']].copy()
                        preds = model.predict(future)
                        
                        # Renomear colunas para consistência
                        preds.rename(columns={"ds": "time"}, inplace=True)
                        
                        # Mesclagem dos dados reais com os previstos
                        merged = pd.merge(
                            test, 
                            preds[['time', 'yhat', 'yhat_lower', 'yhat_upper']], 
                            on='time'
                        )
                        
                        # Garantir que não há valores NaN
                        merged = merged.dropna()
                        
                        if len(merged) == 0:
                            st.error("❌ Não foi possível gerar previsões válidas")
                            progress_bar.empty()
                            status_text.empty()
                        else:
                            status_text.text('Identificando anomalias...')
                            progress_bar.progress(95)
                            
                            # Identificação das anomalias
                            merged['anomaly'] = merged.apply(
                                lambda row: 1 if (row['values'] < row['yhat_lower']) or (row['values'] > row['yhat_upper']) else 0, 
                                axis=1
                            )
                            
                            # Calcular desvio da previsão
                            merged['prediction_error'] = abs(merged['values'] - merged['yhat'])
                            merged['relative_error'] = merged['prediction_error'] / abs(merged['yhat']) * 100
                            
                            anomalies_df = merged[merged['anomaly'] == 1].copy()
                            
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
                            st.subheader(f"🔮 Análise: {st.session_state.prom_query}")
                            
                            # Métricas resumo
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("🔍 Anomalias Detectadas", len(anomalies_df))
                            with col2:
                                st.metric("📊 Taxa de Anomalia", f"{len(anomalies_df)/len(merged)*100:.2f}%" if len(merged) > 0 else "0%")
                            with col3:
                                st.metric("📅 Dias de Treino", days_train)
                            with col4:
                                mae = merged['prediction_error'].mean()
                                st.metric("📏 Erro Médio Absoluto", f"{mae:.4f}")
                            
                            # Criar subplots
                            fig = make_subplots(
                                rows=3, cols=1,
                                subplot_titles=(
                                    'Série Temporal com Previsões e Anomalias',
                                    'Bandas de Confiança (Zoom na Área de Teste)',
                                    'Erro de Previsão'
                                ),
                                vertical_spacing=0.06,
                                shared_xaxes=True,
                                row_heights=[0.4, 0.4, 0.2]
                            )
                            
                            # Gráfico 1: Série temporal completa
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
                                fillcolor="lightgreen",
                                opacity=0.2,
                                line=dict(width=0),
                                row=1, col=1
                            )
                            
                            # Anotação para área de treinamento
                            fig.add_annotation(
                                x=df['time'].iloc[size//2] if size < len(df) else df['time'].iloc[len(df)//4],
                                y=df['values'].max(),
                                text="Área de Treinamento",
                                showarrow=True,
                                arrowhead=2,
                                arrowcolor="green",
                                bgcolor="lightgreen",
                                row=1, col=1
                            )
                            
                            # Previsões
                            if len(merged) > 0:
                                fig.add_trace(
                                    go.Scatter(
                                        x=merged['time'],
                                        y=merged['yhat'],
                                        mode='lines',
                                        line=dict(color='orange', width=2, dash='dash'),
                                        name='Previsão',
                                        hovertemplate='<b>Tempo:</b> %{x}<br><b>Previsão:</b> %{y:.2f}<extra></extra>'
                                    ),
                                    row=1, col=1
                                )
                            
                            # Anomalias
                            if len(anomalies_df) > 0:
                                fig.add_trace(
                                    go.Scatter(
                                        x=anomalies_df['time'],
                                        y=anomalies_df['values'],
                                        mode='markers',
                                        marker=dict(color='red', size=8, symbol='circle'),
                                        name='Anomalias',
                                        hovertemplate='<b>Anomalia</b><br><b>Tempo:</b> %{x}<br><b>Valor:</b> %{y:.2f}<br><b>Previsão:</b> %{customdata:.2f}<extra></extra>',
                                        customdata=anomalies_df['yhat']
                                    ),
                                    row=1, col=1
                                )
                            
                            # Gráfico 2: Zoom na área de teste com bandas de confiança
                            if len(merged) > 0:
                                # Banda superior
                                fig.add_trace(
                                    go.Scatter(
                                        x=merged['time'],
                                        y=merged['yhat_upper'],
                                        mode='lines',
                                        line=dict(width=0),
                                        showlegend=False,
                                        hoverinfo='skip'
                                    ),
                                    row=2, col=1
                                )
                                
                                # Banda inferior e preenchimento
                                fig.add_trace(
                                    go.Scatter(
                                        x=merged['time'],
                                        y=merged['yhat_lower'],
                                        mode='lines',
                                        line=dict(width=0),
                                        fill='tonexty',
                                        fillcolor='rgba(255, 165, 0, 0.2)',
                                        name='Banda de Confiança',
                                        hovertemplate='<b>Tempo:</b> %{x}<br><b>Limite Inferior:</b> %{y:.2f}<extra></extra>'
                                    ),
                                    row=2, col=1
                                )
                                
                                # Dados reais na área de teste
                                fig.add_trace(
                                    go.Scatter(
                                        x=test['time'],
                                        y=test['values'],
                                        mode='lines+markers',
                                        line=dict(color='#8A05BE', width=2),
                                        marker=dict(size=4),
                                        name='Dados Teste',
                                        hovertemplate='<b>Tempo:</b> %{x}<br><b>Valor Real:</b> %{y:.2f}<extra></extra>'
                                    ),
                                    row=2, col=1
                                )
                                
                                # Previsão na área de teste
                                fig.add_trace(
                                    go.Scatter(
                                        x=merged['time'],
                                        y=merged['yhat'],
                                        mode='lines',
                                        line=dict(color='orange', width=2),
                                        name='Previsão Teste',
                                        showlegend=False,
                                        hovertemplate='<b>Tempo:</b> %{x}<br><b>Previsão:</b> %{y:.2f}<extra></extra>'
                                    ),
                                    row=2, col=1
                                )
                                
                                # Anomalias no gráfico de teste
                                if len(anomalies_df) > 0:
                                    fig.add_trace(
                                        go.Scatter(
                                            x=anomalies_df['time'],
                                            y=anomalies_df['values'],
                                            mode='markers',
                                            marker=dict(color='red', size=10, symbol='circle'),
                                            name='Anomalias Teste',
                                            showlegend=False,
                                            hovertemplate='<b>Anomalia</b><br><b>Tempo:</b> %{x}<br><b>Valor:</b> %{y:.2f}<extra></extra>'
                                        ),
                                        row=2, col=1
                                    )
                                
                                # Gráfico 3: Erro de previsão
                                fig.add_trace(
                                    go.Scatter(
                                        x=merged['time'],
                                        y=merged['prediction_error'],
                                        mode='lines+markers',
                                        line=dict(color='purple', width=1.5),
                                        marker=dict(size=4),
                                        name='Erro Absoluto',
                                        hovertemplate='<b>Tempo:</b> %{x}<br><b>Erro:</b> %{y:.4f}<extra></extra>'
                                    ),
                                    row=3, col=1
                                )
                                
                                # Destacar erros das anomalias
                                if len(anomalies_df) > 0:
                                    fig.add_trace(
                                        go.Scatter(
                                            x=anomalies_df['time'],
                                            y=anomalies_df['prediction_error'],
                                            mode='markers',
                                            marker=dict(color='red', size=8, symbol='circle'),
                                            name='Erro Anomalias',
                                            showlegend=False,
                                            hovertemplate='<b>Erro Anomalia</b><br><b>Tempo:</b> %{x}<br><b>Erro:</b> %{y:.4f}<extra></extra>'
                                        ),
                                        row=3, col=1
                                    )
                            
                            # Layout do gráfico
                            fig.update_layout(
                                height=900,
                                title_text="Análise de Anomalias com Prophet",
                                title_x=0.5,
                                showlegend=True,
                                legend=dict(
                                    yanchor="top",
                                    y=0.99,
                                    xanchor="left",
                                    x=1.01
                                )
                            )
                            
                            fig.update_xaxes(title_text="Tempo", row=3, col=1)
                            fig.update_yaxes(title_text="Valor", row=1, col=1)
                            fig.update_yaxes(title_text="Valor", row=2, col=1)
                            fig.update_yaxes(title_text="Erro", row=3, col=1)
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Mostrar estatísticas detalhadas se houver anomalias
                            if len(anomalies_df) > 0:
                                st.subheader("🚨 Detalhes das Anomalias")
                                
                                # Preparar dados para exibição
                                display_anomalies = anomalies_df.copy()
                                display_anomalies['time_formatted'] = display_anomalies['time'].dt.strftime('%d/%m/%Y %H:%M:%S')
                                
                                st.dataframe(
                                    display_anomalies[[
                                        'time_formatted', 'values', 'yhat', 
                                        'yhat_lower', 'yhat_upper', 'prediction_error', 'relative_error'
                                    ]].rename(columns={
                                        'time_formatted': 'Tempo',
                                        'values': 'Valor Real',
                                        'yhat': 'Previsão',
                                        'yhat_lower': 'Limite Inferior',
                                        'yhat_upper': 'Limite Superior',
                                        'prediction_error': 'Erro Absoluto',
                                        'relative_error': 'Erro Relativo (%)'
                                    }),
                                    use_container_width=True
                                )
                                
                                # Estatísticas das anomalias
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("🎯 Erro Médio", f"{anomalies_df['prediction_error'].mean():.4f}")
                                    st.metric("📊 Erro Mínimo", f"{anomalies_df['prediction_error'].min():.4f}")
                                    st.metric("📈 Erro Máximo", f"{anomalies_df['prediction_error'].max():.4f}")
                                with col2:
                                    st.metric("📏 Desvio Padrão", f"{anomalies_df['prediction_error'].std():.4f}")
                                    st.metric("🎯 Erro Relativo Médio", f"{anomalies_df['relative_error'].mean():.2f}%")
                                    rmse = np.sqrt(anomalies_df['prediction_error']**2).mean()
                                    st.metric("📊 RMSE", f"{rmse:.4f}")
                                
                                # Opção para download dos dados
                                csv = display_anomalies.to_csv(index=False)
                                st.download_button(
                                    label="📥 Baixar Anomalias (CSV)",
                                    data=csv,
                                    file_name=f"anomalias_prophet_{st.session_state.prom_query.replace('/', '_')}.csv",
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
    st.page_link("models/homepage.py", label="🏠 Ir para Home", use_container_width=True)