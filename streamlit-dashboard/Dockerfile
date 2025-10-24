FROM python:3.11-slim

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

# Instala o cron e outras dependências necessárias
RUN apt-get update && apt-get install -y cron vim

# Copia todos os arquivos do diretório atual para o diretório de trabalho no contêiner
COPY . /app

# Instala as dependências listadas no arquivo requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copia o crontab file para /etc/cron.d
COPY mycron /etc/cron.d/mycron

# Dá permissão ao cron job
RUN chmod 0644 /etc/cron.d/mycron

# Aplica o cron job
RUN crontab /etc/cron.d/mycron

# Cria o arquivo de log necessário para o cron
RUN touch /var/log/cron.log

# Exponha a porta onde o Streamlit será executado
EXPOSE 8501

# Comando para rodar o cron e o Streamlit
CMD ["sh", "-c", "cron && streamlit run app.py"]
