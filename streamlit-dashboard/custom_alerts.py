import requests
import json
from datetime import datetime, timedelta, timezone

# URL do seu Alertmanager usando a API v2
alertmanager_url = "https://alertmanager.mateusdata.com.br/api/v2/alerts"

# Dados de um alerta no formato Prometheus
alert_data = [
    {
        "labels": {
            "alertname": "minha_metrica",  # Nome do alerta conforme definido na regra
            "severity": "critical",
            "instance": "localhost"
        },
        "annotations": {
            "summary": "Latência do minha_metrica Kafka Alta",
            "description": " teste teste teste A latência do producer Kafka ultrapassou 999."
        },
        "startsAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),  # Data de início do alerta
        "endsAt": (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),  # Data de fim do alerta (1 hora depois)
        "generatorURL": "http://localhost:9090/graph?g0.expr=shore_kafka_producer_latency_count&g0.tab=1"
    }
]

# Configuração de headers (se você estiver usando HTTPS e precisar de autenticação, adicione a autenticação aqui)
headers = {
    'Content-Type': 'application/json',
}

# Enviando a requisição POST para o Alertmanager
try:
    response = requests.post(alertmanager_url, headers=headers, data=json.dumps(alert_data), verify=True)  # 'verify=True' garante a verificação do certificado SSL
    print(f"Status Code: {response.status_code}")
    print(f"Response Text: {response.text}")
    if response.status_code == 200:
        print("Alerta enviado com sucesso!")
    else:
        print(f"Erro ao enviar alerta. Status code: {response.status_code}, Resposta: {response.text}")
except requests.exceptions.RequestException as e:
    print(f"Erro na requisição: {e}")
