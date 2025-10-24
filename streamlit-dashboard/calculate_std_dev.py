import requests
import numpy as np
from prometheus_api_client import PrometheusConnect
from datetime import datetime, timedelta, timezone
import json
from dotenv import load_dotenv
import os

load_dotenv()
BASE_URL = os.getenv("BASE_URL")
ALERTMANAGER_URL = os.getenv("ALERTMANAGER_URL")
STANDARD_DEVIATION = float(os.getenv("STANDARD_DEVIATION", 4))  # Default to 4 if not set
HEADERS = {"Content-Type": "application/json"}

print("Conectando ao Prometheus...")
prom = PrometheusConnect(url=BASE_URL, disable_ssl=True)
print("Conexão com Prometheus estabelecida.")

def get_series_history(metric_name, start_time, end_time, step):
    print(f"Obtendo dados para a métrica: {metric_name} entre {start_time} e {end_time} com intervalo de {step} segundos...")
    query = f"avg_over_time({metric_name}[{step}s])"
    try:
        result = prom.custom_query_range(
            query=query,
            start_time=start_time,
            end_time=end_time,
            step=step
        )
        print(f"Dados obtidos para a métrica {metric_name}.")
        return result
    except Exception as e:
        print(f"Erro ao obter dados para a métrica {metric_name}: {e}")
        return []

def calculate_std_dev(values):
    print("Calculando o desvio padrão...")
    numeric_values = [float(value[1]) for value in values if isinstance(value[1], (int, float, str)) and str(value[1]).replace('.', '', 1).isdigit()]
    if len(numeric_values) > 1:
        std_dev = np.std(numeric_values)
        print(f"Desvio padrão calculado: {std_dev}")
        return std_dev
    print("Não há dados suficientes para calcular o desvio padrão.")
    return 0

def calculate_mean(values):
    print("Calculando a média...")
    numeric_values = [float(value[1]) for value in values if isinstance(value[1], (int, float, str)) and str(value[1]).replace('.', '', 1).isdigit()]
    if numeric_values:
        mean = np.mean(numeric_values)
        print(f"Média calculada: {mean}")
        return mean
    print("Não há dados suficientes para calcular a média.")
    return 0

def send_alert(alertname, summary, description):
    alert_data = [
        {
            "labels": {
                "alertname": alertname,
                "severity": "critical",
                "instance": "localhost"
            },
            "annotations": {
                "summary": summary,
                "description": description
            },
            "startsAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "endsAt": (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "generatorURL": "https://prometheus.mateusdata.com.br/graph?g0.expr=custom_metric&g0.tab=1"
        }
    ]
    try:
        response = requests.post(ALERTMANAGER_URL, headers=HEADERS, data=json.dumps(alert_data), verify=True)
        if response.status_code == 200:
            print("Alerta enviado com sucesso!")
        else:
            print(f"Erro ao enviar alerta. Status code: {response.status_code}, Resposta: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Erro na requisição: {e}")

def main():
    # Ler nomes das métricas do arquivo txt
    with open('metrics.txt', 'r') as file:
        metric_names = [line.strip() for line in file]

    now = datetime.now(timezone.utc)
    start_time = now - timedelta(days=2)
    end_time = now

    print(f"Iniciando o processamento das métricas entre {start_time} e {end_time}...")
    for metric in metric_names:
        print(f"\nProcessando métrica: {metric}")
        data = get_series_history(metric, start_time, end_time, step=60)

        if data:
            all_values = []
            for series in data:
                all_values.extend(series["values"])
                print(f"Dados de série {series['metric']} coletados.")

            if all_values:
                std_dev = calculate_std_dev(all_values)
                mean = calculate_mean(all_values)
                
                threshold_up = mean + STANDARD_DEVIATION * std_dev
                threshold_down = mean - STANDARD_DEVIATION * std_dev
                
                for value_pair in all_values:
                    numeric_value = float(value_pair[1])
                    # A condição para enviar um alerta é quando o valor de uma métrica excede os limites
                    # calculados (limite superior ou inferior) com base na média e no desvio padrão.
                    if numeric_value > threshold_up or numeric_value < threshold_down:
                        print(f"\nValor {numeric_value} do timestamp {value_pair[0]} excedeu os limites para a métrica {metric}. Enviando alerta...")
                        send_alert(
                            alertname=metric,
                            summary=f"Alerta de {metric}",
                            description=f"O valor {numeric_value} para {metric} excedeu os limites. Limite superior: {threshold_up}, Limite inferior: {threshold_down}. Timestamp: {value_pair[0]}"
                        )
            else:
                print(f"Nenhum dado válido encontrado para a métrica {metric}. Não será possível calcular o desvio padrão.")
        else:
            print(f"Nenhum dado encontrado para a métrica {metric}. Não será possível calcular o desvio padrão.")

if __name__ == "__main__":
    main()
