#!/usr/bin/env python3

import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import requests
from dotenv import load_dotenv
import os

load_dotenv()

USE_IA = os.getenv('USE_IA', 'True').lower() == 'true'
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'deepseek-coder')
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_TIMEOUT = int(os.getenv('OLLAMA_TIMEOUT', '90'))


class NubankJsonLogAnalyzer:

    def __init__(self, ollama_url: str = None):
        self.ollama_url = ollama_url or OLLAMA_URL
        self.use_ia = USE_IA


        # Aqui é onde a magia acontece - definimos as chaves e valores que indicam erros no arquivo JSON
        # Essas listas podem ser expandidas conforme necessario
        self.ERROR_LEVEL_KEYS = {'level', 'severity', 'loglevel', 'log_level', 'type'}
        self.ERROR_LEVEL_VALUES = {'error', 'err', 'fatal', 'critical', 'failure', 'exception'}
        self.MESSAGE_KEYS = {'message', 'msg', 'error', 'description', 'value', 'title', 'detail'}
        self.STACKTRACE_KEYS = {'stacktrace', 'stack_trace', 'trace', 'raw_stacktrace', 'exception'}

    def _extract_error_type(self, context: Dict) -> Optional[str]:
        try:
            error_type = context.get('exception', {}).get('values', [{}])[0].get('type')
            if error_type:
                return error_type
        except (IndexError, AttributeError):
            pass
        
        error_node = context.get('error', {})
        if isinstance(error_node, dict):
            return error_node.get('name') or error_node.get('type')
        
        return None

    def _find_error_indicators(self, log_obj: Dict) -> Optional[Dict]:
        if not isinstance(log_obj, dict):
            return None

        found_indicators = {
            'is_error': False,
            'message': None,
            'stacktrace': None,
            'context': log_obj
        }

        queue = [log_obj]
        
        while queue:
            current_obj = queue.pop(0)
            if not isinstance(current_obj, dict):
                continue

            for key, value in current_obj.items():
                key_lower = str(key).lower()

                if key_lower in self.ERROR_LEVEL_KEYS and isinstance(value, str) and value.lower() in self.ERROR_LEVEL_VALUES:
                    found_indicators['is_error'] = True

                if key_lower in self.STACKTRACE_KEYS:
                    found_indicators['is_error'] = True
                    if not found_indicators.get('stacktrace'):
                        found_indicators['stacktrace'] = value

                if key_lower in self.MESSAGE_KEYS and isinstance(value, str) and not found_indicators.get('message'):
                    found_indicators['message'] = value
                
                if isinstance(value, dict):
                    queue.append(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            queue.append(item)
        
        if found_indicators['is_error']:
            if not found_indicators['message'] and 'exception' in log_obj:
                try:
                    found_indicators['message'] = log_obj['exception']['values'][0]['value']
                except (KeyError, IndexError, TypeError):
                    pass
            return found_indicators
            
        return None

    def parse_log_file(self, file_path: str) -> Dict:
        print(f"\nAnalisando arquivo JSON: {file_path}")
        stats = {'total_objects': 0, 'critical_events': []}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Erro Critico: O arquivo '{file_path}' nao eh um JSON valido. Detalhe: {e}")
            return stats
        except Exception as e:
            print(f"Erro Critico ao ler o arquivo: {e}")
            return stats

        log_entries = data if isinstance(data, list) else [data]
        stats['total_objects'] = len(log_entries)

        for i, entry in enumerate(log_entries):
            error_info = self._find_error_indicators(entry)
            if error_info:
                stats['critical_events'].append({'entry_number': i + 1, **error_info})
        
        return stats

    def analyze_with_ollama(self, error_event: Dict) -> str:
        context_str = json.dumps(error_event['context'], indent=2, ensure_ascii=False)
        prompt_parts = [
            "Voce eh um engenheiro SRE especialista em analise de logs de sistemas distribuidos.",
            "\nAnalise o seguinte log de erro em formato JSON:\n",
            "CONTEXTO DO ERRO (JSON):",
            "```json",
            f"{context_str[:3000]}",
            "```",
            "(JSON acima pode estar truncado)\n",
            "INFORMACOES EXTRAIDAS AUTOMATICAMENTE:",
            f"- Mensagem Principal: {error_event.get('message') or 'Nao encontrada'}",
            f"- Stacktrace: {'Encontrado' if error_event.get('stacktrace') else 'Nao encontrado'}\n",
            "Por favor, forneca uma analise tecnica e concisa:",
            "1.  **Resumo do Erro:** O que aconteceu, em 1-2 frases.",
            "2.  **Causa Raiz Provavel:** Qual a falha de programacao ou infraestrutura mais provavel?",
            "3.  **Sugestoes de Correcao:** 2-3 acoes praticas e diretas para resolver o problema."
        ]
        prompt = "\n".join(prompt_parts)
        
        try:
            print("  Consultando a IA para analise... (pode levar um momento)")
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True},
                timeout=OLLAMA_TIMEOUT,
                stream=True
            )
            response.raise_for_status()
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if 'response' in chunk:
                            text = chunk['response']
                            print(text, end='', flush=True)
                            full_response += text
                    except json.JSONDecodeError:
                        continue
            
            print()
            return full_response.strip()
        except requests.exceptions.Timeout:
            return f"Erro: Atingido o tempo limite de {OLLAMA_TIMEOUT}s ao tentar conectar com Ollama."
        except requests.exceptions.RequestException as e:
            return f"Erro: Nao foi possivel conectar com o servidor Ollama em '{self.ollama_url}'. Verifique se ele esta rodando. Detalhe: {e}"
        except Exception as e:
            return f"Erro inesperado durante a analise da IA: {e}"

    def generate_report(self, stats: Dict):
        print("\n" + "="*70)
        print("RELATORIO DE ANALISE DE LOGS JSON")
        if self.use_ia: print(f"(Analise com IA: {'ATIVADA'})")
        print("="*70)

        critical_events = stats['critical_events']
        
        print(f"\nEstatisticas Gerais:")
        print(f"  Total de objetos JSON analisados: {stats['total_objects']}")
        print(f"  Eventos criticos encontrados: {len(critical_events)}")
        
        if not critical_events:
            print("\nNenhum evento critico encontrado com base nas regras de deteccao!")
            return

        print(f"\nDETALHES DOS EVENTOS CRITICOS ENCONTRADOS:")
        
        for idx, event in enumerate(critical_events, 1):
            print("\n" + "-"*70)
            print(f"EVENTO #{idx} (a partir do objeto JSON numero {event['entry_number']})")
            
            print("\n  [ O ERRO REAL ]")
            error_type = self._extract_error_type(event['context'])
            if error_type:
                print(f"  - Tipo de Erro: {error_type}")
            print(f"  - Mensagem: {event.get('message') or 'Nao foi possivel extrair uma mensagem clara.'}")
            
            if self.use_ia:
                print("\n  [ ANALISE DA IA ]")
                diagnosis = self.analyze_with_ollama(event)
                print("")
        print("\n" + "="*70)

def main():
    parser = argparse.ArgumentParser(description="Analisador Inteligente de Logs JSON - Nubank")
    parser.add_argument('file', type=str, help='Caminho do arquivo de log no formato JSON.')
    parser.add_argument('--ollama-url', type=str, default=OLLAMA_URL, help='URL do servidor Ollama.')
    parser.add_argument('--model', type=str, default=OLLAMA_MODEL, help='Nome do modelo Ollama.')
    
    args = parser.parse_args()
    
    if not Path(args.file).exists():
        print(f"Erro: Arquivo nao encontrado em '{args.file}'")
        return
    
    print(f"\n{'Analise com IA ATIVADA' if USE_IA else 'Analise com IA DESATIVADA'}")
    print(f"Modelo: {args.model}")
    print(f"URL Ollama: {args.ollama_url}")
    analyzer = NubankJsonLogAnalyzer(ollama_url=args.ollama_url)
    stats = analyzer.parse_log_file(args.file)
    
    if stats.get('total_objects', 0) > 0:
        analyzer.generate_report(stats)

if __name__ == "__main__":
    main()