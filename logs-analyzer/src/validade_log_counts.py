import pandas as pd
import json
import re
import os
import sys

def detect_separator(line):
    if ';' in line and line.count(';') > line.count(','):
        return ';'
    return ','

def extract_first_uuid(text):
    if not isinstance(text, str):
        return None
    match = re.search(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', text, re.IGNORECASE)
    return match.group(0) if match else None

def find_key_recursive(data, targets):
    if isinstance(data, dict):
        for k, v in data.items():
            if k in targets and pd.notna(v) and str(v).strip() != "":
                return str(v)
        for v in data.values():
            res = find_key_recursive(v, targets)
            if res: return res
    elif isinstance(data, list):
        for item in data:
            res = find_key_recursive(item, targets)
            if res: return res
    return None

def normalize_user_id(row):
    user_keys = ['user-id', 'userId', 'user_id', 'sub', 'Subject', 'User', 'id', 'customer_id', 'customerId', 'client_id', 'client-id', 'clientId', 'uid', 'uuid']
    row_dict = {k: v for k, v in row.items() if pd.notna(v)}
    for field in user_keys:
        if field in row_dict: return str(row_dict[field])
    
    parsed_objects = []
    for k, v in row_dict.items():
        if isinstance(v, str) and v.strip().startswith('{'):
            try:
                parsed = json.loads(v)
                parsed_objects.append(parsed)
                res = find_key_recursive(parsed, user_keys)
                if res: return res
            except: pass
        elif isinstance(v, (dict, list)):
            parsed_objects.append(v)
            res = find_key_recursive(v, user_keys)
            if res: return res
            
    path_keys = ['path', 'url', 'uri', 'location', 'request_uri']
    for k in path_keys:
        val = None
        if k in row_dict: val = row_dict[k]
        elif 'log' in row_dict and isinstance(row_dict['log'], dict) and k in row_dict['log']: val = row_dict['log'][k]
        if val and isinstance(val, str):
            uuid = extract_first_uuid(val)
            if uuid: return uuid

    cid_keys = ['cid', 'correlation_id', 'correlationId', 'request_id', 'requestId', 'trace_id']
    for k in cid_keys:
        res = find_key_recursive(row_dict, [k])
        if res: return res
    return None

def process_files(file_list):
    all_data = []
    print(f"Processing {len(file_list)} files...")
    for file_path in file_list:
        try:
            filename = os.path.basename(file_path)
            df_temp = None
            if filename.lower().endswith('.csv'):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        first_line = f.readline().strip()
                        sep = detect_separator(first_line)
                    df_temp = pd.read_csv(file_path, sep=sep, on_bad_lines='skip')
                except Exception as e:
                    print(f"Error reading CSV {filename}: {e}")
            elif filename.lower().endswith('.json'):
                try:
                    with open(file_path, 'r') as f:
                        content = json.load(f)
                    if isinstance(content, list): df_temp = pd.DataFrame(content)
                    elif isinstance(content, dict): df_temp = pd.DataFrame([content])
                except Exception as e:
                    try:
                        df_temp = pd.read_json(file_path, lines=True)
                    except Exception as e2:
                        print(f"Error reading JSON {filename}: {e2}")

            if df_temp is not None:
                df_temp['source_file'] = filename
                df_temp['normalized_user_id'] = df_temp.apply(normalize_user_id, axis=1)
                all_data.append(df_temp)
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

    if not all_data: return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)

def main():
    base_dir = '/home/data/Projects/nufuturo-ai-observability-sandbox/logs-analyzer/alexandriaLogs'
    
    # Get all CSVs
    csv_files = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith('.csv')]
    # Get all JSONs
    json_files = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith('.json')]
    
    print("\n--- CSV ANALYSIS ---")
    df_csv = process_files(csv_files)
    if not df_csv.empty:
        print(f"Files Processed: {df_csv['source_file'].nunique()}")
        print(f"Total Events: {len(df_csv)}")
        print(f"identified Users: {df_csv['normalized_user_id'].nunique()}")
    else:
        print("No CSV data found.")

    print("\n--- JSON ANALYSIS ---")
    df_json = process_files(json_files)
    if not df_json.empty:
        print(f"Files Processed: {df_json['source_file'].nunique()}")
        print(f"Total Events: {len(df_json)}")
        print(f"identified Users: {df_json['normalized_user_id'].nunique()}")
    else:
        print("No JSON data found.")

if __name__ == "__main__":
    main()
