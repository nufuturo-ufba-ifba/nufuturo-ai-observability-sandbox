#!/usr/bin/env python3
import sys
import re
import os
from datetime import datetime

def filter_important_logs(log_file_path, output_file_path):
    """
    Reads a log file, groups lines into blocks (based on timestamp)
    and does two things:
    1. Prints the important blocks to the terminal.
    2. Saves the same blocks to a new file.
    """
    
    # Regex to REMOVE ANSI color codes
    ansi_stripper = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    
    # Regex to DETECT the start of a block
    # Accepts: 
    # 1. ISO Timestamps: 2025-08-13T...
    # 2. Flutter test lines: 00:00 +0...
    block_start_regex = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[\.,]\d+Z|^\d{2}:\d{2} \+\d+')

    # Keywords that mark a block as "important"
    IMPORTANT_KEYWORDS = [
        'clojure.lang.ExceptionInfo',   # Clojure errors/exceptions
        'ERROR',                        # Generic errors (now catches colored ones)
        'Error:',                       # Dart/Flutter errors (uppercase)
        'Some tests failed.',           # Test failure summary
        'java.lang.ArithmeticException',# Other Java exceptions
        'Compilation failed',           # Flutter compilation error
        'FAIL in',                      # Clojure test failure
        'Tests failed.'                 # Clojure test failure summary
    ]

    current_block_lines = []
    current_block_is_important = False
    lines_written = 0

    try:
        with open(log_file_path, 'r', encoding='utf-8') as rf, \
             open(output_file_path, 'w', encoding='utf-8') as wf:
            
            for line in rf:
                clean_line = ansi_stripper.sub('', line)
                
                if block_start_regex.match(clean_line.strip()):
                    if current_block_is_important:
                        for block_line in current_block_lines:
                            wf.write(block_line)
                            print(block_line, end='')
                            lines_written += 1
                    
                    current_block_lines = [line]
                    
                    if any(keyword in clean_line for keyword in IMPORTANT_KEYWORDS):
                        current_block_is_important = True
                    else:
                        current_block_is_important = False
                else:
                    if current_block_lines:
                        current_block_lines.append(line)
                        
                        if not current_block_is_important and any(keyword in clean_line for keyword in IMPORTANT_KEYWORDS):
                            current_block_is_important = True

            if current_block_is_important:
                for block_line in current_block_lines:
                    wf.write(block_line)
                    print(block_line, end='')
                    lines_written += 1
        
        return lines_written

    except FileNotFoundError:
        print(f"\nError: File not found at '{log_file_path}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 main.py <path_to_log_file>", file=sys.stderr)
        sys.exit(1)
        
    input_log_file = sys.argv[1]
    
    base_name = os.path.splitext(os.path.basename(input_log_file))[0]
    
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    
    output_log_file = f"{base_name}-filtrado-{timestamp}.txt"
    
    total_lines = filter_important_logs(input_log_file, output_log_file)
    
    if total_lines > 0:
        print(f"\n\nSuccess! Filtered logs also saved to:\n{output_log_file}")
    else:
        print(f"\nAnalysis complete. No errors found in '{input_log_file}'.")
        os.remove(output_log_file)
