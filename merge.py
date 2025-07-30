"""
merge.py

A script to merge or process a model with specified CPU and thread settings.

Usage:
    python merge.py --modelname llama3 --cpu r9 --threadnum 16
"""
import os
import argparse
import csv
import re

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Merge or process a model with specified CPU and thread settings."
    )
    parser.add_argument(
        '--modelname',
        required=True,
        type=str,
        help='Name of the model to merge (e.g., llama3)'
    )
    parser.add_argument(
        '--cpu',
        required=True,
        type=str,
        help='CPU identifier to use (e.g., r9)'
    )
    parser.add_argument(
        '--threadnum',
        required=True,
        type=int,
        help='Number of threads to allocate (e.g., 16)'
    )
    return parser.parse_args()


def extract_metrics(file_path):
    """Extracts Med, Avg, and Max GFLOPS from the text file."""
    med = avg = max = None
    # Regex to match patterns like 'Med (199 GFLOPS)'
    med_pattern = re.compile(r"Med.*\((\d+)\sGFLOPS\)")
    avg_pattern = re.compile(r"Avg.*\((\d+)\sGFLOPS\)")
    max_pattern = re.compile(r"Max.*\((\d+)\sGFLOPS\)")
    
    with open(file_path, 'r') as f:
        for line in f:
            if med is None:
                m = med_pattern.search(line)
                if m:
                    med = int(m.group(1))
            if max is None:
                m = max_pattern.search(line)
                if m:
                    max = int(m.group(1))
            if avg is None:
                m = avg_pattern.search(line)
                if m:
                    avg = int(m.group(1))
    
    return med, max, avg


def main():
    args = parse_arguments()
    modelname = args.modelname
    cpu = args.cpu
    threadnum = args.threadnum

    # Directory path
    folder_path = os.path.join(modelname, cpu, str(threadnum))
    if not os.path.isdir(folder_path):
        print(f"Error: Directory '{folder_path}' does not exist.")
        return

    # Prepare CSV file to store results
    csv_filename = f"{folder_path}/merge_{cpu}_{threadnum}.csv"
    fieldnames = ['Filename', 'Med', 'Max', 'Avg']

    with open(csv_filename, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Traverse the folder and process all text files
        for filename in os.listdir(folder_path):
            if filename[0].isdigit():
                file_path = os.path.join(folder_path, filename)

                # Extract Med, Avg, and Max values
                med, max, avg = extract_metrics(file_path)
                #print(f"med={med} max={max} avg={avg}")

                if med is not None and avg is not None and max is not None:
                    writer.writerow({
                        'Filename': filename,
                        'Med': med,
                        'Max': max,
                        'Avg': avg
                    })
                else:
                    print(f"Warning: Could not extract all metrics from {filename}")

    print(f"Data merged into {csv_filename} successfully.")

if __name__ == '__main__':
    main()
