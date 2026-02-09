
import pickle
import pandas as pd
import os

# Define file paths
INPUT_FILE = "manual_labeled_data.pkl"
OUTPUT_FILE = "final_labeled_report.csv"

def generate_report():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, 'rb') as f:
        labeled_data = pickle.load(f)

    if not labeled_data:
        print("No labeled data found.")
        return

    print(f"Processing {len(labeled_data)} records...")

    report_data = []

    for item in labeled_data:
        user_id = item.get('user_id')
        user_true_loc = item.get('user_true_loc')
        
        toponyms = item.get('toponyms', [])
        for topo in toponyms:
            word = topo.get('word')
            human_label_idx = topo.get('human_label_idx')
            candidates = topo.get('candidates', [])

            # Data to be included in the report
            row = {
                'user_id': user_id,
                'user_true_lat': user_true_loc[0] if user_true_loc else None,
                'user_true_lon': user_true_loc[1] if user_true_loc else None,
                'toponym': word,
                'human_label_idx': human_label_idx,
                'selected_name': None,
                'selected_state': None,
                'selected_lat': None,
                'selected_lon': None,
                'candidate_couunt': len(candidates)
            }

            if human_label_idx is not None and 0 <= human_label_idx < len(candidates):
                selected_candidate = candidates[human_label_idx]
                row['selected_name'] = selected_candidate.get('name')
                row['selected_state'] = selected_candidate.get('state')
                row['selected_lat'] = selected_candidate.get('lat')
                row['selected_lon'] = selected_candidate.get('lon')
            elif human_label_idx == -1: # Example case for 'None of the above' or similar if implemented
                 row['selected_name'] = "NONE"

            report_data.append(row)

    if not report_data:
        print("No report data generated.")
        return

    df = pd.DataFrame(report_data)
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"Successfully saved report to {OUTPUT_FILE}")
    print(df.head())

if __name__ == "__main__":
    generate_report()