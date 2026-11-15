#!/usr/bin/env python3
"""
Compare Model Evaluations
=========================
Creates a comparison table of dialogue evaluations across different models.
"""

import json
import os
import pandas as pd

# Model evaluation file paths
MODEL_PATHS = {
    "qwen-turbo": "qwen-turbo/dialogue_evaluations_llm_judge/kinds_scores_summary.json",
    "llama3": "llama3/dialogue_evaluations_llm_judge/kinds_scores_summary.json",
    "gpt-4o": "gpt-4o/dialogue_evaluations_llm_judge/kinds_scores_summary.json",
}

# Kind labels
KIND_LABELS = {
    "1": "Dual Highest Metric",
    "2": "Dual Random",
    "3": "LLM Random",
    "4": "LLM Highest Metric",
    "5": "Two-Agent",
    "6": "Two-Agent Simulate",
}

METRICS = ["fluency", "accuracy", "realism", "coverage", "helpfulness", "overall"]


def load_model_data():
    """Load evaluation data for all models"""
    model_data = {}
    for model_name, filepath in MODEL_PATHS.items():
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    model_data[model_name] = json.load(f)
                print(f"✅ Loaded {model_name}")
            except Exception as e:
                print(f"❌ Error loading {model_name}: {e}")
        else:
            print(f"⚠️  {model_name}: File not found at {filepath}")
    return model_data


def create_comparison_table(model_data):
    """Create a comparison table across models"""
    rows = []
    
    for kind in ["1", "2", "3", "4", "5", "6"]:
        kind_label = KIND_LABELS.get(kind, f"Kind {kind}")
        
        for metric in METRICS:
            row = {
                "Kind": kind_label,
                "Metric": metric.capitalize(),
            }
            
            # Get scores for each model
            for model_name in ["qwen-turbo", "llama3", "gpt-4o"]:
                if model_name in model_data and kind in model_data[model_name]:
                    avg = model_data[model_name][kind].get("averages", {}).get(metric, None)
                    row[model_name] = f"{avg:.2f}" if avg is not None else "N/A"
                else:
                    row[model_name] = "N/A"
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def create_summary_table(model_data):
    """Create a summary table with overall averages per model"""
    summary_rows = []
    
    for metric in METRICS:
        row = {"Metric": metric.capitalize()}
        
        for model_name in ["qwen-turbo", "llama3", "gpt-4o"]:
            if model_name in model_data:
                scores = []
                for kind in ["1", "2", "3", "4", "5", "6"]:
                    if kind in model_data[model_name]:
                        avg = model_data[model_name][kind].get("averages", {}).get(metric)
                        if avg is not None:
                            scores.append(avg)
                
                if scores:
                    row[model_name] = f"{sum(scores) / len(scores):.2f}"
                else:
                    row[model_name] = "N/A"
            else:
                row[model_name] = "N/A"
        
        summary_rows.append(row)
    
    return pd.DataFrame(summary_rows)


def create_kind_comparison_table(model_data):
    """Create a table comparing each kind across models"""
    rows = []
    
    for kind in ["1", "2", "3", "4", "5", "6"]:
        kind_label = KIND_LABELS.get(kind, f"Kind {kind}")
        row = {"Kind": kind_label}
        
        for model_name in ["qwen-turbo", "llama3", "gpt-4o"]:
            if model_name in model_data and kind in model_data[model_name]:
                overall = model_data[model_name][kind].get("averages", {}).get("overall")
                row[model_name] = f"{overall:.2f}" if overall is not None else "N/A"
            else:
                row[model_name] = "N/A"
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("Model Evaluation Comparison")
    print("=" * 70)
    print()
    
    # Load data
    model_data = load_model_data()
    
    if not model_data:
        print("❌ No evaluation data found!")
        return
    
    print()
    
    # Create tables
    print("Creating comparison tables...")
    
    # 1. Detailed comparison (all metrics for all kinds)
    detailed_df = create_comparison_table(model_data)
    detailed_csv = "model_evaluation_comparison_detailed.csv"
    detailed_df.to_csv(detailed_csv, index=False)
    print(f"✅ Saved detailed comparison to {detailed_csv}")
    
    # 2. Summary (average across all kinds per model)
    summary_df = create_summary_table(model_data)
    summary_csv = "model_evaluation_comparison_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"✅ Saved summary comparison to {summary_csv}")
    
    # 3. Kind comparison (overall score per kind per model)
    kind_df = create_kind_comparison_table(model_data)
    kind_csv = "model_evaluation_comparison_by_kind.csv"
    kind_df.to_csv(kind_csv, index=False)
    print(f"✅ Saved kind comparison to {kind_csv}")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Average Scores Across All Kinds")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("COMPARISON: Overall Score by Kind")
    print("=" * 70)
    print(kind_df.to_string(index=False))
    
    # Create LaTeX table
    print("\n" + "=" * 70)
    print("LaTeX Table Format")
    print("=" * 70)
    print()
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("Metric & qwen-turbo & llama3 & gpt-4o \\\\")
    print("\\midrule")
    for _, row in summary_df.iterrows():
        metric = row['Metric']
        qwen = row.get('qwen-turbo', 'N/A')
        llama = row.get('llama3', 'N/A')
        gpt = row.get('gpt-4o', 'N/A')
        print(f"{metric} & {qwen} & {llama} & {gpt} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Average evaluation scores across all dialogue generation methods (kinds 1-6) for different LLM models.}")
    print("\\label{tab:model-comparison}")
    print("\\end{table}")


if __name__ == "__main__":
    main()

