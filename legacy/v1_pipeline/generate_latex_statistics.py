#!/usr/bin/env python3
"""
Generate LaTeX tables from dialogue statistics
"""

import json
import os
from pathlib import Path
from collections import Counter

def generate_latex_tables(directory: str = "full_dialogues_dataset"):
    """Generate LaTeX tables from dialogue statistics"""
    
    # Load statistics
    stats_file = os.path.join(directory, "dialogue_statistics.json")
    if not os.path.exists(stats_file):
        print(f"Statistics file not found: {stats_file}")
        return
    
    with open(stats_file, 'r', encoding='utf-8') as f:
        stats = json.load(f)
    
    # Get all dialogue files for detailed analysis
    dir_path = Path(directory)
    dialogue_files = sorted(
        [f for f in dir_path.glob("dialogue_*.json")],
        key=lambda x: int(x.stem.split('_')[1]) if x.stem.split('_')[1].isdigit() else 0
    )
    
    # Calculate additional statistics
    total_dialogues = stats['summary']['total_dialogues']
    total_rounds = stats['summary']['total_rounds']
    avg_rounds = stats['summary']['avg_rounds']
    
    dialogue_rounds_distribution = []
    features_per_dialogue = []
    products_with_price = 0
    products_without_price = 0
    similar_products_count = []
    
    for dialogue_file in dialogue_files:
        try:
            with open(dialogue_file, 'r', encoding='utf-8') as f:
                dialogue = json.load(f)
            
            dialogue_rounds_distribution.append(dialogue.get('dialogue_rounds', 0))
            features_per_dialogue.append(len(dialogue.get('features_discussed', [])))
            
            price = dialogue.get('price')
            if price is not None and price != 0:
                products_with_price += 1
            else:
                products_without_price += 1
            
            similar_products = dialogue.get('similar_products', [])
            if similar_products:
                similar_products_count.append(len(similar_products))
                
        except Exception as e:
            continue
    
    rounds_counter = Counter(dialogue_rounds_distribution)
    avg_features = sum(features_per_dialogue) / len(features_per_dialogue) if features_per_dialogue else 0
    avg_similar = sum(similar_products_count) / len(similar_products_count) if similar_products_count else 0
    
    # Filter out 0 values for min calculation
    valid_rounds = [r for r in dialogue_rounds_distribution if r > 0]
    min_rounds = min(valid_rounds) if valid_rounds else 0
    max_rounds = max(dialogue_rounds_distribution) if dialogue_rounds_distribution else 0
    
    # Generate LaTeX content
    latex_content = """\\documentclass[12pt,a4paper]{article}
\\usepackage[utf8]{inputenc}
\\usepackage{booktabs}
\\usepackage{multirow}
\\usepackage{array}
\\usepackage{xcolor}
\\usepackage{geometry}
\\geometry{margin=1in}

\\title{Generated Dialogues Statistics}
\\author{Dialogue Data Analysis}
\\date{\\today}

\\begin{document}

\\maketitle

\\section{Overall Statistics}
\\begin{table}[h]
\\centering
\\caption{Overall Statistics of Generated Dialogues}
\\begin{tabular}{|l|r|}
\\hline
\\textbf{Metric} & \\textbf{Value} \\\\
\\hline
Total Dialogues & """ + str(total_dialogues) + """ \\\\
\\hline
Total Dialogue Rounds & """ + str(total_rounds) + """ \\\\
\\hline
Average Dialogue Rounds & """ + f"{avg_rounds:.2f}" + """ \\\\
\\hline
Minimum Dialogue Rounds & """ + str(min_rounds) + """ \\\\
\\hline
Maximum Dialogue Rounds & """ + str(max_rounds) + """ \\\\
\\hline
Average Features Discussed & """ + f"{avg_features:.2f}" + """ \\\\
\\hline
Products with Price & """ + str(products_with_price) + """ (""" + f"{products_with_price/total_dialogues*100:.1f}" + """\\%) \\\\
\\hline
Products without Price & """ + str(products_without_price) + """ (""" + f"{products_without_price/total_dialogues*100:.1f}" + """\\%) \\\\
\\hline
Average Similar Products & """ + f"{avg_similar:.2f}" + """ \\\\
\\hline
\\end{tabular}
\\end{table}

\\section{Category Distribution}
\\begin{table}[h]
\\centering
\\caption{Dialogue Distribution by Category}
\\begin{tabular}{|l|r|r|}
\\hline
\\textbf{Category} & \\textbf{Count} & \\textbf{Percentage} \\\\
\\hline
"""
    
    # Add categories
    categories = stats['categories']
    sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
    for category, count in sorted_categories:
        percentage = (count / total_dialogues) * 100
        latex_content += f"{category} & {count} & {percentage:.1f}\\% \\\\\n\\hline\n"
    
    latex_content += """\\end{tabular}
\\end{table}

\\section{Brand Distribution}
\\begin{table}[h]
\\centering
\\caption{Dialogue Distribution by Brand (Top 10)}
\\begin{tabular}{|l|r|r|}
\\hline
\\textbf{Brand} & \\textbf{Count} & \\textbf{Percentage} \\\\
\\hline
"""
    
    # Add top brands
    brands = stats['brands']
    sorted_brands = sorted(brands.items(), key=lambda x: x[1], reverse=True)[:10]
    for brand, count in sorted_brands:
        percentage = (count / total_dialogues) * 100
        latex_content += f"{brand} & {count} & {percentage:.1f}\\% \\\\\n\\hline\n"
    
    latex_content += """\\end{tabular}
\\end{table}

\\section{Dialogue Rounds Distribution}
\\begin{table}[h]
\\centering
\\caption{Distribution of Dialogue Rounds}
\\begin{tabular}{|c|r|r|}
\\hline
\\textbf{Number of Rounds} & \\textbf{Number of Dialogues} & \\textbf{Percentage} \\\\
\\hline
"""
    
    # Add rounds distribution
    for rounds in sorted(rounds_counter.keys()):
        count = rounds_counter[rounds]
        percentage = (count / total_dialogues) * 100
        latex_content += f"{rounds} & {count} & {percentage:.1f}\\% \\\\\n\\hline\n"
    
    latex_content += """\\end{tabular}
\\end{table}

\\section{Cluster Distribution}
\\begin{table}[h]
\\centering
\\caption{Dialogue Distribution by Cluster (Top 10)}
\\begin{tabular}{|l|r|r|}
\\hline
\\textbf{Cluster} & \\textbf{Count} & \\textbf{Percentage} \\\\
\\hline
"""
    
    # Add top clusters
    clusters = stats['clusters']
    sorted_clusters = sorted(clusters.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0)
    sorted_clusters = sorted(sorted_clusters, key=lambda x: x[1], reverse=True)[:10]
    for cluster, count in sorted_clusters:
        percentage = (count / total_dialogues) * 100
        latex_content += f"Cluster {cluster} & {count} & {percentage:.1f}\\% \\\\\n\\hline\n"
    
    latex_content += """\\end{tabular}
\\end{table}

\\end{document}
"""
    
    # Save LaTeX file
    output_file = os.path.join(directory, "dialogue_statistics.tex")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    print(f"✅ LaTeX file generated: {output_file}")
    
    # Also create a simpler table-only version
    simple_latex = """\\begin{table}[h]
\\centering
\\caption{Generated Dialogues Statistics}
\\begin{tabular}{|l|r|}
\\hline
\\textbf{Metric} & \\textbf{Value} \\\\
\\hline
Total Dialogues & """ + str(total_dialogues) + """ \\\\
\\hline
Total Dialogue Rounds & """ + str(total_rounds) + """ \\\\
\\hline
Average Dialogue Rounds & """ + f"{avg_rounds:.2f}" + """ \\\\
\\hline
Minimum Dialogue Rounds & """ + str(min_rounds) + """ \\\\
\\hline
Maximum Dialogue Rounds & """ + str(max_rounds) + """ \\\\
\\hline
Average Features Discussed & """ + f"{avg_features:.2f}" + """ \\\\
\\hline
Products with Price & """ + str(products_with_price) + """ (""" + f"{products_with_price/total_dialogues*100:.1f}" + """\\%) \\\\
\\hline
Products without Price & """ + str(products_without_price) + """ (""" + f"{products_without_price/total_dialogues*100:.1f}" + """\\%) \\\\
\\hline
Average Similar Products & """ + f"{avg_similar:.2f}" + """ \\\\
\\hline
\\end{tabular}
\\end{table}

\\begin{table}[h]
\\centering
\\caption{Category Distribution}
\\begin{tabular}{|l|r|r|}
\\hline
\\textbf{Category} & \\textbf{Count} & \\textbf{Percentage} \\\\
\\hline
"""
    
    for category, count in sorted_categories:
        percentage = (count / total_dialogues) * 100
        simple_latex += f"{category} & {count} & {percentage:.1f}\\% \\\\\n\\hline\n"
    
    simple_latex += """\\end{tabular}
\\end{table}

\\begin{table}[h]
\\centering
\\caption{Dialogue Rounds Distribution}
\\begin{tabular}{|c|r|r|}
\\hline
\\textbf{Number of Rounds} & \\textbf{Number of Dialogues} & \\textbf{Percentage} \\\\
\\hline
"""
    
    for rounds in sorted(rounds_counter.keys()):
        count = rounds_counter[rounds]
        percentage = (count / total_dialogues) * 100
        simple_latex += f"{rounds} & {count} & {percentage:.1f}\\% \\\\\n\\hline\n"
    
    simple_latex += """\\end{tabular}
\\end{table}
"""
    
    simple_output = os.path.join(directory, "dialogue_statistics_tables_only.tex")
    with open(simple_output, 'w', encoding='utf-8') as f:
        f.write(simple_latex)
    
    print(f"✅ Simple tables-only LaTeX file generated: {simple_output}")
    
    return output_file, simple_output


def main():
    """Main function"""
    directory = "full_dialogues_dataset"
    generate_latex_tables(directory)


if __name__ == "__main__":
    main()

