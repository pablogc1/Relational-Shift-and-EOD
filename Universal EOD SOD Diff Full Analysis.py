import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import re
from tqdm import tqdm
from collections import Counter
from matplotlib.collections import LineCollection 
import matplotlib.patches as mpatches

# ==============================================================================
#                      ******** USER CONFIGURATION ********
# ==============================================================================
BASE_RESULTS_DIR = "/Volumes/PortableSSD/OD Results" 
OUTPUT_SUBDIR = "EOD WOD SOD Results" # All tables and plots will be saved here

CORPUS_CONFIG = {
    "ground_filtered": {"folder": "Ground Filtered", "file": "extracted_definitions_ground_filtered.txt"},
    "null_model": {"folder": "Null Model", "file": "extracted_definitions_null_model.txt"},
    "random_removal": {"folder": "Random Removal", "file": "extracted_definitions_random_removal.txt"},
    "targeted_removal": {"folder": "Targeted Removal", "file": "extracted_definitions_targeted_removal.txt"},
    "ai_generated": {"folder": "AI Generated", "file": "extracted_definitions_ai_generated.txt"},
    "merriam_webster": {"folder": "Complex", "file": "extracted_definitions_complex.txt"}
}
# ==============================================================================
#                      Global Settings & File Paths
# ==============================================================================
FULL_OUTPUT_DIR = os.path.join(BASE_RESULTS_DIR, OUTPUT_SUBDIR)

# --- Input Files ---
SUMMARY_REPORT_FILE = os.path.join(FULL_OUTPUT_DIR, "summary_report.txt")
EOD_RESULTS_FILE = os.path.join(FULL_OUTPUT_DIR, "eod_results_final.csv")
LOG_HISTOGRAM_FILE = os.path.join(FULL_OUTPUT_DIR, "log_binned_histogram_data.csv")

# --- Output Files (Tables) ---
PROPERTIES_CSV = os.path.join(FULL_OUTPUT_DIR, "table_corpus_properties.csv")
MASTER_SUMMARY_CSV = os.path.join(FULL_OUTPUT_DIR, "table_master_divergence_summary.csv")
CORPUS_DIVERGENCE_CSV = os.path.join(FULL_OUTPUT_DIR, "table_corpus_divergence_summary.csv") 
BRITTLENESS_REPORT_TXT = os.path.join(FULL_OUTPUT_DIR, "table_word_brittleness_rankings.txt")
GLOBAL_BRITTLENESS_CSV = os.path.join(FULL_OUTPUT_DIR, "table_global_word_brittleness.csv")
LANGUAGE_BRITTLENESS_CSV = os.path.join(FULL_OUTPUT_DIR, "table_language_word_brittleness.csv")


# --- Plotting Settings ---
PLOT_DPI = 300
plt.style.use('seaborn-v0_8-whitegrid')
TOP_N_WORDS = 20
ALL_CORPORA_KEYS = list(CORPUS_CONFIG.keys())

# ==============================================================================
#                        1. Data Loading & Parsing
# ==============================================================================

def load_all_data():
    """Loads and parses all necessary input files for the entire analysis."""
    print("[1/13] Loading and parsing all data sources...")
    data = {}
    
    # --- Parse Summary Report for Diff Metrics ---
    try:
        with open(SUMMARY_REPORT_FILE, 'r') as f: content = f.read()
        sections = re.split(r'-{80}\nComparison: (.+?)\n-{80}', content)
        metrics_data = []
        for i in range(1, len(sections), 2):
            key = sections[i].strip().lower().replace(" ", "_")
            block = sections[i+1]
            def get_float(p, t):
                m = re.search(p, t, re.DOTALL)
                return float(m.group(1).replace(",", "")) if m else np.nan
            current_metrics = {'Comparison': key}
            for metric in ['SOD', 'WOD', 'SOD_TL', 'WOD_TL']:
                 metric_block_match = re.search(fr"--- Metric: {metric} ---\n(.*?)(?=\n\n--- Metric:|\Z)", block, re.DOTALL)
                 if metric_block_match:
                     current_metrics[f'Mean_Abs_{metric}_Diff'] = get_float(r"Absolute Mean:\s+([\d,.-]+)", metric_block_match.group(1))
            metrics_data.append(current_metrics)
        data['diff_summary'] = pd.DataFrame(metrics_data)
        print(f"  -> Successfully parsed '{os.path.basename(SUMMARY_REPORT_FILE)}'")
    except Exception as e:
        print(f"  -> WARNING: Could not parse '{os.path.basename(SUMMARY_REPORT_FILE)}'. Error: {e}")
        data['diff_summary'] = pd.DataFrame()

    # --- Load and RESHAPE EOD Results ---
    try:
        df_eod_wide = pd.read_csv(EOD_RESULTS_FILE)
        id_vars = [c for c in ['master_idx', 'word'] if c in df_eod_wide.columns]
        value_vars = [c for c in df_eod_wide.columns if c not in id_vars]
        df_eod_long = df_eod_wide.melt(id_vars=id_vars, value_vars=value_vars, var_name='Metric', value_name='Value')
        df_eod_long['Comparison'] = df_eod_long['Metric'].str.replace('eod_score_', '').str.replace('eod_tl_', '')
        df_eod_long['Metric_Type'] = df_eod_long['Metric'].apply(lambda x: 'EOD_Score' if 'score' in x else 'EOD_TL')
        df_eod_final = df_eod_long.pivot_table(index=[*id_vars, 'Comparison'], columns='Metric_Type', values='Value').reset_index()
        df_eod_final.rename(columns={'EOD_Score': 'Mean_EOD_Score', 'EOD_TL': 'Mean_EOD_TL'}, inplace=True)
        data['eod_results_long'] = df_eod_final
        print(f"  -> Successfully loaded and reshaped '{os.path.basename(EOD_RESULTS_FILE)}'")
    except Exception as e:
        print(f"  -> WARNING: Could not load '{os.path.basename(EOD_RESULTS_FILE)}'. Error: {e}")
        data['eod_results_long'] = pd.DataFrame()
        
    # --- Load Log Histogram Data ---
    try:
        data['log_histograms'] = pd.read_csv(LOG_HISTOGRAM_FILE)
        print(f"  -> Successfully loaded '{os.path.basename(LOG_HISTOGRAM_FILE)}'")
    except Exception as e:
        print(f"  -> WARNING: Could not load '{os.path.basename(LOG_HISTOGRAM_FILE)}'. Error: {e}")
        data['log_histograms'] = pd.DataFrame()
    return data

# ==============================================================================
#                   2. Core Analysis & Table Generation
# ==============================================================================

def create_master_summary_df(all_data):
    """Creates a single master DataFrame with all summary metrics per comparison."""
    df_diffs = all_data.get('diff_summary', pd.DataFrame())
    df_eod_long = all_data.get('eod_results_long', pd.DataFrame())
    if df_eod_long.empty or df_diffs.empty: return pd.DataFrame()

    eod_summary = df_eod_long[df_eod_long['Mean_EOD_Score'] > -1].groupby('Comparison').agg(
        Mean_EOD_Score=('Mean_EOD_Score', 'mean'), 
        Mean_EOD_TL=('Mean_EOD_TL', 'mean')
    ).reset_index()
    
    master_df = pd.merge(eod_summary, df_diffs, on='Comparison', how='outer')
    master_df.to_csv(MASTER_SUMMARY_CSV, index=False, float_format="%.2f")
    return master_df

def calculate_corpus_properties():
    """Calculates mean definition length and word usage frequency for each corpus."""
    properties = []
    for key, config in CORPUS_CONFIG.items():
        filepath = os.path.join(BASE_RESULTS_DIR, config["folder"], config["file"])
        if not os.path.exists(filepath): continue
        definitions = {head.strip().lower(): def_text.strip().split() for line in open(filepath, 'r') if ':' in line for head, def_text in [line.split(":", 1)]}
        if not definitions: continue
        token_counts = Counter(token for def_tokens in definitions.values() for token in def_tokens)
        properties.append({
            "Corpus": key, 
            "Mean_Def_Length": sum(len(d) for d in definitions.values()) / len(definitions), 
            "Mean_Word_Usage_Freq": sum(token_counts.values()) / len(token_counts) if token_counts else 0
        })
    df_properties = pd.DataFrame(properties)
    return df_properties

def calculate_metric_correlations(master_df):
    """Calculates and prints correlations between primary divergence metrics."""
    if master_df.empty or len(master_df) < 2: return
    df = master_df.copy()
    df['Log10_EOD_Score'] = np.log10(df['Mean_EOD_Score'] + 1)
    df['Log10_Abs_SOD_Diff'] = np.log10(df['Mean_Abs_SOD_Diff'] + 1)
    
    raw_pearson = df['Mean_EOD_Score'].corr(df['Mean_Abs_SOD_Diff'], method='pearson')
    log_pearson = df['Log10_EOD_Score'].corr(df['Log10_Abs_SOD_Diff'], method='pearson')
    spearman_corr = df['Mean_EOD_Score'].corr(df['Mean_Abs_SOD_Diff'], method='spearman')

    print("\n" + "="*80 + "\n      TABLE 2: CORRELATION BETWEEN DIVERGENCE METRICS\n(Semantic Stability vs. Relational Shift)\n" + "-"*80)
    print(f"  -> Pearson on Raw Values (r):          {raw_pearson:.4f}")
    print(f"  -> Pearson on Log10 Values (r_log):    {log_pearson:.4f}")
    print(f"  -> Spearman's Rank Corr. (rho):        {spearman_corr:.4f}\n" + "="*80)

def calculate_corpus_divergence_index(master_df):
    """Calculates the average divergence and a unified rank for each corpus."""
    if master_df.empty: return pd.DataFrame()
    
    index_data = []
    for corpus_key in ALL_CORPORA_KEYS:
        relevant_comparisons = master_df[master_df['Comparison'].str.contains(corpus_key)]
        if relevant_comparisons.empty: continue
        
        avg_metrics = relevant_comparisons[[
            'Mean_EOD_Score', 'Mean_EOD_TL', 
            'Mean_Abs_SOD_Diff', 'Mean_Abs_SOD_TL_Diff'
        ]].mean().to_dict()
        
        avg_metrics['Corpus'] = corpus_key
        index_data.append(avg_metrics)

    df_index = pd.DataFrame(index_data)
    df_index.rename(columns={
        'Mean_EOD_Score': 'Avg EOD Score', 'Mean_EOD_TL': 'Avg EOD TL',
        'Mean_Abs_SOD_Diff': 'Avg SOD Diff', 'Mean_Abs_SOD_TL_Diff': 'Avg SOD TL Diff'
    }, inplace=True)
    
    metrics_to_normalize = ['Avg EOD Score', 'Avg EOD TL', 'Avg SOD Diff', 'Avg SOD TL Diff']
    for metric in metrics_to_normalize:
        min_val = df_index[metric].min()
        max_val = df_index[metric].max()
        df_index[f'{metric} (Norm)'] = (df_index[metric] - min_val) / (max_val - min_val) if (max_val - min_val) != 0 else 0

    df_index['Unified Divergence Rank'] = df_index[[f'{m} (Norm)' for m in metrics_to_normalize]].mean(axis=1)
    df_index = df_index.sort_values(by='Unified Divergence Rank', ascending=False).reset_index(drop=True)
    
    df_save = df_index.copy()
    df_save['Corpus'] = df_save['Corpus'].str.replace("_", " ").str.title().str.replace("Merriam Webster", "Complex")
    df_save.to_csv(CORPUS_DIVERGENCE_CSV, index=False, float_format="%.4f")
    
    print("\n" + "="*80 + "\n              TABLE 3: CORPUS DIVERGENCE INDEX SUMMARY\n" + "-"*80)
    df_print = df_index[['Corpus', 'Avg EOD Score', 'Avg EOD TL', 'Avg SOD Diff', 'Avg SOD TL Diff', 'Unified Divergence Rank']].copy()
    df_print['Corpus'] = df_print['Corpus'].str.replace("_", " ").str.title().str.replace("Merriam Webster", "Complex")
    
    df_print['Avg EOD Score'] = df_print['Avg EOD Score'].apply(lambda x: f'{x:,.0f}')
    df_print['Avg SOD Diff'] = df_print['Avg SOD Diff'].apply(lambda x: f'{x:,.0f}')
    df_print['Avg EOD TL'] = df_print['Avg EOD TL'].apply(lambda x: f'{x:.2f}')
    df_print['Avg SOD TL Diff'] = df_print['Avg SOD TL Diff'].apply(lambda x: f'{x:.2f}')
    df_print['Unified Divergence Rank'] = df_print['Unified Divergence Rank'].apply(lambda x: f'{x:.4f}')

    print(df_print.to_string(index=False))
    print("="*80)
    
    return df_index

def calculate_property_divergence_correlation(df_divergence, df_properties):
    """Calculates correlation between corpus properties and all divergence scores."""
    if df_divergence.empty or df_properties.empty: return
    
    df_prop_mod = df_properties.copy()
    df_prop_mod['Corpus_Key'] = df_prop_mod['Corpus']
    df_prop_mod['Corpus'] = df_prop_mod['Corpus'].str.replace("_", " ").str.title().str.replace("Merriam Webster", "Complex")
    
    df_div_mod = df_divergence.copy()
    df_div_mod['Corpus_Key'] = df_div_mod['Corpus'].str.lower().replace(" ", "_", regex=True).replace("complex", "merriam_webster", regex=True)
    
    merged_df = pd.merge(df_div_mod, df_prop_mod.drop('Corpus', axis=1), on='Corpus_Key')

    correlations = {}
    div_metrics = ['Avg EOD Score', 'Avg EOD TL', 'Avg SOD Diff', 'Avg SOD TL Diff']
    prop_metrics = ['Mean_Def_Length', 'Mean_Word_Usage_Freq']
    
    for div_m in div_metrics:
        for prop_m in prop_metrics:
            key = f"{div_m} vs. {prop_m.replace('_', ' ')}"
            correlations[key] = merged_df[div_m].corr(merged_df[prop_m])

    print("\n" + "="*80 + "\n   TABLE 4: CORRELATION OF DIVERGENCE WITH CORPUS PROPERTIES\n" + "-"*80)
    print(f"{'Correlation Pair':<55} {'Pearson (r)':<20}")
    print(f"{'-'*55} {'-'*20}")
    for key, val in correlations.items():
        print(f"{key:<55} {val: <20.4f}")
    print("="*80)

# ==============================================================================
#           *** UPDATED & NEW FUNCTIONS HERE ***
# ==============================================================================
def analyze_divergence_from_null(master_df):
    """Filters and ranks comparisons against the null model, including a unified rank."""
    if master_df.empty: return
    
    null_comparisons = master_df[master_df['Comparison'].str.contains('null_model')].copy()
    null_comparisons['Corpus'] = null_comparisons['Comparison'].str.replace('_vs_null_model', '').str.replace('null_model_vs_', '')
    
    # --- Calculate Unified Divergence Rank for this subset ---
    metrics_to_normalize = ['Mean_EOD_Score', 'Mean_EOD_TL', 'Mean_Abs_SOD_Diff', 'Mean_Abs_SOD_TL_Diff']
    for metric in metrics_to_normalize:
        min_val = null_comparisons[metric].min()
        max_val = null_comparisons[metric].max()
        null_comparisons[f'{metric} (Norm)'] = (null_comparisons[metric] - min_val) / (max_val - min_val) if (max_val - min_val) != 0 else 0

    null_comparisons['Unified Divergence Rank'] = null_comparisons[[f'{m} (Norm)' for m in metrics_to_normalize]].mean(axis=1)
    null_comparisons = null_comparisons.sort_values(by='Unified Divergence Rank', ascending=False)
    
    print("\n" + "="*80 + "\n              TABLE 5: DIVERGENCE FROM THE NULL MODEL\n" + "-"*80)
    df_print = null_comparisons[['Corpus', 'Mean_EOD_Score', 'Mean_EOD_TL', 'Mean_Abs_SOD_Diff', 'Mean_Abs_SOD_TL_Diff', 'Unified Divergence Rank']].copy()
    df_print['Corpus'] = df_print['Corpus'].str.replace("_", " ").str.title().str.replace("Merriam Webster", "Complex")
    
    df_print['Mean_EOD_Score'] = df_print['Mean_EOD_Score'].apply(lambda x: f'{x:,.0f}')
    df_print['Mean_Abs_SOD_Diff'] = df_print['Mean_Abs_SOD_Diff'].apply(lambda x: f'{x:,.0f}')
    df_print['Mean_EOD_TL'] = df_print['Mean_EOD_TL'].apply(lambda x: f'{x:.2f}')
    df_print['Mean_Abs_SOD_TL_Diff'] = df_print['Mean_Abs_SOD_TL_Diff'].apply(lambda x: f'{x:.2f}')
    df_print['Unified Divergence Rank'] = df_print['Unified Divergence Rank'].apply(lambda x: f'{x:.4f}')
    
    print(df_print.to_string(index=False))
    print("="*80)

def calculate_linguistic_divergence(master_df):
    """Calculates average divergence of each corpus against the linguistic group, including a unified rank."""
    if master_df.empty: return

    LINGUISTIC_CORPORA = [k for k in ALL_CORPORA_KEYS if k != 'null_model']
    divergence_data = []

    for corpus_key in ALL_CORPORA_KEYS:
        target_corpora = [k for k in LINGUISTIC_CORPORA if k != corpus_key]
        if not target_corpora and corpus_key != 'null_model': continue

        # Handle the null model separately: compare it against all linguistic corpora
        if corpus_key == 'null_model':
            target_corpora = LINGUISTIC_CORPORA
        
        relevant_comparisons = master_df[
            master_df['Comparison'].apply(lambda x: 
                (x.startswith(f"{corpus_key}_vs_") and x.split('_vs_')[1] in target_corpora) or
                (x.endswith(f"_vs_{corpus_key}") and x.split('_vs_')[0] in target_corpora)
            )
        ]
        
        if relevant_comparisons.empty: continue
        avg_metrics = relevant_comparisons[['Mean_EOD_Score', 'Mean_EOD_TL', 'Mean_Abs_SOD_Diff', 'Mean_Abs_SOD_TL_Diff']].mean().to_dict()
        avg_metrics['Corpus'] = corpus_key
        divergence_data.append(avg_metrics)

    df_ling_div = pd.DataFrame(divergence_data)

    # --- Calculate Unified Divergence Rank for this subset ---
    metrics_to_normalize = ['Mean_EOD_Score', 'Mean_EOD_TL', 'Mean_Abs_SOD_Diff', 'Mean_Abs_SOD_TL_Diff']
    for metric in metrics_to_normalize:
        min_val = df_ling_div[metric].min()
        max_val = df_ling_div[metric].max()
        df_ling_div[f'{metric} (Norm)'] = (df_ling_div[metric] - min_val) / (max_val - min_val) if (max_val - min_val) != 0 else 0
    
    df_ling_div['Unified Divergence Rank'] = df_ling_div[[f'{m} (Norm)' for m in metrics_to_normalize]].mean(axis=1)
    df_ling_div = df_ling_div.sort_values(by='Unified Divergence Rank', ascending=False).reset_index(drop=True)

    print("\n" + "="*80 + "\n         TABLE 6: DIVERGENCE FROM LINGUISTIC CORPORA GROUP\n" + "-"*80)
    df_print = df_ling_div[['Corpus', 'Mean_EOD_Score', 'Mean_EOD_TL', 'Mean_Abs_SOD_Diff', 'Mean_Abs_SOD_TL_Diff', 'Unified Divergence Rank']].copy()
    df_print['Corpus'] = df_print['Corpus'].str.replace("_", " ").str.title().str.replace("Merriam Webster", "Complex")
    
    df_print['Mean_EOD_Score'] = df_print['Mean_EOD_Score'].apply(lambda x: f'{x:,.0f}')
    df_print['Mean_Abs_SOD_Diff'] = df_print['Mean_Abs_SOD_Diff'].apply(lambda x: f'{x:,.0f}')
    df_print['Mean_EOD_TL'] = df_print['Mean_EOD_TL'].apply(lambda x: f'{x:.2f}')
    df_print['Mean_Abs_SOD_TL_Diff'] = df_print['Mean_Abs_SOD_TL_Diff'].apply(lambda x: f'{x:.2f}')
    df_print['Unified Divergence Rank'] = df_print['Unified Divergence Rank'].apply(lambda x: f'{x:.4f}')

    print(df_print.to_string(index=False))
    print("="*80)


def calculate_global_brittleness(all_data):
    """Aggregates EOD scores to find universally brittle and robust words."""
    df_eod_long = all_data.get('eod_results_long', pd.DataFrame())
    if df_eod_long is None or df_eod_long.empty: return

    valid_scores = df_eod_long[df_eod_long['Mean_EOD_Score'] > -1].copy()
    global_brittleness = valid_scores.groupby('word')['Mean_EOD_Score'].mean().reset_index()
    global_brittleness.rename(columns={'Mean_EOD_Score': 'Average_EOD_Score'}, inplace=True)

    brittle = global_brittleness.sort_values(by='Average_EOD_Score', ascending=False).head(TOP_N_WORDS)
    robust = global_brittleness.sort_values(by='Average_EOD_Score', ascending=True).head(TOP_N_WORDS)

    brittle.rename(columns={'word': 'Most Brittle Word', 'Average_EOD_Score': 'Avg EOD Score (Brittle)'}, inplace=True)
    robust.rename(columns={'word': 'Most Robust Word', 'Average_EOD_Score': 'Avg EOD Score (Robust)'}, inplace=True)
    
    side_by_side = pd.concat([brittle.reset_index(drop=True), robust.reset_index(drop=True)], axis=1)

    side_by_side.to_csv(GLOBAL_BRITTLENESS_CSV, index=False, float_format="%.2f")

    print("\n" + "="*80 + f"\n          TABLE 7: TOP {TOP_N_WORDS} GLOBALLY BRITTLE & ROBUST WORDS\n" + "-"*80)
    print(side_by_side.to_string(index=False, float_format="%.2f"))
    print("="*80)

def calculate_language_brittleness(all_data):
    """Aggregates EOD scores across only language corpora (excluding null model)."""
    df_eod_long = all_data.get('eod_results_long', pd.DataFrame())
    if df_eod_long is None or df_eod_long.empty: return

    language_comparisons = df_eod_long[~df_eod_long['Comparison'].str.contains('null_model')].copy()
    
    valid_scores = language_comparisons[language_comparisons['Mean_EOD_Score'] > -1].copy()
    language_brittleness = valid_scores.groupby('word')['Mean_EOD_Score'].mean().reset_index()
    language_brittleness.rename(columns={'Mean_EOD_Score': 'Language_Avg_EOD_Score'}, inplace=True)

    brittle = language_brittleness.sort_values(by='Language_Avg_EOD_Score', ascending=False).head(TOP_N_WORDS)
    robust = language_brittleness.sort_values(by='Language_Avg_EOD_Score', ascending=True).head(TOP_N_WORDS)

    brittle.rename(columns={'word': 'Most Brittle Word (Language)', 'Language_Avg_EOD_Score': 'Avg EOD Score'}, inplace=True)
    robust.rename(columns={'word': 'Most Robust Word (Language)', 'Language_Avg_EOD_Score': 'Avg EOD Score'}, inplace=True)
    
    side_by_side = pd.concat([brittle.reset_index(drop=True), robust.reset_index(drop=True)], axis=1)

    side_by_side.to_csv(LANGUAGE_BRITTLENESS_CSV, index=False, float_format="%.2f")

    print("\n" + "="*80 + f"\n       TABLE 8: TOP {TOP_N_WORDS} LANGUAGE-ONLY BRITTLE & ROBUST WORDS\n" + "-"*80)
    print(side_by_side.to_string(index=False, float_format="%.2f"))
    print("="*80)

# ==============================================================================
#                   3. Visualization & Reporting
# ==============================================================================

def generate_impact_plots(master_df):
    """Generates all high-level bar chart plots for the report."""
    print(f"\n[{9}/13] Generating Corpus Impact Overview Plots...")
    if master_df.empty: return

    df_plot = master_df.copy()
    df_plot['Log10_Mean_EOD_Score'] = np.log10(df_plot['Mean_EOD_Score'] + 1)
    df_plot['Comparison'] = df_plot['Comparison'].str.replace("_", " ").str.title().str.replace("Merriam Webster", "Complex")
    df_sorted = df_plot.sort_values(by='Mean_EOD_Score', ascending=False)
    
    bar_palette = {cat: color for cat, color in zip(df_sorted['Comparison'].unique(), sns.color_palette("plasma", n_colors=len(df_sorted['Comparison'].unique())))}

    def create_bar_plot(y_col, title, y_label, filename):
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(data=df_sorted, x='Comparison', y=y_col, hue='Comparison', palette=bar_palette, legend=False, dodge=False)
        plt.title(title, fontsize=20, pad=20); plt.xlabel('Corpus Comparison', fontsize=14); plt.ylabel(y_label, fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12); plt.grid(True, axis='y', linestyle='--'); plt.tight_layout()
        for p in ax.patches: ax.annotate(f'{p.get_height():,.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 9), textcoords='offset points')
        plt.savefig(os.path.join(FULL_OUTPUT_DIR, filename), dpi=PLOT_DPI); plt.close()

    create_bar_plot('Mean_EOD_Score', 'Semantic Stability: Mean Eigen-OD Score', 'Mean EOD Score (Higher is Less Stable)', 'plot_impact_mean_eod.png')
    create_bar_plot('Log10_Mean_EOD_Score', 'Semantic Stability: Log-Transformed Mean EOD Score', 'Log10(Mean EOD Score + 1)', 'plot_impact_log_mean_eod.png')
    create_bar_plot('Mean_EOD_TL', 'Semantic Stability: Mean EOD Termination Level', 'Mean EOD Termination Level', 'plot_impact_mean_eod_tl.png')
    create_bar_plot('Mean_Abs_SOD_Diff', 'Relational Shift: Mean Absolute SOD Difference', 'Mean Absolute Difference in SOD Scores', 'plot_impact_mean_abs_sod_diff.png')
    create_bar_plot('Mean_Abs_SOD_TL_Diff', 'Structural Path Divergence: Mean SOD_TL Difference', 'Mean Absolute Termination Level Difference', 'plot_impact_mean_abs_sod_tl_diff.png')
    print("  -> Corpus Impact plots created.")

def create_multicolor_dashed_line(x, y, color1, color2, label, dashes_per_segment=8):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    all_sub_segments, all_sub_segment_colors = [], []
    for segment in segments:
        p1, p2 = segment[0], segment[1]
        sub_points_x = np.linspace(p1[0], p2[0], dashes_per_segment + 1)
        sub_points_y = np.linspace(p1[1], p2[1], dashes_per_segment + 1)
        sub_points = np.array([sub_points_x, sub_points_y]).T.reshape(-1, 1, 2)
        new_segments = np.concatenate([sub_points[:-1], sub_points[1:]], axis=1)
        all_sub_segments.extend(new_segments)
        sub_colors = [color1, color2] * (dashes_per_segment // 2)
        if dashes_per_segment % 2 == 1: sub_colors.append(color1)
        all_sub_segment_colors.extend(sub_colors)
    return LineCollection(all_sub_segments, colors=all_sub_segment_colors, linewidth=2.5, label=label)

def plot_distribution_profiles(all_data):
    """Generates the advanced distribution plots for SOD/WOD scores."""
    print(f"\n[{10}/13] Generating Distribution of Damage Profile Plots...")
    df_hist = all_data.get('log_histograms', pd.DataFrame())
    if df_hist is None or df_hist.empty: return

    CORPUS_COLORS = {'ai_generated': '#32CD32', 'null_model': '#FF0000', 'complex': '#9A0EEA', 'ground_filtered': '#00008B', 'targeted_removal': '#069AF3', 'random_removal': '#0000FF'}
    DISPLAY_NAME_TO_KEY = {'Ai Generated': 'ai_generated', 'Null Model': 'null_model', 'Complex': 'complex', 'Ground Filtered': 'ground_filtered', 'Targeted Removal': 'targeted_removal', 'Random Removal': 'random_removal'}
    
    df_hist['probability'] = df_hist['count'] / df_hist.groupby('comparison')['count'].transform('sum')
    score_metrics = [m for m in df_hist['metric'].unique() if '_tl' not in m]
    for metric in score_metrics:
        fig, ax = plt.subplots(figsize=(16, 9))
        df_plot = df_hist[df_hist['metric'] == metric].copy()
        df_plot['comparison'] = df_plot['comparison'].str.replace("_", " ").str.title().str.replace("Merriam Webster", "Complex")
        for comparison_name in sorted(df_plot['comparison'].unique()):
            line_data = df_plot[df_plot['comparison'] == comparison_name]
            try:
                part1_name, part2_name = comparison_name.split(' Vs ')
                color1, color2 = CORPUS_COLORS[DISPLAY_NAME_TO_KEY[part1_name]], CORPUS_COLORS[DISPLAY_NAME_TO_KEY[part2_name]]
            except (ValueError, KeyError): color1, color2 = '#808080', '#808080'
            lc = create_multicolor_dashed_line(line_data['log10_bin_start'].values, line_data['probability'].values, color1, color2, comparison_name)
            ax.add_collection(lc)
        ax.autoscale_view()
        ax.set_title(f'Distribution of Pairwise Difference Magnitudes ({metric.upper()})', fontsize=20, pad=20)
        ax.set_xlabel('Log10( |Difference| + 1 )', fontsize=16); ax.set_ylabel('Probability', fontsize=16)
        ax.grid(True, which='both', linestyle='--')
        legend_elements = [mpatches.Patch(color=c, label=n) for n, k in sorted(DISPLAY_NAME_TO_KEY.items()) for c in [CORPUS_COLORS[k]]]
        ax.legend(handles=legend_elements, title='Corpus Legend', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(FULL_OUTPUT_DIR, f'plot_distribution_{metric}.png'), dpi=PLOT_DPI); plt.close()
    print("  -> Distribution profile plots created.")

def generate_brittleness_text_report(all_data):
    """Generates the detailed text file of most/least brittle words."""
    print(f"\n[{11}/13] Generating Per-Comparison Word Brittleness Text Report...")
    df_eod_long = all_data.get('eod_results_long', pd.DataFrame())
    if df_eod_long is None or df_eod_long.empty: return

    with open(BRITTLENESS_REPORT_TXT, "w") as f:
        f.write("="*90 + "\n" + "Word Semantic Stability Report".center(90) + "\n" + "="*90 + "\n\n")
        f.write(f"This report lists the Top {TOP_N_WORDS} most 'brittle' (highest EOD score) and most 'robust' (lowest positive EOD score) words for each inter-corpus comparison.\n")
        for comparison_key in sorted(df_eod_long['Comparison'].unique()):
            f.write("\n\n" + "-"*90 + f"\nComparison: {comparison_key.replace('_', ' ').title().replace('Merriam Webster', 'Complex')}\n" + "-"*90 + "\n")
            valid_df = df_eod_long[(df_eod_long['Comparison'] == comparison_key) & (df_eod_long['Mean_EOD_Score'] > -1)][['word', 'Mean_EOD_Score']].copy()
            if valid_df.empty:
                f.write("No valid EOD scores for this comparison.\n"); continue
            brittle = valid_df.sort_values(by='Mean_EOD_Score', ascending=False).head(TOP_N_WORDS).rename(columns={'word': f'Top {TOP_N_WORDS} Most Brittle Words', 'Mean_EOD_Score': 'EOD Score'})
            robust = valid_df[valid_df['Mean_EOD_Score'] >= 0].sort_values(by='Mean_EOD_Score', ascending=True).head(TOP_N_WORDS).rename(columns={'word': f'Top {TOP_N_WORDS} Most Robust Words', 'Mean_EOD_Score': 'EOD Score'})
            side_by_side = pd.concat([brittle.reset_index(drop=True), robust.reset_index(drop=True)], axis=1)
            f.write(side_by_side.to_string(index=False))
    print("  -> Per-Comparison Word Brittleness report created.")

# ==============================================================================
#                               Main Execution
# ==============================================================================
def main():
    """Main function to run the entire analysis and reporting pipeline."""
    print("#"*60 + "\n###   STARTING DEFINITIVE EOD & DIFFERENCE ANALYSIS   ###\n" + "#"*60)
    os.makedirs(FULL_OUTPUT_DIR, exist_ok=True)
    
    all_data = load_all_data()

    print(f"\n[{2}/13] Calculating Corpus Properties...")
    df_properties = calculate_corpus_properties()
    print("\n" + "="*80 + "\n                     TABLE 1: CORPUS PROPERTIES\n" + "-"*80)
    print(df_properties.to_string(index=False, float_format="%.2f") + f"\n{'='*80}")
    df_properties.to_csv(PROPERTIES_CSV, index=False, float_format="%.3f")

    master_summary_df = create_master_summary_df(all_data)

    print(f"\n[{3}/13] Calculating Correlation Between Divergence Metrics...")
    calculate_metric_correlations(master_summary_df)
    
    print(f"\n[{4}/13] Calculating Corpus Divergence Index...")
    df_divergence = calculate_corpus_divergence_index(master_summary_df)
    
    print(f"\n[{5}/13] Calculating Correlation of Divergence with Properties...")
    calculate_property_divergence_correlation(df_divergence, df_properties)
    
    print(f"\n[{6}/13] Analyzing Divergence from the Null Model...")
    analyze_divergence_from_null(master_summary_df)

    print(f"\n[{7}/13] Analyzing Divergence from Linguistic Corpora Group...")
    calculate_linguistic_divergence(master_summary_df)

    print(f"\n[{8}/13] Generating Corpus Impact Overview Plots...")
    generate_impact_plots(master_summary_df)
    
    print(f"\n[{9}/13] Generating Distribution of Damage Profile Plots...")
    plot_distribution_profiles(all_data)
    
    print(f"\n[{10}/13] Generating Per-Comparison Word Brittleness Text Report...")
    generate_brittleness_text_report(all_data)
    
    print(f"\n[{11}/13] Calculating Global Word Brittleness...")
    calculate_global_brittleness(all_data)
    
    print(f"\n[{12}/13] Calculating Language-Only Word Brittleness...")
    calculate_language_brittleness(all_data)
        
    print(f"\n\n[{13}/13] Analysis Pipeline Complete.") 
    print("\n" + "#"*49 + "\n###   FULL ANALYSIS PIPELINE COMPLETED   ###\n" + "#"*49)

if __name__ == '__main__':
    main()