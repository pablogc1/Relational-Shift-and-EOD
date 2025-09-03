################################################################################################################################################################################

[w007104@login3 ~]$ cat sbatch_step1_sync.slurm
#!/bin/bash
#SBATCH --job-name=sync_corpus_indices
#SBATCH --output=sync_corpus_indices_%j.out
#SBATCH --error=sync_corpus_indices_%j.err
#SBATCH --time=100:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1

# Purge modules and load a consistent Python environment
module purge
module load Python/3.10.8-GCCcore-12.2.0-bare

# Execute the Python script for synchronization
# The script will handle all file I/O and logic internally
echo "--> Starting Step 1: Vocabulary and Index Synchronization..."
python3 -u step1_synchronize_indices.py
echo "--> Step 1 finished."
[w007104@login3 ~]$ cat step1_synchronize_indices.py
import os
import json
from tqdm import tqdm

# ==============================================================================
#                      Configuration
# ==============================================================================
# Define the corpora we are working with.
# The keys will be used to name output files.
CORPORA = {
    "ground_filtered": "extracted_definitions_ground_filtered.txt",
    "null_model": "extracted_definitions_null_model.txt",
    "random_removal": "extracted_definitions_random_removal.txt",
    "targeted_removal": "extracted_definitions_targeted_removal.txt",
    # CHANGE 1: Update the AI generated filename
    "ai_generated": "extracted_definitions_ai_generated.txt",
    # CHANGE 2: Add the new Merriam-Webster corpus
    "merriam_webster": "extracted_definitions_merriam_webster.txt"
}

# Source for the pre-calculated OD pairs data
PAIRS_RESULTS_FILES = {
    "ground_filtered": "pairs_results_ground_filtered.txt",
    "null_model": "pairs_results_null_model.txt",
    "random_removal": "pairs_results_random_removal.txt",
    "targeted_removal": "pairs_results_targeted_removal.txt",
    "ai_generated": "pairs_results_ai_generated.txt",
    # CHANGE 2: Add the new Merriam-Webster pairs file
    "merriam_webster": "pairs_results_merriam_webster.txt"
}

# --- Output File Names ---
COMMON_VOCAB_FILE = "common_vocabulary.txt"
MASTER_INDEX_FILE = "master_word_to_idx.json"
SYNCED_PAIRS_DIR = "synced_pairs_results"


# ==============================================================================
#                            Helper Functions
# ==============================================================================

def read_vocabulary(file_path):
    """Reads a definitions file and returns a set of all headwords."""
    if not os.path.exists(file_path):
        print(f"  -> FATAL: Definition file not found: {file_path}")
        exit(1)
    vocab = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line: continue
            head, _ = line.split(":", 1)
            vocab.add(head.strip().lower())
    return vocab

def read_ordered_vocabulary(file_path):
    """Reads a definitions file and returns an ordered list of headwords."""
    ordered_vocab = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line: continue
            head, _ = line.split(":", 1)
            ordered_vocab.append(head.strip().lower())
    return ordered_vocab

# ==============================================================================
#                            Main Logic
# ==============================================================================

def main():
    """
    Main execution function to perform synchronization.
    """
    print("--- Starting Vocabulary Synchronization ---")

    # --- 1. Find Common Vocabulary ---
    print("\n[1/4] Finding common vocabulary across all corpora...")
    all_vocabs = [read_vocabulary(fp) for fp in CORPORA.values()]
    
    # Find the intersection of all vocabulary sets
    common_words = set.intersection(*all_vocabs)
    
    # Sort for consistent ordering
    common_words_sorted = sorted(list(common_words))
    
    print(f"  -> Found {len(common_words_sorted)} common words.")

    # --- 2. Create and Save Master Index ---
    print("\n[2/4] Creating and saving master index...")
    master_word_to_idx = {word: i for i, word in enumerate(common_words_sorted)}

    # Save the common vocabulary list for easy reference
    with open(COMMON_VOCAB_FILE, "w", encoding="utf-8") as f:
        for word in common_words_sorted:
            f.write(f"{word}\n")
    print(f"  -> Saved common vocabulary list to '{COMMON_VOCAB_FILE}'")

    # Save the master index map as a JSON file
    with open(MASTER_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(master_word_to_idx, f, indent=4)
    print(f"  -> Saved master word-to-index map to '{MASTER_INDEX_FILE}'")
    
    if not os.path.exists(SYNCED_PAIRS_DIR):
        os.makedirs(SYNCED_PAIRS_DIR)
        print(f"  -> Created directory for synchronized results: '{SYNCED_PAIRS_DIR}'")

    # --- 3. Process and Re-map Each Pairs File ---
    print("\n[3/4] Re-mapping pairs results files to master indices...")
    for corpus_key, pairs_file_path in PAIRS_RESULTS_FILES.items():
        print(f"\n  --- Processing corpus: {corpus_key} ---")
        
        if not os.path.exists(pairs_file_path):
            print(f"  -> WARNING: Pairs file not found: {pairs_file_path}. Skipping.")
            continue
        
        # Load the original ordered vocabulary for this specific corpus
        # to translate its old indices back to words.
        original_def_file = CORPORA[corpus_key]
        original_ordered_vocab = read_ordered_vocabulary(original_def_file)
        # The original indices were 1-based, so we create a 0-based map
        original_idx_to_word = {i: word for i, word in enumerate(original_ordered_vocab)}

        output_path = os.path.join(SYNCED_PAIRS_DIR, f"pairs_sync_{corpus_key}.txt")
        
        lines_written = 0
        total_lines_read = 0
        
        with open(pairs_file_path, "r") as f_in, open(output_path, "w") as f_out:
            for line in tqdm(f_in, desc=f"Re-mapping {corpus_key}", unit=" lines"):
                total_lines_read += 1
                parts = line.strip().split()
                if len(parts) < 2: continue
                
                # Original indices were 1-based in your script
                old_idx1 = int(parts[0]) - 1
                old_idx2 = int(parts[1]) - 1
                
                word1 = original_idx_to_word.get(old_idx1)
                word2 = original_idx_to_word.get(old_idx2)
                
                # Check if both words are in our common vocabulary
                if word1 in common_words and word2 in common_words:
                    # Get the new master indices
                    new_idx1 = master_word_to_idx[word1]
                    new_idx2 = master_word_to_idx[word2]
                    
                    # Construct the new line with re-mapped indices
                    remaining_data = " ".join(parts[2:])
                    # We use master indices (0-based) for consistency
                    f_out.write(f"{new_idx1} {new_idx2} {remaining_data}\n")
                    lines_written += 1
        
        print(f"  -> Finished processing. Read {total_lines_read} lines.")
        print(f"  -> Wrote {lines_written} synchronized lines to '{output_path}'")

    print("\n[4/4] Synchronization process complete.")
    print("\nNext steps:")
    print("1. Proceed with Step 2 (calculating pairwise differences) using the files in the 'synced_pairs_results' directory.")
    print("2. Proceed with Step 3 (EOD calculation).")


if __name__ == "__main__":
    main()

################################################################################################################################################################################

[w007104@login3 ~]$ cat sbatch_process_corpora_serially.slurm
#!/bin/bash
#SBATCH --job-name=serial_corpus_processor
#SBATCH --output=serial_corpus_processor_%j.out
#SBATCH --error=serial_corpus_processor_%j.err
#SBATCH --time=10:00:00  # Generous time for all 4 corpora
#SBATCH --mem=8G         # This master script doesn't need much memory
#SBATCH --cpus-per-task=1

# --- Setup ---
echo "--- Master Serial Processor Started ---"
module purge
module load Python/3.10.8-GCCcore-12.2.0-bare
pip install --user pandas tqdm > /dev/null 2>&1

# --- Define Corpora ---
CORPORA_KEYS=("ground_filtered" "null_model" "random_removal" "targeted_removal" "ai_generated" "merriam_webster")
WORKER_SLURM_SCRIPT="sbatch_indexing_worker.slurm" # The worker script for indexing

# --- Check for worker script ---
if [ ! -f "${WORKER_SLURM_SCRIPT}" ]; then
    echo "FATAL ERROR: The worker script '${WORKER_SLURM_SCRIPT}' was not found."
    exit 1
fi

# --- Loop Through Each Corpus One by One ---
for CORPUS_KEY in "${CORPORA_KEYS[@]}"; do
    echo ""
    echo "========================================================"
    echo "Processing Corpus: ${CORPUS_KEY}"
    echo "========================================================"

    # --- 1. Split the large file for the current corpus ---
    SYNC_FILE="synced_pairs_results/pairs_sync_${CORPUS_KEY}.txt"
    SPLIT_DIR="split_files_${CORPUS_KEY}"
    
    if [ ! -f "${SYNC_FILE}" ]; then
        echo "WARNING: Sync file not found for ${CORPUS_KEY}. Skipping."
        continue
    fi
    
    echo "Step 1: Splitting ${SYNC_FILE}..."
    rm -rf "${SPLIT_DIR}" # Clean up previous runs
    mkdir -p "${SPLIT_DIR}"
    split -l 5000000 --numeric-suffixes=1 -a 2 "${SYNC_FILE}" "${SPLIT_DIR}/chunk_"
    
    # --- 2. Count chunks and launch a waiting job array ---
    NUM_CHUNKS=$(ls -1 "${SPLIT_DIR}"/chunk_* | wc -l)
    if [ "${NUM_CHUNKS}" -eq 0 ]; then
        echo "WARNING: No chunks created for ${CORPUS_KEY}. Skipping."
        continue
    fi
    
    echo "Step 2: Submitting ${NUM_CHUNKS} worker jobs for ${CORPUS_KEY} and waiting..."
    
    # Use an environment variable to tell the worker which corpus to work on
    export CORPUS_TO_PROCESS=${CORPUS_KEY}
    
    # Submit the worker array and WAIT for it to complete.
    sbatch --wait --array=1-${NUM_CHUNKS} ${WORKER_SLURM_SCRIPT}

    # --- 3. Merge the temporary index files for this corpus ---
    FINAL_INDEX_DIR="rich_index_${CORPUS_KEY}"
    TEMP_DIR_PATTERN="temp_index_parts_${CORPUS_KEY}/job_*"

    echo "Step 3: Merging temporary index files for ${CORPUS_KEY}..."
    mkdir -p "${FINAL_INDEX_DIR}"
    
    # Find all unique index files (e.g., 0.csv, 1.csv) and merge them.
    find ${TEMP_DIR_PATTERN} -type f -name "*.csv" -printf "%f\n" | sort -u | while read filename; do
        cat ${TEMP_DIR_PATTERN}/${filename} >> "${FINAL_INDEX_DIR}/${filename}"
    done
    
    # --- 4. Clean up temporary files for this corpus ---
    echo "Step 4: Cleaning up temporary files for ${CORPUS_KEY}..."
    rm -rf "split_files_${CORPUS_KEY}"
    rm -rf "temp_index_parts_${CORPUS_KEY}"
    
    echo "--- Finished processing ${CORPUS_KEY} ---"
done

echo ""
echo "========================================================"
echo "All corpora have been processed successfully."
echo "========================================================"
[w007104@login3 ~]$ cat sbatch_indexing_worker.slurm
#!/bin/bash
#SBATCH --job-name=indexing_worker
#SBATCH --output=indexing_worker_%A_%a.out
#SBATCH --error=indexing_worker_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=1
# NOTE: The --array range is provided by the master script.

# --- Environment Setup ---
module purge
module load Python/3.10.8-GCCcore-12.2.0-bare
pip install --user pandas tqdm > /dev/null 2>&1

# --- Identify which chunk file to process ---
# The master script has set this environment variable for us.
SPLIT_DIR="split_files_${CORPUS_TO_PROCESS}"
CHUNK_FILES=($(ls -1 "${SPLIT_DIR}"/chunk_* | sort))
CHUNK_TO_PROCESS="${CHUNK_FILES[$SLURM_ARRAY_TASK_ID-1]}"

echo "Worker ${SLURM_ARRAY_TASK_ID} for corpus ${CORPUS_TO_PROCESS} processing ${CHUNK_TO_PROCESS}"

# --- Execute Python Script ---
python3 -u step_python_indexer.py \
    --input_file "${CHUNK_TO_PROCESS}" \
    --corpus_key "${CORPUS_TO_PROCESS}" \
    --job_id "${SLURM_ARRAY_TASK_ID}"

echo "Worker ${SLURM_ARRAY_TASK_ID} for corpus ${CORPUS_TO_PROCESS} finished."
[w007104@login3 ~]$ cat step_python_indexer.py
import os
import argparse
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Create a partial rich index from one chunk file.")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--corpus_key", type=str, required=True)
    parser.add_argument("--job_id", type=int, required=True)
    args = parser.parse_args()

    # Each corpus will have its own temporary parent directory.
    temp_output_dir = f"temp_index_parts_{args.corpus_key}/job_{args.job_id}"
    os.makedirs(temp_output_dir, exist_ok=True)
    
    data_buffer = defaultdict(list)
    
    HEADER = ['idx1', 'idx2', 'sod', 'sod_tl', 'wod', 'wod_tl', 'god']
    DTYPES = {'idx1': 'uint32', 'idx2': 'uint32', 'sod': 'int64', 'sod_tl': 'int16', 'wod': 'int64', 'wod_tl': 'int16'}

    try:
        df = pd.read_csv(args.input_file, sep=' ', header=None, names=HEADER, dtype=DTYPES, usecols=HEADER[:-1])
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"INFO: Job {args.job_id} found empty or missing file '{args.input_file}'. Finished.")
        return

    df = df[(df['sod'] != -1) & (df['wod'] != -1)].copy()

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Job {args.job_id} ({args.corpus_key})"):
        idx1, idx2 = row['idx1'], row['idx2']
        sod_score, wod_score, sod_tl, wod_tl = row['sod'], row['wod'], row['sod_tl'], row['wod_tl']
        
        line_content = f"{sod_score},{wod_score},{sod_tl},{wod_tl}\n"
        data_buffer[idx1].append(f"{idx2},{line_content}")
        data_buffer[idx2].append(f"{idx1},{line_content}")
        
    for idx, lines in data_buffer.items():
        with open(os.path.join(temp_output_dir, f"{idx}.csv"), "a") as f:
            f.writelines(lines)
            
    print(f"Job {args.job_id}: Finished processing {args.input_file}.")

if __name__ == "__main__":
    main()

################################################################################################################################################################################

[w007104@login3 ~]$ cat sbatch_step3_run_eod.slurm
#!/bin/bash
#SBATCH --job-name=eod_calculation
#SBATCH --output=eod_calculation_%A_%a.out
#SBATCH --error=eod_calculation_%A_%a.err
#SBATCH --array=1-40   # <-- CHANGED: Reduced to 40 jobs
#SBATCH --time=100:00:00 # <-- CHANGED: Reduced requested wall time
#SBATCH --mem=24G       # Keep memory generous for loading dictionaries
#SBATCH --cpus-per-task=16 # <-- CHANGED: Increased CPU cores per job

# --- Environment Setup ---
module purge
module load Python/3.10.8-GCCcore-12.2.0-bare
pip install --user tqdm > /dev/null 2>&1

# Create a directory for the EOD results
mkdir -p eod_results

# --- Execute the Python Worker Script ---
# The Python script will now use all 16 requested cores for its multiprocessing pool.
python3 -u step3_python_eod_worker.py \
    --job_id ${SLURM_ARRAY_TASK_ID} \
    --total_jobs ${SLURM_ARRAY_TASK_COUNT} \
    --num_workers ${SLURM_CPUS_PER_TASK}

echo "EOD Worker Job ${SLURM_ARRAY_TASK_ID} has finished."
[w007104@login3 ~]$ cat step3_python_eod_worker.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EOD Calculation Worker (Step 3 - Corrected)

This version fixes the CSV writing bug by using pandas to generate the output,
which correctly handles special characters in words (e.g., commas), thus
preventing downstream ParserErrors.
"""

import os
import argparse
import itertools
from collections import Counter
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd  # <-- Import pandas

DEFINITIONS_FILES = {
    "ground_filtered": "extracted_definitions_ground_filtered.txt",
    "null_model": "extracted_definitions_null_model.txt",
    "random_removal": "extracted_definitions_random_removal.txt",
    "targeted_removal": "extracted_definitions_targeted_removal.txt",
    # CHANGE 1: Update the AI generated filename
    "ai_generated": "extracted_definitions_ai_generated.txt",
    # CHANGE 2: Add the new Merriam-Webster corpus
    "merriam_webster": "extracted_definitions_merriam_webster.txt"
}
COMMON_VOCAB_FILE = "common_vocabulary.txt"
OUTPUT_DIR = "eod_results"

# --- Globals for worker processes ---
worker_sets_dicts = None
worker_word_map = None

# ===============================================
#   EOD LOGIC (This part remains unchanged)
# ===============================================

def process_GOD(words_to_expand, sets_dict_A, sets_dict_B, max_level=100):
    all_elements_seen = set(words_to_expand)
    elements_to_open_now = set(words_to_expand)
    for level in range(1, max_level + 1):
        elements_for_next_level = set()
        for word in elements_to_open_now:
            elements_for_next_level.update(sets_dict_A.get(word, []))
            elements_for_next_level.update(sets_dict_B.get(word, []))
        new_elements = elements_for_next_level - all_elements_seen
        if not new_elements: return level
        all_elements_seen.update(new_elements)
        elements_to_open_now = new_elements
    return max_level

def run_EOD_engine(seed_word, sets_dict_A, sets_dict_B, omega_god, key_A="Side 1", key_B="Side 2", verbose=False):
    E, U, R = {}, {}, {}
    log_steps = []
    
    E[(0, 1)], E[(0, 2)] = Counter([seed_word]), Counter([seed_word])
    global_E_side1, global_E_side2 = E[(0, 1)].copy(), E[(0, 2)].copy()

    for side in [1, 2]:
        U[(0, side)], R[(0, side)] = E[(0, side)].copy(), Counter()
    
    for level in range(1, omega_god + 2):
        if verbose: log_steps.append(f"\n--- Level {level} ---")
        
        E[(level, 1)] = Counter(e for elem in E.get((level - 1, 1), {}) for e in sets_dict_A.get(elem, []))
        E[(level, 2)] = Counter(e for elem in E.get((level - 1, 2), {}) for e in sets_dict_B.get(elem, []))
        if verbose:
            log_steps.append(f"  Expansion ({key_A}): {dict(E[(level, 1)])}")
            log_steps.append(f"  Expansion ({key_B}): {dict(E[(level, 2)])}")
            
        U[(level, 1)], R[(level, 1)] = E[(level, 1)].copy(), Counter()
        U[(level, 2)], R[(level, 2)] = E[(level, 2)].copy(), Counter()

        global_E_side1 += E.get((level, 1), Counter())
        global_E_side2 += E.get((level, 2), Counter())

        for m in range(level + 1):
            for side in [1, 2]:
                words_to_cancel = []
                check_set = global_E_side2 if side == 1 else global_E_side1
                for u_word in list(U.get((m, side), {})):
                    if check_set[u_word] > 0:
                        words_to_cancel.append(u_word)
                if words_to_cancel and verbose:
                    log_steps.append(f"  Cancellation: At level m={m} on side {side}, words {words_to_cancel} are canceled.")
                for u_word in words_to_cancel:
                    count = U[(m, side)].pop(u_word)
                    R[(m, side)][u_word] += count
        
        termination_triggered = False
        for m_term, s_term in U.keys():
            if m_term > 0 and not U[(m_term, s_term)]:
                termination_triggered = True
                break

        if termination_triggered:
            if verbose: log_steps.append("\nTERMINATION: An uncanceled set U (at level > 0) became empty.")
            score = sum(count * lvl for (lvl, s), r_set in R.items() for word, count in r_set.items())
            if verbose:
                log_steps.append("\n--- Final Score Calculation ---")
                for (lvl, side), r_set in sorted(R.items()):
                    if not r_set: continue
                    for word, count in r_set.items():
                        log_steps.append(f"  Level {lvl} ({key_A if side == 1 else key_B}): {count} cancellation(s) of '{word}' -> +{count * lvl} to score")
                log_steps.append("---------------------------------")
                log_steps.append(f"  Total Score = {score}")
                return score, level, "\n".join(log_steps)
            return score, level
            
        if level > omega_god:
            if verbose: return -1, level, "\n".join(log_steps) + "\nTERMINATION: GOD rule."
            return -1, level
            
    final_level = omega_god + 1
    return -1, final_level

# ===============================================
#       UTILITY AND I/O FUNCTIONS
# ===============================================

def read_definitions(file_path):
    definitions = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                head, def_text = line.split(":", 1)
                definitions[head.strip().lower()] = def_text.strip().split()
    return definitions

def run_canary_test(sets_dicts):
    print("--- Running Comprehensive EOD Canary Test for 'money' ---")
    word_to_test = "money"
    log_filename = "canary_test_eod_log.txt"
    corpus_combinations = list(itertools.combinations(DEFINITIONS_FILES.keys(), 2))

    with open(log_filename, "w", encoding="utf-8") as f:
        f.write(f"Comprehensive EOD Canary Test for word: '{word_to_test}'\n")
        
        for key_A, key_B in corpus_combinations:
            f.write("\n\n" + "="*80 + "\n")
            f.write(f"   EOD TEST: {key_A} vs {key_B}\n")
            f.write("="*80 + "\n\n")

            dict_A, dict_B = sets_dicts[key_A], sets_dicts[key_B]
            if word_to_test not in dict_A or word_to_test not in dict_B:
                f.write(f"SKIPPED: '{word_to_test}' not present in both corpora definitions.\n")
                continue

            omega_god = process_GOD([word_to_test], dict_A, dict_B)
            f.write(f"--- GOD Calculation ---\nResult: ω_GOD = {omega_god}\n\n")

            f.write("--- EOD Calculation (verbose) ---\n")
            eod_score, term_level, eod_log = run_EOD_engine(word_to_test, dict_A, dict_B, omega_god, key_A=key_A, key_B=key_B, verbose=True)
            f.write(eod_log)
            f.write(f"\n\n--- SUMMARY for {key_A} vs {key_B} ---\n")
            f.write(f"Final EOD Score: {eod_score}\n")
            f.write(f"Terminated at level: {term_level}\n")
            f.write("="*80)

    print(f"--- Canary test complete. Detailed log saved to '{log_filename}' ---")

def init_worker(sets_dicts, word_map):
    global worker_sets_dicts, worker_word_map
    worker_sets_dicts, worker_word_map = sets_dicts, word_map

# =================================================================
#               *** FIX IS IN THIS FUNCTION ***
# =================================================================
def worker_process_eod(word):
    """
    FIX: This function now returns a dictionary, which is safer for
    data with special characters. Pandas will handle converting this
    dictionary to a correctly formatted CSV row.
    """
    word_idx = worker_word_map[word]
    result_dict = {"master_idx": word_idx, "word": word}
    corpus_combinations = list(itertools.combinations(DEFINITIONS_FILES.keys(), 2))

    for key_A, key_B in corpus_combinations:
        dict_A = worker_sets_dicts[key_A]
        dict_B = worker_sets_dicts[key_B]
        
        # EOD can only run if the word exists in both definition sets for that pair
        if word not in dict_A or word not in dict_B:
            eod_score, term_level = -2, -2 # Use a special code for "word not found"
        else:
            omega_god = process_GOD([word], dict_A, dict_B)
            eod_score, term_level = run_EOD_engine(word, dict_A, dict_B, omega_god)
        
        result_dict[f"eod_score_{key_A}_vs_{key_B}"] = eod_score
        result_dict[f"eod_tl_{key_A}_vs_{key_B}"] = term_level
        
    return result_dict

# =================================================================
#               *** FIX IS IN THIS FUNCTION ***
# =================================================================
def main():
    parser = argparse.ArgumentParser(description="Run Eigen Ontological Differentiation analysis.")
    parser.add_argument("--job_id", type=int, required=True)
    parser.add_argument("--total_jobs", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()
    
    print("Loading all definition files...")
    all_sets_dicts = {key: read_definitions(fp) for key, fp in DEFINITIONS_FILES.items()}
    
    with open(COMMON_VOCAB_FILE, "r") as f:
        common_words = [line.strip() for line in f]
    master_word_to_idx = {word: i for i, word in enumerate(common_words)}
    
    if args.job_id == 1:
        run_canary_test(all_sets_dicts)

    chunk_size = (len(common_words) + args.total_jobs - 1) // args.total_jobs
    start = (args.job_id - 1) * chunk_size
    end = min(len(common_words), start + chunk_size)
    words_for_this_job = common_words[start:end]

    output_file = os.path.join(OUTPUT_DIR, f"eod_results_part_{args.job_id}.csv")
    print(f"Job {args.job_id}: Processing {len(words_for_this_job)} words using {args.num_workers} workers...")
    
    # --- FIX: Use Pandas to write the CSV correctly ---
    results_list = []
    init_args = (all_sets_dicts, master_word_to_idx)
    with Pool(processes=args.num_workers, initializer=init_worker, initargs=init_args) as pool:
        results_iterator = pool.imap_unordered(worker_process_eod, words_for_this_job)
        for result in tqdm(results_iterator, total=len(words_for_this_job), desc=f"Job {args.job_id}"):
            if result:
                results_list.append(result)

    if results_list:
        df_results = pd.DataFrame(results_list)
        
        # Define header columns to ensure a consistent order
        header_cols = ["master_idx", "word"]
        for key_A, key_B in itertools.combinations(DEFINITIONS_FILES.keys(), 2):
            header_cols.append(f"eod_score_{key_A}_vs_{key_B}")
            header_cols.append(f"eod_tl_{key_A}_vs_{key_B}")
        
        # Reorder DataFrame columns and save to CSV
        df_results = df_results[header_cols]
        df_results.to_csv(output_file, index=False)

    print(f"Job {args.job_id}: Analysis complete. Results saved to {output_file}")

if __name__ == '__main__':
    main()

################################################################################################################################################################################
[w007104@login3 ~]$ cat sbatch_step3b_merge_eod.slurm
#!/bin/bash
#SBATCH --job-name=merge_eod_results
#SBATCH --output=merge_eod_results_%j.out
#SBATCH --error=merge_eod_results_%j.err
#SBATCH --time=00:15:00   # Merging is very fast, 15 minutes is plenty.
#SBATCH --mem=2G          # Low memory needed for this task.
#SBATCH --cpus-per-task=1

echo "--- Starting Standalone Merge Job for EOD Results ---"

# --- Configuration ---
TEMP_DIR="eod_results"
FINAL_FILE="eod_results_final.csv"
PART_FILES_PATTERN="${TEMP_DIR}/eod_results_part_*.csv"

# --- Pre-flight Check ---
# Check if the temporary directory and any part files exist before proceeding.
if [ ! -d "${TEMP_DIR}" ]; then
    echo "FATAL: Directory '${TEMP_DIR}' not found. Cannot merge. Exiting."
    exit 1
fi
if ! ls ${PART_FILES_PATTERN} 1> /dev/null 2>&1; then
    echo "FATAL: No EOD part files (e.g., 'eod_results_part_*.csv') found in '${TEMP_DIR}' to merge. Exiting."
    exit 1
fi

echo "Found part files in '${TEMP_DIR}'. Proceeding with merge..."

# --- Robust Merging Logic ---

# 1. Find the first non-empty part file and copy its header to the final file.
# This is robust and handles cases where the first few jobs might not produce output.
HEADER_WRITTEN=false
# 'ls -v' sorts numerically (e.g., part_1, part_2, ... part_10)
for f in $(ls -v ${PART_FILES_PATTERN}); do
    # Check if file exists AND is not empty
    if [ -s "$f" ]; then
        head -n 1 "$f" > "${FINAL_FILE}"
        HEADER_WRITTEN=true
        echo "Header written to '${FINAL_FILE}' from source file: $f"
        break
    fi
done

# If no header was written (meaning all part files were empty), exit gracefully.
if [ "$HEADER_WRITTEN" = false ]; then
    echo "WARNING: All EOD part files were empty. The final file '${FINAL_FILE}' will also be empty."
    # Create an empty file to prevent downstream scripts from failing on a missing file.
    touch "${FINAL_FILE}"
    exit 0
fi

# 2. Append the data (all lines EXCEPT the header) from ALL part files.
# The '-q' (quiet) flag prevents tail from printing "==> filename <==" headers between files.
# 'ls -v' is used again to ensure the order is correct.
tail -n +2 -q $(ls -v ${PART_FILES_PATTERN}) >> "${FINAL_FILE}"

# --- Final Verification ---
LINE_COUNT=$(wc -l < "${FINAL_FILE}")
echo "Merge complete."
echo "Final file '${FINAL_FILE}' has been created with ${LINE_COUNT} lines."
echo "--- Merge Job Finished Successfully ---"

################################################################################################################################################################################
[w007104@login3 ~]$ cat sbatch_step4a_calc_diffs.slurm
#!/bin/bash
#SBATCH --job-name=calc_pairwise_diffs
#SBATCH --output=calc_pairwise_diffs_%A_%a.out
#SBATCH --error=calc_pairwise_diffs_%A_%a.err
#SBATCH --array=1-100   # We divide the 20,867 common words into 100 jobs
#SBATCH --time=05:00:00
#SBATCH --mem=8G        # Low memory needed due to the indexed approach
#SBATCH --cpus-per-task=1

# --- Environment Setup ---
module purge
module load Python/3.10.8-GCCcore-12.2.0-bare
pip install --user pandas tqdm > /dev/null 2>&1

# Create a directory for the temporary output parts
mkdir -p temp_diff_parts

# --- Execute the Python Worker Script ---
# The script will use the job ID to figure out which slice of words to process.
python3 -u step4_python_diff_worker.py \
    --job_id ${SLURM_ARRAY_TASK_ID} \
    --total_jobs ${SLURM_ARRAY_TASK_COUNT}

echo "Worker Job ${SLURM_ARRAY_TASK_ID} has finished."
[w007104@login3 ~]$ cat step4_python_diff_worker.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pairwise Difference Worker (Step 4a - Corrected)

This version fixes the double-counting bug by only processing pairs where
the first index is smaller than the second (master_idx1 < master_idx2).
This ensures each pair is processed exactly once across the entire job array.
It also fixes the output to use commas consistently.
"""
import os
import argparse
import itertools
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
INDEX_DIR_TEMPLATE = "rich_index_{corpus_key}"
COMMON_VOCAB_FILE = "common_vocabulary.txt"
OUTPUT_DIR = "temp_diff_parts"
CORPORA_KEYS = ["ground_filtered", "null_model", "random_removal", "targeted_removal", "ai_generated", "merriam_webster"]

def process_word(word_index, corpus_combinations, data_handles):
    """
    For a single word, calculates its pairwise differences with all subsequent
    words across all 6 corpus combinations.
    """
    for key_A, key_B in corpus_combinations:
        try:
            path_A = os.path.join(INDEX_DIR_TEMPLATE.format(corpus_key=key_A), f"{word_index}.csv")
            path_B = os.path.join(INDEX_DIR_TEMPLATE.format(corpus_key=key_B), f"{word_index}.csv")
            
            df_A = pd.read_csv(path_A, header=None, names=['partner_idx', 'sod_A', 'wod_A', 'sod_tl_A', 'wod_tl_A']).set_index('partner_idx')
            df_B = pd.read_csv(path_B, header=None, names=['partner_idx', 'sod_B', 'wod_B', 'sod_tl_B', 'wod_tl_B']).set_index('partner_idx')
            
            df_merged = df_A.join(df_B, how='inner')
            
            if df_merged.empty:
                continue

            # ==============================================================
            #           *** FIX #1: Prevent Double Counting ***
            # ==============================================================
            # Only consider pairs where this word_index is the smaller one.
            # This ensures the pair (i, j) is processed but (j, i) is not.
            df_merged = df_merged[df_merged.index > word_index]

            if df_merged.empty:
                continue
            # ==============================================================

            df_merged['diff_sod'] = df_merged['sod_A'] - df_merged['sod_B']
            df_merged['diff_wod'] = df_merged['wod_A'] - df_merged['wod_B']
            df_merged['diff_sod_tl'] = df_merged['sod_tl_A'] - df_merged['sod_tl_B']
            df_merged['diff_wod_tl'] = df_merged['wod_tl_A'] - df_merged['wod_tl_B']
            
            df_merged['master_idx1'] = word_index
            df_merged.reset_index(inplace=True)
            df_merged.rename(columns={'partner_idx': 'master_idx2'}, inplace=True)
            
            # Reorder columns for the output file
            output_df = df_merged[['master_idx1', 'master_idx2', 'diff_sod', 'diff_sod_tl', 'diff_wod', 'diff_wod_tl']]
            
            # ==============================================================
            #           *** FIX #2: Consistent Comma Delimiter ***
            # ==============================================================
            # to_csv defaults to using a comma, which is what we want.
            # No need to specify sep=',' but it makes the intent clear.
            output_key = f"{key_A}_vs_{key_B}"
            output_df.to_csv(data_handles[output_key], header=False, index=False, sep=',')

        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"Error processing word {word_index} for {key_A} vs {key_B}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Calculate pairwise OD differences using rich indices.")
    parser.add_argument("--job_id", type=int, required=True)
    parser.add_argument("--total_jobs", type=int, required=True)
    args = parser.parse_args()
    
    with open(COMMON_VOCAB_FILE, "r") as f:
        common_words_indices = list(range(len(f.readlines())))

    chunk_size = (len(common_words_indices) + args.total_jobs - 1) // args.total_jobs
    start = (args.job_id - 1) * chunk_size
    end = min(len(common_words_indices), start + chunk_size)
    indices_for_this_job = common_words_indices[start:end]
    
    print(f"Job {args.job_id}: Processing {len(indices_for_this_job)} word indices (from {start} to {end-1}).")
    
    corpus_combinations = list(itertools.combinations(CORPORA_KEYS, 2))
    
    data_handles = {}
    for key_A, key_B in corpus_combinations:
        output_key = f"{key_A}_vs_{key_B}"
        output_path = os.path.join(OUTPUT_DIR, f"diffs_{output_key}_part_{args.job_id}.csv") # Note: using .csv extension
        data_handles[output_key] = open(output_path, "w")

    for word_idx in tqdm(indices_for_this_job, desc=f"Job {args.job_id}"):
        process_word(word_idx, corpus_combinations, data_handles)

    for handle in data_handles.values():
        handle.close()
        
    print(f"Job {args.job_id}: Finished.")

if __name__ == "__main__":
    main()
    
################################################################################################################################################################################

[w007104@login3 ~]$ cat sbatch_step4b_merge_diffs.slurm
#!/bin/bash
#SBATCH --job-name=merge_diffs
#SBATCH --output=merge_diffs_%j.out
#SBATCH --error=merge_diffs_%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

echo "--- Starting Final Merge of Pairwise Difference Files ---"

# --- Configuration ---
TEMP_DIR="temp_diff_parts"
FINAL_DIR="pairwise_diff_results"
# This order MUST EXACTLY MATCH the Python script's CORPORA_KEYS list
CORPORA_KEYS=("ground_filtered" "null_model" "random_removal" "targeted_removal" "ai_generated" "merriam_webster")

# Clean up previous runs and create fresh directories
rm -rf "${FINAL_DIR}"
mkdir -p "${FINAL_DIR}"

# ==============================================================================
#      *** THE CORRECT FIX: Replicate Python's itertools.combinations ***
# ==============================================================================
# This logic uses array indices to generate pairs based on their original
# position in the array, exactly matching the Python worker script.

num_keys=${#CORPORA_KEYS[@]}

# Outer loop from the first element (index 0) to the second-to-last
for (( i=0; i<num_keys-1; i++ )); do

    # Inner loop from the element AFTER the outer loop's element to the last
    for (( j=i+1; j<num_keys; j++ )); do
    
        # Get the keys based on their index
        key_A=${CORPORA_KEYS[i]}
        key_B=${CORPORA_KEYS[j]}

        # Construct the filename key exactly as the Python script does
        COMBO_KEY="${key_A}_vs_${key_B}"
        PART_FILES_PATTERN="${TEMP_DIR}/diffs_${COMBO_KEY}_part_*.csv"
        FINAL_FILE="${FINAL_DIR}/diffs_${COMBO_KEY}.csv"

        # Use a safer way to check if files exist
        if ! ls ${PART_FILES_PATTERN} 1> /dev/null 2>&1; then
            echo "No part files found for ${COMBO_KEY}. Skipping."
            continue
        fi

        echo "Merging results for: ${COMBO_KEY}"

        # Create the final file and add the header.
        echo "master_idx1,master_idx2,diff_sod,diff_sod_tl,diff_wod,diff_wod_tl" > "${FINAL_FILE}"

        # Concatenate all partial files into the final file.
        cat ${PART_FILES_PATTERN} >> "${FINAL_FILE}"
    done
done
# ==============================================================================

echo "--- Merge Complete. Cleaning up temporary directory. ---"
rm -rf "${TEMP_DIR}"

echo "--- Final difference files are ready in '${FINAL_DIR}'. ---"

################################################################################################################################################################################
[w007104@login3 ~]$ cat sbatch_step5a_loghist_worker.slurm
#!/bin/bash
#SBATCH --job-name=loghist_worker
#SBATCH --output=loghist_worker_%A_%a.out
#SBATCH --error=loghist_worker_%A_%a.err
#SBATCH --array=1-10      # A job for each of the 6 corpus combinations
#SBATCH --time=10:00:00  # 1 hour should be plenty for one combination
#SBATCH --mem=24G        # 16GB is safe for loading one large diff file
#SBATCH --cpus-per-task=16 # Give pandas a few cores for processing

# --- Environment Setup ---
module purge
module load Python/3.10.8-GCCcore-12.2.0-bare
pip install --user pandas tqdm numpy > /dev/null 2>&1

# Create a directory for the temporary output parts
mkdir -p temp_loghist_parts

# --- Define the 6 unique combinations in a specific order ---
# This array maps the SLURM job ID to a specific task.
COMPARISONS=(
    "ground_filtered_vs_null_model"
    "ground_filtered_vs_random_removal"
    "ground_filtered_vs_targeted_removal"
    "ground_filtered_vs_ai_generated"
    "ground_filtered_vs_merriam_webster"
    "null_model_vs_random_removal"
    "null_model_vs_targeted_removal"
    "null_model_vs_ai_generated"
    "null_model_vs_merriam_webster"
    "random_removal_vs_targeted_removal"
    "random_removal_vs_ai_generated"
    "random_removal_vs_merriam_webster"
    "targeted_removal_vs_ai_generated"
    "targeted_removal_vs_merriam_webster"
    "ai_generated_vs_merriam_webster"
)

# Get the specific comparison for this job from the array
CURRENT_COMPARISON=${COMPARISONS[$SLURM_ARRAY_TASK_ID-1]}

echo "Worker Job ${SLURM_ARRAY_TASK_ID}: Processing comparison '${CURRENT_COMPARISON}'"

# --- Execute the Python Worker Script ---
python3 -u step5a_python_loghist_worker.py --comparison_key "${CURRENT_COMPARISON}"

echo "Worker Job ${SLURM_ARRAY_TASK_ID} has finished."
[w007104@login3 ~]$ cat step5a_python_loghist_worker.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# step5a_python_loghist_worker.py (Corrected with Chunking)
#
import pandas as pd
import numpy as np
import argparse
import os
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Generate log-binned histogram data for one corpus combination.")
    parser.add_argument("--comparison_key", type=str, required=True)
    args = parser.parse_args()

    # --- Configuration ---
    input_dir = "pairwise_diff_results"
    output_dir = "temp_loghist_parts"
    chunk_size = 5_000_000  # Process 5 million rows at a time
    
    input_file = os.path.join(input_dir, f"diffs_{args.comparison_key}.csv")
    output_file = os.path.join(output_dir, f"loghist_{args.comparison_key}.csv")
    
    if not os.path.exists(input_file):
        print(f"INFO: Input file not found, skipping: {input_file}")
        return

    print(f"Reading {input_file} in chunks...")
    
    metrics = ['sod', 'wod', 'sod_tl', 'wod_tl']
    # Use a dictionary of lists to store binned data for all metrics
    binned_data_collector = {f'diff_{m}': [] for m in metrics}

    try:
        reader = pd.read_csv(input_file, sep=',', chunksize=chunk_size)

        for chunk in tqdm(reader, desc=f"Processing {args.comparison_key}"):
            for metric in metrics:
                col = f'diff_{metric}'
                if col not in chunk.columns: continue

                abs_diffs = chunk[col].abs()
                log_abs_diffs = np.log10(abs_diffs + 1)
                binned_data_collector[col].append(log_abs_diffs)
    
    except Exception as e:
        print(f"ERROR: Could not read or process {input_file}. Error: {e}")
        return

    # Now, concatenate the results and compute the final histogram
    all_final_hist_data = []
    for metric in metrics:
        col = f'diff_{metric}'
        if not binned_data_collector[col]: continue

        print(f"Finalizing histogram for: {col}")
        # Combine all the processed chunks for this metric
        final_log_series = pd.concat(binned_data_collector[col], ignore_index=True)
        
        max_log_val = int(np.ceil(final_log_series.max()))
        bins = np.arange(0, max_log_val + 2)
        
        binned_series = pd.cut(final_log_series, bins=bins, right=False, labels=bins[:-1])
        hist_counts = binned_series.value_counts().sort_index()
        
        for bin_start, count in hist_counts.items():
            all_final_hist_data.append({
                'comparison': args.comparison_key,
                'metric': col,
                'log10_bin': bin_start,
                'count': count
            })

    if all_final_hist_data:
        df_out = pd.DataFrame(all_final_hist_data)
        df_out.to_csv(output_file, index=False)
        print(f"Successfully wrote log-histogram data to {output_file}")

if __name__ == "__main__":
    main()

################################################################################################################################################################################
[w007104@login3 ~]$ cat sbatch_step5b_merge_loghist.slurm
#!/bin/bash
#SBATCH --job-name=merge_loghist
#SBATCH --output=merge_loghist_%j.out
#SBATCH --error=merge_loghist_%j.err
#SBATCH --time=10:10:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1

echo "--- Starting Merge Job for Log-Histogram Data ---"

# --- Configuration ---
TEMP_DIR="temp_loghist_parts"
FINAL_FILE="log_histogram_data.csv"
PART_FILES_PATTERN="${TEMP_DIR}/loghist_*.csv"

# --- Pre-flight Check ---
# The only change is on the next line: 2&>1 has been corrected to 2>&1
if ! ls ${PART_FILES_PATTERN} 1> /dev/null 2>&1; then
    echo "FATAL: No log-histogram part files found in '${TEMP_DIR}' to merge."
    exit 1
fi

# --- Merge Logic ---
# 1. Write the header from the first file that exists.
HEADER_WRITTEN=false
for f in $(ls -v ${PART_FILES_PATTERN}); do
    if [ -s "$f" ]; then
        head -n 1 "$f" > "${FINAL_FILE}"
        HEADER_WRITTEN=true
        break
    fi
done

if [ "$HEADER_WRITTEN" = false ]; then
    echo "WARNING: All part files were empty. Exiting."
    exit 0
fi

# 2. Append data (without headers) from all part files.
tail -n +2 -q $(ls -v ${PART_FILES_PATTERN}) >> "${FINAL_FILE}"

echo "Merge complete. Final data in '${FINAL_FILE}'."
echo "Cleaning up temporary directory..."
rm -rf "${TEMP_DIR}"
echo "--- Merge Job Finished ---"
################################################################################################################################################################################
[w007104@login3 ~]$ cat sbatch_step5_final_report.slurm
#!/bin/bash
#SBATCH --job-name=final_summary_report
#SBATCH --output=final_summary_report_%j.out
#SBATCH --error=final_summary_report_%j.err
#SBATCH --time=100:00:00  # 2 hours is very safe for this task
#SBATCH --mem=24G        # Generous memory for pandas to read result files
#SBATCH --cpus-per-task=1

# --- Environment Setup ---
echo "--- Starting Definitive Final Report Generation ---"
module purge
module load Python/3.10.8-GCCcore-12.2.0-bare
pip install --user pandas tqdm numpy > /dev/null 2>&1

# --- Execute the Python Aggregation Script ---
python3 -u step5_generate_summary.py

# --- Final Check ---
if [ $? -eq 0 ]; then
    echo "Python script finished successfully."
    echo "Please check the final output files: summary_report.txt and log_binned_histogram_data.csv"
    echo ""
    echo "--- FINAL REPORT PREVIEW ---"
    cat summary_report.txt
    echo "--- END OF PREVIEW ---"
else
    echo "ERROR: Python script exited with a non-zero status."
fi

echo "--- Final Report Generation Job Complete ---"
[w007104@login3 ~]$ cat step5_generate_summary.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definitive Final Summary Report Generator (Step 5)

This script generates two final, comprehensive analysis outputs:
1.  A detailed summary_report.txt containing all classical and advanced
    distributional statistics (Mean, Median, IQR, Outliers).
2.  A clean log_binned_histogram_data.csv file for creating distributional plots.

It uses a memory-efficient two-pass strategy for robust calculation on large files.
"""
import pandas as pd
import itertools
import os
from tqdm import tqdm
import numpy as np

# --- Configuration ---
EOD_RESULTS_FILE = "eod_results_final.csv"
PAIRWISE_DIFF_DIR = "pairwise_diff_results"
COMMON_VOCAB_FILE = "common_vocabulary.txt"
FINAL_REPORT_FILE = "summary_report.txt"
LOG_HISTOGRAM_FILE = "log_binned_histogram_data.csv"

CORPORA_KEYS = ["ground_filtered", "null_model", "random_removal", "targeted_removal", "ai_generated", "merriam_webster"]
CHUNK_SIZE = 5_000_000

def main():
    """
    Main function to generate all final analysis outputs.
    """
    # --- Pre-flight Checks ---
    if not all(os.path.exists(f) for f in [EOD_RESULTS_FILE, PAIRWISE_DIFF_DIR, COMMON_VOCAB_FILE]):
        print("FATAL: One or more required input files or directories are missing.")
        return

    all_log_hist_data = []

    with open(FINAL_REPORT_FILE, "w") as f_report:
        print("--- Generating Final, All-Inclusive Summary Report ---")

        # =============================================================
        # Section 1: Global Analysis Statistics
        # =============================================================
        print("Calculating Section 1: Global Stats...")
        f_report.write("="*80 + "\nSECTION 1: GLOBAL ANALYSIS STATISTICS\n" + "="*80 + "\n\n")
        with open(COMMON_VOCAB_FILE, "r") as vocab_f:
            num_common_words = len(vocab_f.readlines())
        total_possible_pairs = (num_common_words * (num_common_words - 1)) // 2
        f_report.write(f"{'Total Common Words:':<35} {num_common_words:,}\n")
        f_report.write(f"{'Total Possible Word Pairs:':<35} {total_possible_pairs:,}\n\n")

        # =============================================================
        # Section 2: Eigen Ontological Differentiation (EOD) Summary
        # =============================================================
        print("Calculating Section 2: EOD Summary...")
        f_report.write("="*80 + "\nSECTION 2: EIGEN ONTOLOGICAL DIFFERENTIATION (EOD) SUMMARY\n" + "="*80 + "\n")
        try:
            df_eod = pd.read_csv(EOD_RESULTS_FILE)
            eod_summary_data = []
            for key_A, key_B in itertools.combinations(CORPORA_KEYS, 2):
                combo_key = f"{key_A}_vs_{key_B}"
                score_col, tl_col = f"eod_score_{combo_key}", f"eod_tl_{combo_key}"
                valid_runs = df_eod[df_eod[score_col] > -1]
                eod_summary_data.append({
                    "Corpus Combination": combo_key, "Valid Words": f"{len(valid_runs):,}",
                    "Mean EOD Score": valid_runs[score_col].mean(), "Median EOD Score": valid_runs[score_col].median(),
                    "Mean EOD TL": valid_runs[tl_col].mean(), "Median EOD TL": valid_runs[tl_col].median()
                })
            df_eod_summary = pd.DataFrame(eod_summary_data)
            f_report.write(df_eod_summary.to_string(index=False, float_format="%.2f") + "\n")
        except Exception as e:
            f_report.write(f"Could not process EOD results due to an error: {e}\n")

        # =============================================================
        # Section 3: Inter-Corpus Pairwise Difference Summary
        # =============================================================
        print("\nCalculating Section 3: Full Pairwise Difference Summary...")
        f_report.write("\n\n" + "="*80 + "\nSECTION 3: INTER-CORPUS PAIRWISE DIFFERENCE SUMMARY\n" + "="*80 + "\n")
        
        metrics = ['sod', 'wod', 'sod_tl', 'wod_tl']
        
        for key_A, key_B in itertools.combinations(CORPORA_KEYS, 2):
            combo_key = f"{key_A}_vs_{key_B}"
            f_report.write("\n\n" + "-"*80 + f"\nComparison: {combo_key.upper()}\n" + "-"*80 + "\n")
            
            diff_file = os.path.join(PAIRWISE_DIFF_DIR, f"diffs_{combo_key}.csv")
            if not os.path.exists(diff_file):
                f_report.write("  -> Difference file not found. Skipping.\n")
                continue

            # --- Pass 1: Memory-efficient Mean calculation ---
            print(f"  ({combo_key}) Pass 1/2: Calculating Mean via chunking...")
            net_sum = pd.Series(0.0, index=[f'diff_{m}' for m in metrics])
            total_count = 0
            try:
                reader = pd.read_csv(diff_file, sep=',', chunksize=CHUNK_SIZE)
                for chunk in tqdm(reader, desc=f"  Mean ({combo_key})"):
                    net_sum += chunk[[f'diff_{m}' for m in metrics]].sum()
                    total_count += len(chunk)
                net_mean = net_sum / total_count if total_count > 0 else net_sum
                coverage = (total_count / total_possible_pairs) * 100 if total_possible_pairs > 0 else 0
                f_report.write(f"  Pairs Compared: {total_count:,} ({coverage:.2f}% coverage)\n\n")
            except Exception as e:
                f_report.write(f"  -> ERROR during mean calculation: {e}. Skipping.\n")
                continue

            # --- Pass 2: Memory-intensive calculations, one metric at a time ---
            print(f"  ({combo_key}) Pass 2/2: Calculating Median, IQR, and Outliers...")
            for metric in metrics:
                col = f'diff_{metric}'
                f_report.write(f"  --- Metric: {metric.upper()} ---\n")
                try:
                    data_series = pd.read_csv(diff_file, sep=',', usecols=[col]).squeeze("columns")
                    abs_series = data_series.abs()
                    
                    # Classical Stats
                    f_report.write(f"    Net Mean:       {net_mean[col]:,.2f}\n")
                    f_report.write(f"    Net Median:     {data_series.median():,.2f}\n")
                    f_report.write(f"    Absolute Mean:  {(abs_series.mean()):,.2f}\n")
                    f_report.write(f"    Absolute Median:{abs_series.median():,.2f}\n")
                    
                    # Advanced Distributional Stats (on log-transformed data)
                    log_abs_series = np.log10(abs_series + 1)
                    q1 = log_abs_series.quantile(0.25)
                    q3 = log_abs_series.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outliers = log_abs_series[(log_abs_series < lower_bound) | (log_abs_series > upper_bound)]
                    
                    f_report.write(f"    Log10 IQR:      {iqr:.4f}\n")
                    f_report.write(f"    Extreme Outliers: {len(outliers):,} ({len(outliers)/total_count:.4%})\n")

                    # Generate and store log histogram data for this metric
                    max_log = int(np.ceil(log_abs_series.max()))
                    bins = np.arange(0, max_log + 2)
                    binned_data = pd.cut(log_abs_series, bins=bins, right=False).value_counts().sort_index()
                    for interval, count in binned_data.items():
                        all_log_hist_data.append({
                            'comparison': combo_key, 'metric': col,
                            'log10_bin_start': interval.left, 'count': count
                        })

                except Exception as e:
                    f_report.write(f"    -> ERROR processing column {col}: {e}\n")

    # --- Write the separate log histogram data file ---
    if all_log_hist_data:
        print(f"\nWriting log histogram data to '{LOG_HISTOGRAM_FILE}'...")
        df_log_hist = pd.DataFrame(all_log_hist_data)
        df_log_hist.to_csv(LOG_HISTOGRAM_FILE, index=False)
        print("Log histogram data file created successfully.")

    print("\n--- Definitive Summary Report Generation Complete! ---")

if __name__ == "__main__":
    main()

################################################################################################################################################################################

