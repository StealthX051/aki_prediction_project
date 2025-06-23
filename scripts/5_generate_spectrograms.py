import os
import pandas as pd
import numpy as np
import vitaldb
import logging
import time
from scipy import signal
from tqdm import tqdm
import multiprocessing

# --- Central Configuration ---
# This block holds all key parameters for easy modification and experimentation.

# 1. Data Paths
COHORT_FILE_PATH = 'data/processed/final_cohort_with_labels.csv'
TRAIN_SET_PATH = 'data/processed/preop_train_cleaned.csv'
TEST_SET_PATH = 'data/processed/preop_test_cleaned.csv'
SPECTROGRAM_OUTPUT_DIR = 'data/processed/spectrograms/pleth'
FINAL_TRAIN_DF_PATH = 'data/processed/train_data.csv'
FINAL_TEST_DF_PATH = 'data/processed/test_data.csv'
LOG_FILE_PATH = 'scripts/spectrogram_generation.log'

# 2. Waveform & Processing Parameters
TRACK_NAME = 'SNUADC/PLETH'
ORIGINAL_SR = 500
DOWNSAMPLE_SR = 100
SEGMENT_SECONDS = 10
SEGMENT_SAMPLES = DOWNSAMPLE_SR * SEGMENT_SECONDS

# 3. Spectrogram Generation Parameters
SPECTROGRAM_WINDOW_SAMPLES = 256
SPECTROGRAM_OVERLAP_SAMPLES = 128
SPECTROGRAM_FFT_POINTS = 256

# 4. Signal Cleaning Parameters
FILTER_ORDER = 4
LOW_HZ = 0.5
HIGH_HZ = 8.0

# --- Best Practice Functions ---

def setup_logging():
    """Sets up logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE_PATH, mode='w'), # Start with a fresh log each run
            logging.StreamHandler()
        ]
    )

def process_segment(segment, sr):
    """
    Processes a single 10-second waveform segment by cleaning it and
    converting it into a spectrogram.
    """
    # 1. Quality Control on input
    if np.all(np.isnan(segment)) or np.nanvar(segment) < 1e-6:
        return None
        
    try:
        # 2. Signal Filtering
        b, a = signal.butter(FILTER_ORDER, [LOW_HZ, HIGH_HZ], btype='band', fs=sr)
        filtered_segment = signal.filtfilt(b, a, segment)
    except (ValueError, np.linalg.LinAlgError):
        # Handle cases where the filter fails on unstable signals
        return None
    
    # 3. Spectrogram Generation
    f, t, Sxx = signal.spectrogram(filtered_segment,
                                   fs=sr,
                                   nperseg=SPECTROGRAM_WINDOW_SAMPLES,
                                   noverlap=SPECTROGRAM_OVERLAP_SAMPLES,
                                   nfft=SPECTROGRAM_FFT_POINTS)
    
    # 4. Conversion to Decibels
    Sxx_db = 20 * np.log10(Sxx + 1e-9)

    # 5. ***CRITICAL FIX***: Final quality check on the output spectrogram
    # Ensure that the output itself is not all NaN before returning.
    if np.all(np.isnan(Sxx_db)):
        return None
        
    return Sxx_db

def process_single_case(caseid):
    """
    Worker function to process all data for a single caseid.
    This will be run in parallel for each case.
    """
    try:
        raw_waveform = vitaldb.load_case(caseid, [TRACK_NAME], 1/ORIGINAL_SR)
        if np.all(np.isnan(raw_waveform)):
            return (caseid, 'skipped_nan')

        resample_len = int(len(raw_waveform) * DOWNSAMPLE_SR / ORIGINAL_SR)
        resampled_waveform = signal.resample(raw_waveform, resample_len).flatten()

        num_segments = len(resampled_waveform) // SEGMENT_SAMPLES
        if num_segments == 0:
            return (caseid, 'skipped_short')

        for i in range(num_segments):
            segment = resampled_waveform[i * SEGMENT_SAMPLES:(i + 1) * SEGMENT_SAMPLES]
            spectrogram = process_segment(segment, DOWNSAMPLE_SR)

            if spectrogram is not None:
                filename = f"{caseid}_{i}.npy"
                filepath = os.path.join(SPECTROGRAM_OUTPUT_DIR, filename)
                np.save(filepath, spectrogram.astype(np.float32))
        
        return (caseid, 'success')
    except Exception as e:
        return (caseid, f'error: {str(e)}')

# --- Main Processing Logic ---

def main():
    """Main function to drive the spectrogram generation pipeline."""
    setup_logging()
    start_time = time.time()
    logging.info("--- Phase 5: Scalable Waveform Data Generation Pipeline (PARALLEL) ---")

    # 1. Load Cohort and Split Information
    logging.info("Step 1/5: Loading cohort and data splits...")
    try:
        cohort_df = pd.read_csv(COHORT_FILE_PATH, usecols=['caseid', 'aki_label'])
        train_df = pd.read_csv(TRAIN_SET_PATH, usecols=['caseid'])
        test_df = pd.read_csv(TEST_SET_PATH, usecols=['caseid'])
    except FileNotFoundError as e:
        logging.error(f"A required data file is missing: {e}. Halting.")
        return

    label_map = cohort_df.set_index('caseid')['aki_label'].to_dict()
    train_caseids = set(train_df['caseid'])
    cases_to_process = cohort_df['caseid'].unique().tolist()

    # 2. Prepare Output Directories and Checkpoints
    logging.info(f"Step 2/5: Ensuring output directory exists at '{SPECTROGRAM_OUTPUT_DIR}'...")
    os.makedirs(SPECTROGRAM_OUTPUT_DIR, exist_ok=True)
    
    processed_caseids = {int(f.split('_')[0]) for f in os.listdir(SPECTROGRAM_OUTPUT_DIR) if f.endswith('.npy')}
    if processed_caseids:
        logging.info(f"Found {len(processed_caseids)} already processed cases. They will be skipped.")
        cases_to_process = [c for c in cases_to_process if c not in processed_caseids]
    
    logging.info(f"Total cases to process in this run: {len(cases_to_process)}")
    if not cases_to_process:
        logging.info("No new cases to process. Moving to final file creation.")
    else:
        # 3. Process Cases in Parallel
        logging.info(f"Step 3/5: Starting parallel processing with {multiprocessing.cpu_count()} workers...")
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = list(tqdm(pool.imap_unordered(process_single_case, cases_to_process), total=len(cases_to_process)))

        success_count = sum(1 for r in results if r[1] == 'success')
        error_count = sum(1 for r in results if 'error' in r[1])
        logging.info(f"Parallel processing complete. Success: {success_count}, Errors: {error_count}")
        for caseid, status in results:
            if 'error' in status:
                logging.error(f"Case {caseid} failed with status: {status}")

    # 4. Re-scan ALL files to create final dataframes
    logging.info("Step 4/5: Scanning all spectrograms to create final mapping files...")
    all_results = []
    all_files = os.listdir(SPECTROGRAM_OUTPUT_DIR)
    for filename in tqdm(all_files, "Mapping files"):
        if filename.endswith(".npy"):
            try:
                caseid = int(filename.split('_')[0])
                if caseid in label_map:
                    dataset_type = 'train' if caseid in train_caseids else 'test'
                    all_results.append({
                        'image': os.path.join('spectrograms/pleth', filename),
                        'aki_label': label_map[caseid],
                        'dataset_type': dataset_type
                    })
            except (KeyError, ValueError, IndexError):
                continue
    
    # 5. Create and Save Final DataFrames
    logging.info("Step 5/5: Creating and saving final train/test dataframes...")
    if not all_results:
        logging.error("No spectrograms were found or mapped. Halting.")
        return

    results_df = pd.DataFrame(all_results)
    final_train_df = results_df[results_df['dataset_type'] == 'train'][['image', 'aki_label']]
    final_test_df = results_df[results_df['dataset_type'] == 'test'][['image', 'aki_label']]
    final_train_df.to_csv(FINAL_TRAIN_DF_PATH, index=False)
    final_test_df.to_csv(FINAL_TEST_DF_PATH, index=False)

    end_time = time.time()
    duration_minutes = (end_time - start_time) / 60
    logging.info("--- Phase 5 Complete ---")
    logging.info(f"Script finished in {duration_minutes:.2f} minutes.")
    logging.info(f"Total training samples available: {len(final_train_df)}")
    logging.info(f"Total testing samples available: {len(final_test_df)}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
