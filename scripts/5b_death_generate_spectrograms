# =============================================================================
# 5_generate_spectrograms_death.py
#
# This is the main, production-ready script for generating spectrograms for
# the entire dataset, specifically adapted for the mortality prediction task.
#
# Key Features:
# - Processes all cases from the final mortality cohort.
# - Uses a highly optimized numpy-based method for fast image creation.
# - Uses multithreading for high-speed processing.
# - Saves spectrograms as standardized 256x256 3-channel RGB PNGs.
# - Implements robust logging to a file for monitoring progress.
# - Includes checkpointing to automatically resume if interrupted.
# - Includes graceful shutdown handling to ensure manifests are always written.
# =============================================================================

# --- Standard Library Imports ---
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union, Tuple, List

# --- Third-Party Imports ---
import numpy as np
import pandas as pd
import vitaldb
import matplotlib.cm as cm # Import colormaps directly
from scipy import signal
from tqdm import tqdm
from PIL import Image

# --- Configuration Block ---

# 1. Data and Path Configuration
# Note: These paths are relative to the 'scripts' directory where this file lives.
# UPDATED to point to the new death cohort files.
COHORT_FILE_PATH = '../data/processed/final_cohort_with_death_label.csv'
TRAIN_IDS_FILE = '../data/processed/preop_train_cleaned_death_cohort.csv'
TEST_IDS_FILE = '../data/processed/preop_test_cleaned_death_cohort.csv'
OUTPUT_DIR = '../data/processed/'
SPECTROGRAM_SUBDIR = 'spectrograms/pleth' # This remains the same
LOG_FILE_PATH = 'spectrogram_generation_death.log' # New log file for this run

# 2. Waveform & Processing Parameters
TRACK_NAME = 'SNUADC/PLETH'
ORIGINAL_SR = 500
DOWNSAMPLE_SR = 100
SEGMENT_SECONDS = 10
SEGMENT_SAMPLES = DOWNSAMPLE_SR * SEGMENT_SECONDS
MAX_NAN_PERCENTAGE = 0.05

# 3. Spectrogram and Image Configuration
FILTER_ORDER = 4
LOW_HZ = 0.5
HIGH_HZ = 8.0
SPECTROGRAM_WINDOW_SAMPLES = 256
SPECTROGRAM_OVERLAP_SAMPLES = 128
SPECTROGRAM_FFT_POINTS = 256
IMAGE_DIMS = (256, 256)

# --- Core Processing Functions ---

def setup_logging():
    """Configures logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE_PATH, mode='w'),
            logging.StreamHandler()
        ]
    )

def save_spectrogram_as_image_optimized(Sxx_db: np.ndarray, output_path: str):
    """
    Saves a spectrogram matrix as a 3-channel RGB PNG using a fast,
    numpy-centric method that bypasses matplotlib's rendering engine.
    """
    # Flip the array vertically so the 0 Hz frequency is at the bottom.
    Sxx_db_flipped = np.flipud(Sxx_db)
    
    # Normalize the spectrogram data to the [0, 1] range for the colormap
    vmin, vmax = np.percentile(Sxx_db_flipped, [5, 95])
    Sxx_db_clipped = np.clip(Sxx_db_flipped, vmin, vmax)
    Sxx_db_normalized = (Sxx_db_clipped - vmin) / (vmax - vmin)
    
    # Apply the 'viridis' colormap
    rgba_array = cm.viridis(Sxx_db_normalized)
    
    # Convert the RGBA array to an 8-bit RGB array
    rgb_array = (rgba_array[:, :, :3] * 255).astype(np.uint8)
    
    # Create an image from the numpy array and resize it
    img = Image.fromarray(rgb_array).resize(IMAGE_DIMS, Image.Resampling.LANCZOS)
    
    # Save the final image
    img.save(output_path, 'png')

def process_segment(segment: np.ndarray) -> Union[np.ndarray, None]:
    """
    Processes a single, clean 10-second segment.
    Returns only the spectrogram matrix.
    """
    if np.var(segment) < 1e-6: return None
    try:
        b, a = signal.butter(FILTER_ORDER, [LOW_HZ, HIGH_HZ], btype='band', fs=DOWNSAMPLE_SR)
        filtered_segment = signal.filtfilt(b, a, segment)
        _, _, Sxx = signal.spectrogram(filtered_segment, fs=DOWNSAMPLE_SR, nperseg=SPECTROGRAM_WINDOW_SAMPLES, noverlap=SPECTROGRAM_OVERLAP_SAMPLES, nfft=SPECTROGRAM_FFT_POINTS)
        Sxx_db = 20 * np.log10(Sxx + 1e-9)
        return Sxx_db.astype(np.float32) if not np.all(np.isnan(Sxx_db)) else None
    except (ValueError, np.linalg.LinAlgError):
        return None

def process_case(case_data: dict, output_dir: str) -> List[Tuple[str, int, int]]:
    """Main worker function to process a single patient case."""
    # UPDATED to use 'death_label'
    caseid, opstart_sec, opend_sec, death_label = case_data['caseid'], case_data['opstart'], case_data['opend'], case_data['death_label']
    results = []
    try:
        raw_waveform = vitaldb.load_case(caseid, TRACK_NAME, 1 / ORIGINAL_SR)
        if raw_waveform is None or len(raw_waveform) == 0:
            return []

        op_waveform = raw_waveform[int(opstart_sec * ORIGINAL_SR):int(opend_sec * ORIGINAL_SR)]
        if len(op_waveform) < (SEGMENT_SECONDS * ORIGINAL_SR):
            return []

        resample_len = int(len(op_waveform) * DOWNSAMPLE_SR / ORIGINAL_SR)
        if resample_len <= 0: return []
        resampled_waveform = signal.resample(op_waveform, resample_len).flatten()

        for i in range(0, len(resampled_waveform) - SEGMENT_SAMPLES, SEGMENT_SAMPLES):
            segment = resampled_waveform[i: i + SEGMENT_SAMPLES]
            if np.isnan(segment).sum() > 0:
                if (np.isnan(segment).sum() / SEGMENT_SAMPLES) <= MAX_NAN_PERCENTAGE:
                    segment = pd.Series(segment).interpolate(method='linear', limit_direction='both').to_numpy()
                else:
                    continue

            spectrogram_matrix = process_segment(segment)
            if spectrogram_matrix is not None:
                segment_index = i // SEGMENT_SAMPLES
                output_filename = f"{caseid}_{segment_index}.png"
                output_filepath = os.path.join(output_dir, output_filename)
                save_spectrogram_as_image_optimized(spectrogram_matrix, output_filepath)
                # UPDATED to append 'death_label'
                results.append((output_filepath, death_label, caseid))
    except Exception as e:
        logging.error(f"Failed to process case {caseid}: {e}", exc_info=False)
    return results

# --- Main Execution Block ---

def main():
    """Main function to drive the spectrogram generation pipeline."""
    setup_logging()
    logging.info("--- Phase 5: Scalable Waveform Data Generation Pipeline (Mortality Cohort) ---")

    # 1. Setup and Load Data
    output_spectrogram_dir = os.path.join(OUTPUT_DIR, SPECTROGRAM_SUBDIR)
    os.makedirs(output_spectrogram_dir, exist_ok=True)
    
    try:
        cohort_df = pd.read_csv(COHORT_FILE_PATH)
        # The cleaned data files only contain caseids, so they don't need to be changed
        train_ids_df = pd.read_csv(TRAIN_IDS_FILE)
        test_ids_df = pd.read_csv(TEST_IDS_FILE)
        train_case_ids = set(train_ids_df['caseid'])
        test_case_ids = set(test_ids_df['caseid'])
    except FileNotFoundError as e:
        logging.error(f"CRITICAL ERROR: A required data file is missing: {e}. Halting.")
        return

    # This 'finally' block ensures that manifest generation ALWAYS runs,
    # even if the script is interrupted with Ctrl+C.
    try:
        # 2. Checkpointing: Identify already processed cases
        logging.info("Step 1/3: Loading cohort and identifying cases to process...")
        try:
            processed_files = os.listdir(output_spectrogram_dir)
            processed_case_ids = {int(f.split('_')[0]) for f in processed_files if f.endswith('.png')}
            logging.info(f"Found {len(processed_case_ids)} already processed cases. These will be skipped.")
            cases_to_process_df = cohort_df[~cohort_df['caseid'].isin(processed_case_ids)]
        except FileNotFoundError:
            processed_case_ids = set()
            cases_to_process_df = cohort_df
        
        tasks = cases_to_process_df.to_dict('records')
        if not tasks:
            logging.info("All cases have already been processed. Moving to manifest generation.")
        else:
            # 3. Process Cases in Parallel
            logging.info(f"Step 2/3: Starting parallel processing for {len(tasks)} new cases...")
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                future_to_case = {executor.submit(process_case, task, output_spectrogram_dir): task for task in tasks}
                progress = tqdm(as_completed(future_to_case), total=len(tasks), desc="Generating Spectrograms")
                for future in progress:
                    try:
                        future.result()
                    except Exception as e:
                        logging.error(f"A case-level task failed during execution: {e}")

    except KeyboardInterrupt:
        logging.warning("\nKeyboard interrupt detected! Proceeding to final manifest generation.")
    
    finally:
        # 4. Generate Final CSV Manifests
        logging.info("--- Entering Finalization Step (runs on success, error, or interrupt) ---")
        logging.info("Step 3/3: Scanning all spectrograms to create final mapping files...")
        
        try:
            all_generated_files = [os.path.join(output_spectrogram_dir, f) for f in os.listdir(output_spectrogram_dir) if f.endswith('.png')]
            if not all_generated_files:
                logging.error("No PNG files found after processing. Cannot create manifests. Exiting.")
                return

            # UPDATED to use 'death_label'
            label_map = cohort_df.set_index('caseid')['death_label'].to_dict()
            all_results = []
            for f_path in all_generated_files:
                try:
                    caseid = int(os.path.basename(f_path).split('_')[0])
                    if caseid in label_map:
                        all_results.append((os.path.abspath(f_path), label_map[caseid], caseid))
                except (IndexError, ValueError):
                    logging.warning(f"Could not parse caseid from filename: {f_path}. Skipping.")
                    continue

            # UPDATED to use 'death_label'
            results_df = pd.DataFrame(all_results, columns=['image', 'death_label', 'caseid'])
            train_df = results_df[results_df['caseid'].isin(train_case_ids)][['image', 'death_label']]
            test_df = results_df[results_df['caseid'].isin(test_case_ids)][['image', 'death_label']]
            
            # UPDATED to new manifest filenames
            final_train_manifest = os.path.join(OUTPUT_DIR, 'train_data_death.csv')
            final_test_manifest = os.path.join(OUTPUT_DIR, 'test_data_death.csv')

            train_df.to_csv(final_train_manifest, index=False)
            test_df.to_csv(final_test_manifest, index=False)
            
            logging.info("Final manifests created successfully.")
            logging.info(f"Total training samples available: {len(train_df)}")
            logging.info(f"Total testing samples available: {len(test_df)}")
            logging.info(f"Train manifest saved to: {final_train_manifest}")
            logging.info(f"Test manifest saved to: {final_test_manifest}")
            logging.info("--- Phase 5 Complete ---")

        except Exception as e:
            logging.error(f"An error occurred during final manifest generation: {e}", exc_info=True)


if __name__ == '__main__':
    # This makes the script safe to use with multiprocessing on Windows
    # Although we are using ThreadPoolExecutor, it's good practice.
    import multiprocessing
    multiprocessing.freeze_support()
    main()
