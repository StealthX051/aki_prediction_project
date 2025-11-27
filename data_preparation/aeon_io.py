import os
import numpy as np
import pandas as pd
import joblib
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union, Any
from collections import defaultdict
from data_preparation.inputs import (
    AEON_OUT_DIR, AEON_SAVE_FORMATS, AEON_PAD_POLICY, AEON_PAD_LENGTH, 
    AEON_PAD_FILL, AEON_STRICT_CHANNELS, AEON_WINDOW_POLICY, OUTCOME
)

@dataclass
class AeonSeriesPayload:
    caseid: int
    waveform: str
    target_sr: float
    seg_full: Optional[np.ndarray] = None               # non-windowed
    win_mat: Optional[np.ndarray] = None                # shape (n_windows, win_samp) if windowed, else None
    valid_window_mask: Optional[np.ndarray] = None      # boolean mask length = n_windows
    length: Optional[int] = None                        # seg_full length or win_samp

def pad_1d(x: np.ndarray, pad_len: int, fill_value) -> np.ndarray:
    """Right-pad 1D array to pad_len."""
    if x.shape[0] >= pad_len:
        return x[:pad_len]
    
    if not isinstance(fill_value, (int, float)):
        fill_value = 0 
        
    return np.pad(x, (0, pad_len - x.shape[0]), mode='constant', constant_values=fill_value)

def collate_and_save_aeon(case_buffers: Dict[int, Dict[str, np.ndarray]], 
                          window_buffers: Dict[Tuple[int, int], Dict[str, np.ndarray]],
                          caseid_to_outcome: Dict[int, Any],
                          channel_order: List[str],
                          is_windowed: bool):
    """
    Aggregates processed waveforms and saves them in Aeon-compatible formats.
    """
    logging.info("Starting Aeon export collation...")
    os.makedirs(AEON_OUT_DIR, exist_ok=True)
    
    if not is_windowed:
        # --- Non-Windowed Export ---
        logging.info(f"Collating non-windowed data for {len(case_buffers)} cases.")
        
        final_cases = []
        final_labels = []
        
        # 1. Filter cases based on strict channel policy
        for caseid, wave_dict in case_buffers.items():
            if AEON_STRICT_CHANNELS:
                if all(c in wave_dict for c in channel_order):
                    final_cases.append((caseid, wave_dict))
                    final_labels.append({'caseid': caseid, OUTCOME: caseid_to_outcome.get(caseid)})
            else:
                final_cases.append((caseid, wave_dict))
                final_labels.append({'caseid': caseid, OUTCOME: caseid_to_outcome.get(caseid)})
        
        logging.info(f"Retained {len(final_cases)} cases after applying AEON_STRICT_CHANNELS={AEON_STRICT_CHANNELS}")

        if final_cases:
            # Save labels
            y_df = pd.DataFrame(final_labels)
            y_path = os.path.join(AEON_OUT_DIR, 'y_nonwindowed.csv')
            y_df.to_csv(y_path, index=False)
            logging.info(f"Saved non-windowed labels to {y_path}")

            # 2. Handle padding and save formats
            pad_len = AEON_PAD_LENGTH
            if AEON_PAD_POLICY == 'in_memory_pad' and pad_len is None:
                pad_len = max(len(seg) for _, wave_dict in final_cases for seg in wave_dict.values())
            
            # --- Save np_list_pkl (unequal length) ---
            if "np_list_pkl" in AEON_SAVE_FORMATS or AEON_PAD_POLICY == "aeon_padding_transformer":
                X_list = []
                for _, wave_dict in final_cases:
                    # Shape (n_channels, n_timepoints_i)
                    arr = np.stack([wave_dict[c] for c in channel_order if c in wave_dict])
                    X_list.append(arr)
                
                list_path = os.path.join(AEON_OUT_DIR, 'X_nonwindowed_np_list.pkl')
                joblib.dump(X_list, list_path)
                logging.info(f"Saved unequal length list (for PaddingTransformer) to {list_path}")

            # --- Save padded formats ---
            if AEON_PAD_POLICY == 'in_memory_pad':
                logging.info(f"Padding to length {pad_len} with fill '{AEON_PAD_FILL}'")
                padded_dicts = [] # List of {channel: 1D_array}
                
                for caseid, wave_dict in final_cases:
                    padded_wave_dict = {}
                    for c in channel_order:
                        if c in wave_dict:
                            padded_wave_dict[c] = pad_1d(wave_dict[c], pad_len, AEON_PAD_FILL)
                        else:
                            padded_wave_dict[c] = np.full(pad_len, fill_value=AEON_PAD_FILL, dtype=np.float32)
                    padded_dicts.append(padded_wave_dict)

                if "nested_pkl" in AEON_SAVE_FORMATS:
                    # Build nested DataFrame (n_cases x n_channels)
                    index = pd.Index([caseid for caseid, _ in final_cases], name='caseid')
                    X_nested = pd.DataFrame(columns=channel_order, index=index)
                    for i, (caseid, _) in enumerate(final_cases):
                        for c in channel_order:
                            X_nested.loc[caseid, c] = padded_dicts[i][c]
                    
                    nested_path = os.path.join(AEON_OUT_DIR, 'X_nonwindowed_nested.pkl')
                    joblib.dump(X_nested, nested_path)
                    logging.info(f"Saved padded nested DataFrame to {nested_path}")

                if "numpy3d_npz" in AEON_SAVE_FORMATS:
                    # Stack to (n_cases, n_channels, n_timepoints)
                    X_3d = np.stack([
                        np.stack([d[c] for c in channel_order]) for d in padded_dicts
                    ])
                    npz_path = os.path.join(AEON_OUT_DIR, 'X_nonwindowed.npz')
                    np.savez_compressed(npz_path, X=X_3d, caseids=y_df['caseid'].values, channels=channel_order)
                    logging.info(f"Saved padded 3D NumPy to {npz_path} with shape {X_3d.shape}")
        
    else:
        # --- Windowed Export ---
        logging.info(f"Collating windowed data for {len(window_buffers)} valid windows.")
        
        final_windows = [] # List of ( (caseid, win_idx), {wave_dict} )
        
        # 1. Filter windows based on policy
        if AEON_WINDOW_POLICY == 'intersection':
            for (caseid, win_idx), wave_dict in window_buffers.items():
                if all(c in wave_dict for c in channel_order):
                    final_windows.append(((caseid, win_idx), wave_dict))
        else: # 'union' (or default)
            final_windows = list(window_buffers.items()) # Keep all
        
        logging.info(f"Retained {len(final_windows)} windows after applying {AEON_WINDOW_POLICY} policy.")

        if final_windows:
            # Get final window length (should be consistent)
            win_len = final_windows[0][1][list(final_windows[0][1].keys())[0]].shape[0]
            
            # Build labels
            y_data = []
            for (caseid, win_idx), _ in final_windows:
                y_data.append({
                    'caseid': caseid, 
                    'window_idx': win_idx, 
                    OUTCOME: caseid_to_outcome.get(caseid),
                    'bag_id': caseid # for aeon
                })
            y_df = pd.DataFrame(y_data)
            y_path = os.path.join(AEON_OUT_DIR, 'y_windowed.csv')
            y_df.to_csv(y_path, index=False)
            logging.info(f"Saved windowed labels to {y_path}")

            # Prepare data for formats
            window_data_dicts = [] # List of {channel: 1D_array}
            multi_index = []
            
            for (caseid, win_idx), wave_dict in final_windows:
                multi_index.append((caseid, win_idx))
                processed_wave_dict = {}
                for c in channel_order:
                    if c in wave_dict:
                        processed_wave_dict[c] = wave_dict[c]
                    else:
                        # Pad missing channels (if union)
                        processed_wave_dict[c] = np.full(win_len, fill_value=AEON_PAD_FILL, dtype=np.float32)
                window_data_dicts.append(processed_wave_dict)

            if "nested_pkl" in AEON_SAVE_FORMATS:
                index = pd.MultiIndex.from_tuples(multi_index, names=['caseid', 'window_idx'])
                X_nested = pd.DataFrame(columns=channel_order, index=index)
                
                # This is faster than .loc
                for c in channel_order:
                    X_nested[c] = [d[c] for d in window_data_dicts]
                
                nested_path = os.path.join(AEON_OUT_DIR, 'X_windowed_nested.pkl')
                joblib.dump(X_nested, nested_path)
                logging.info(f"Saved windowed nested DataFrame to {nested_path}")

            if "numpy3d_npz" in AEON_SAVE_FORMATS:
                X_3d = np.stack([
                    np.stack([d[c] for c in channel_order]) for d in window_data_dicts
                ])
                npz_path = os.path.join(AEON_OUT_DIR, 'X_windowed.npz')
                np.savez_compressed(npz_path, X=X_3d, caseids=y_df['caseid'].values, channels=channel_order)
                logging.info(f"Saved windowed 3D NumPy to {npz_path} with shape {X_3d.shape}")
