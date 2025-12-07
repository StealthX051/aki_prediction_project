import os
import numpy as np
import pandas as pd
import joblib
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union, Any, FrozenSet
from numbers import Number
from data_preparation.inputs import (
    AEON_OUT_DIR, AEON_SAVE_FORMATS, AEON_PAD_POLICY, AEON_PAD_LENGTH, 
    AEON_PAD_FILL, AEON_STRICT_CHANNELS, AEON_WINDOW_POLICY, OUTCOME
)

@dataclass(frozen=True)
class AeonExportConfig:
    out_dir: str = AEON_OUT_DIR
    save_formats: FrozenSet[str] = field(
        default_factory=lambda: frozenset(AEON_SAVE_FORMATS)
    )
    pad_policy: str = AEON_PAD_POLICY  # 'in_memory_pad' or 'aeon_padding_transformer'
    # pad_length can be None (to infer) or an int
    pad_length: Optional[int] = AEON_PAD_LENGTH
    # fill value can be number, string 'nan', or None
    pad_fill: Union[float, str, None] = AEON_PAD_FILL
    strict_channels: bool = AEON_STRICT_CHANNELS
    window_policy: str = AEON_WINDOW_POLICY  # 'intersection' or 'union'
    outcome_col: str = OUTCOME

def pad_1d(
    x: np.ndarray,
    pad_len: int,
    fill_value: Union[Number, str, None],
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Right-pad (or truncate) a 1D array to pad_len.

    - Truncates if len(x) > pad_len.
    - Pads with `fill_value` if len(x) < pad_len.
    - Casts output to `dtype`.
    """
    x = np.asarray(x, dtype=dtype)

    if x.ndim != 1:
        raise ValueError(f"pad_1d expects 1D array, got shape {x.shape}")

    if pad_len <= 0:
        raise ValueError(f"pad_len must be positive, got {pad_len}")

    # handle common sentinel values
    if fill_value is None or (isinstance(fill_value, str) and fill_value.lower() == "nan"):
        fill = np.nan
    elif isinstance(fill_value, Number):
        fill = fill_value # Numpy cast handles this automatically during assignment/padding
    else:
        # Fallback for unexpected types
        try:
            fill = dtype.type(fill_value) # Try legacy way, or just fill_value
        except Exception:
             # In newer numpy, just pass value, let np.pad handle or cast explicitly
             fill = fill_value

    if x.shape[0] >= pad_len:
        return x[:pad_len]

    pad_width = pad_len - x.shape[0]
    return np.pad(x, (0, pad_width), mode="constant", constant_values=fill)

def collate_and_save_aeon(
    case_buffers: Dict[int, Dict[str, np.ndarray]], 
    window_buffers: Dict[Tuple[int, int], Dict[str, np.ndarray]],
    caseid_to_outcome: Dict[int, Any],
    channel_order: List[str],
    is_windowed: bool,
    config: AeonExportConfig = AeonExportConfig(),
) -> Dict[str, Any]:
    """
    Aggregates processed waveforms and saves them in Aeon-compatible formats.
    
    Returns metadata like:
    {
        "is_windowed": bool,
        "n_samples": int,
        "paths": {"nested_pkl": "...", "numpy3d_npz": "...", "y": "..."},
        "shape": {"X_3d": (n, c, t)},
    }
    """
    logging.info(f"Starting Aeon export collation (Windowed={is_windowed})...")
    os.makedirs(config.out_dir, exist_ok=True)
    
    metadata = {
        "is_windowed": is_windowed,
        "n_samples": 0,
        "paths": {},
        "shape": {}
    }

    if not is_windowed:
        # --- Non-Windowed Export ---
        logging.info(f"Collating non-windowed data for {len(case_buffers)} cases.")
        
        final_cases: List[Tuple[int, Dict[str, np.ndarray]]] = []
        final_labels: List[Dict[str, Any]] = []
        
        # 1. Filter cases based on config
        for caseid, wave_dict in case_buffers.items():
            # ensure we have at least one required channel
            present = [c for c in channel_order if c in wave_dict]

            if config.strict_channels and len(present) != len(channel_order):
                logging.debug(f"Dropping case {caseid} (strict): missing {set(channel_order) - set(present)}")
                continue

            if not present:
                logging.debug(f"Dropping case {caseid}: no channels from channel_order present.")
                continue

            if caseid not in caseid_to_outcome:
                logging.warning(f"No outcome for caseid={caseid}; setting {config.outcome_col}=NaN.")
            label_val = caseid_to_outcome.get(caseid, np.nan)

            final_cases.append((caseid, wave_dict))
            final_labels.append({"caseid": caseid, config.outcome_col: label_val})
        
        logging.info(f"Retained {len(final_cases)} cases after applying strict_channels={config.strict_channels}")
        metadata["n_samples"] = len(final_cases)

        if final_cases:
            # Save labels
            y_df = pd.DataFrame(final_labels)
            y_path = os.path.join(config.out_dir, 'y_nonwindowed.csv')
            y_df.to_csv(y_path, index=False)
            metadata["paths"]["y"] = y_path
            logging.info(f"Saved non-windowed labels to {y_path}")

            # 2. Handle padding and save formats
            pad_len = config.pad_length
            if config.pad_policy == 'in_memory_pad' and pad_len is None:
                pad_len = max(
                    wave_dict[c].shape[0]
                    for _, wave_dict in final_cases
                    for c in channel_order
                    if c in wave_dict
                )
                logging.info(f"Auto-inferred pad_len={pad_len} from channel_order.")
            
            # --- Save np_list_pkl (unequal length supported, or padded manually) ---
            if "np_list_pkl" in config.save_formats or config.pad_policy == "aeon_padding_transformer":
                X_list: List[np.ndarray] = []
                for caseid, wave_dict in final_cases:
                    channel_arrays = []
                    for c in channel_order:
                        if c in wave_dict:
                            arr = np.asarray(wave_dict[c], dtype=np.float32)
                        else:
                            # If non-strict, we might be missing a channel.
                            # We must pad it to match other channels in THE SAME CASE if we want a valid 2D array per case.
                            # But wait, usually all channels in a case share same time support.
                            # We'll use the length of the first available channel in this case.
                            example_c = next(c_ for c_ in channel_order if c_ in wave_dict)
                            example_len = wave_dict[example_c].shape[0]
                            arr = np.full(example_len, fill_value=config.pad_fill if config.pad_fill is not None else 0, dtype=np.float32)
                        channel_arrays.append(arr)
                    
                    # sanity check: all same length within case
                    lengths = {a.shape[0] for a in channel_arrays}
                    if len(lengths) > 1:
                        raise ValueError(f"Case {caseid}: inconsistent channel lengths: {lengths}")
                    X_list.append(np.stack(channel_arrays, axis=0)) # Shape (n_channels, n_timepoints)
                
                list_path = os.path.join(config.out_dir, 'X_nonwindowed_np_list.pkl')
                joblib.dump(X_list, list_path)
                metadata["paths"]["np_list_pkl"] = list_path
                logging.info(f"Saved list (len={len(X_list)}) to {list_path}")

            # --- Save padded formats ---
            if config.pad_policy == 'in_memory_pad':
                if pad_len is None or pad_len <= 0:
                    raise ValueError(f"Invalid pad_len={pad_len} for in_memory_pad")

                logging.info(f"Padding to length {pad_len} with fill '{config.pad_fill}'")
                
                padded_dicts: List[Dict[str, np.ndarray]] = []
                for caseid, wave_dict in final_cases:
                    padded_wave_dict: Dict[str, np.ndarray] = {}
                    for c in channel_order:
                        if c in wave_dict:
                            padded_wave_dict[c] = pad_1d(
                                wave_dict[c],
                                pad_len,
                                fill_value=config.pad_fill,
                                dtype=np.float32,
                            )
                        else:
                            padded_wave_dict[c] = np.full(
                                pad_len, fill_value=config.pad_fill if config.pad_fill is not None else 0, dtype=np.float32
                            )
                    padded_dicts.append(padded_wave_dict)

                if "nested_pkl" in config.save_formats:
                    # Build nested DataFrame (n_cases x n_channels)
                    index = pd.Index([caseid for caseid, _ in final_cases], name='caseid')
                    X_nested = pd.DataFrame(columns=channel_order, index=index)
                    for i, (caseid, _) in enumerate(final_cases):
                        for c in channel_order:
                            X_nested.loc[caseid, c] = padded_dicts[i][c]
                    
                    nested_path = os.path.join(config.out_dir, 'X_nonwindowed_nested.pkl')
                    joblib.dump(X_nested, nested_path)
                    metadata["paths"]["nested_pkl"] = nested_path
                    logging.info(f"Saved padded nested DataFrame to {nested_path}")

                if "numpy3d_npz" in config.save_formats:
                    # Stack to (n_cases, n_channels, n_timepoints)
                    X_3d = np.stack([
                        np.stack([d[c] for c in channel_order]) for d in padded_dicts
                    ])
                    # Ensure float32
                    X_3d = X_3d.astype(np.float32)
                    
                    npz_path = os.path.join(config.out_dir, 'X_nonwindowed.npz')
                    channels_arr = np.array(channel_order, dtype="U32")
                    np.savez_compressed(npz_path, X=X_3d, caseids=y_df['caseid'].values, channels=channels_arr)
                    metadata["paths"]["numpy3d_npz"] = npz_path
                    metadata["shape"]["X_3d"] = X_3d.shape
                    logging.info(f"Saved padded 3D NumPy to {npz_path} with shape {X_3d.shape}")
        
    else:
        # --- Windowed Export ---
        logging.info(f"Collating windowed data for {len(window_buffers)} buffers.")
        
        final_windows = [] # List of ( (caseid, win_idx), {wave_dict} )
        
        if config.window_policy not in {"intersection", "union"}:
            raise ValueError(f"Unsupported AEON_WINDOW_POLICY={config.window_policy}")

        # 1. Filter windows
        for (caseid, win_idx), wave_dict in window_buffers.items():
            if config.window_policy == "intersection":
                if all(c in wave_dict for c in channel_order):
                    final_windows.append(((caseid, win_idx), wave_dict))
            else: # union
                 if any(c in wave_dict for c in channel_order):
                    final_windows.append(((caseid, win_idx), wave_dict))
        
        logging.info(f"Retained {len(final_windows)} windows after applying {config.window_policy} policy.")
        metadata["n_samples"] = len(final_windows)

        if final_windows:
            # Infer window length from first valid channel of first window
            first_wave_dict = final_windows[0][1]
            first_chan_key = next(iter(first_wave_dict)) # There must be at least one
            win_len = first_wave_dict[first_chan_key].shape[0]
            
            # 2. Build labels
            y_data = []
            for (caseid, win_idx), _ in final_windows:
                if caseid not in caseid_to_outcome:
                     logging.warning(f"No outcome for caseid={caseid} (win {win_idx}).")
                
                y_data.append({
                    'caseid': caseid, 
                    'window_idx': win_idx, 
                    config.outcome_col: caseid_to_outcome.get(caseid, np.nan),
                    'bag_id': caseid 
                })
            y_df = pd.DataFrame(y_data)
            y_path = os.path.join(config.out_dir, 'y_windowed.csv')
            y_df.to_csv(y_path, index=False)
            metadata["paths"]["y"] = y_path
            logging.info(f"Saved windowed labels to {y_path}")

            # 3. Prepare data for formats
            window_data_dicts: List[Dict[str, np.ndarray]] = []
            multi_index = []
            
            fill_val_float = config.pad_fill if (config.pad_fill is not None and not isinstance(config.pad_fill, str)) else 0.0
            if isinstance(config.pad_fill, str) and config.pad_fill.lower() == 'nan':
                 fill_val_float = np.nan
            
            for (caseid, win_idx), wave_dict in final_windows:
                multi_index.append((caseid, win_idx))
                processed_wave_dict: Dict[str, np.ndarray] = {}
                for c in channel_order:
                    if c in wave_dict:
                        arr = np.asarray(wave_dict[c], dtype=np.float32)
                        if arr.shape[0] != win_len:
                            raise ValueError(
                                f"Inconsistent window length for case {caseid}, win {win_idx}, "
                                f"channel {c}: got {arr.shape[0]}, expected {win_len}"
                            )
                        processed_wave_dict[c] = arr
                    else:
                        processed_wave_dict[c] = np.full(
                            win_len, fill_value=fill_val_float, dtype=np.float32
                        )
                window_data_dicts.append(processed_wave_dict)

            if "nested_pkl" in config.save_formats:
                index = pd.MultiIndex.from_tuples(multi_index, names=['caseid', 'window_idx'])
                X_nested = pd.DataFrame(columns=channel_order, index=index)
                
                # This is faster than .loc
                for c in channel_order:
                    X_nested[c] = [d[c] for d in window_data_dicts]
                
                nested_path = os.path.join(config.out_dir, 'X_windowed_nested.pkl')
                joblib.dump(X_nested, nested_path)
                metadata["paths"]["nested_pkl"] = nested_path
                logging.info(f"Saved windowed nested DataFrame to {nested_path}")

            if "numpy3d_npz" in config.save_formats:
                X_3d = np.stack([
                    np.stack([d[c] for c in channel_order]) for d in window_data_dicts
                ])
                # Ensure float32
                X_3d = X_3d.astype(np.float32)
                
                npz_path = os.path.join(config.out_dir, 'X_windowed.npz')
                channels_arr = np.array(channel_order, dtype="U32")
                np.savez_compressed(npz_path, X=X_3d, caseids=y_df['caseid'].values, channels=channels_arr)
                metadata["paths"]["numpy3d_npz"] = npz_path
                metadata["shape"]["X_3d"] = X_3d.shape
                logging.info(f"Saved windowed 3D NumPy to {npz_path} with shape {X_3d.shape}")

    return metadata
