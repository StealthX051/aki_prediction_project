import vitaldb
import numpy as np
import pandas as pd
from data_preparation.waveform_processing import WAVEFORM_SPECS, process_signal

def debug_case():
    caseid = 3594
    waveform_key = 'SNUADC/PLETH'
    
    # From cohort file (approximate times based on file view, but I'll just load the whole case if I can or find the times)
    # Actually I need the opstart/opend.
    # Let's read them from the cohort file.
    cohort = pd.read_csv('data/processed/aki_pleth_ecg_co2_awp.csv')
    row = cohort[cohort['caseid'] == caseid].iloc[0]
    opstart = row['opstart']
    opend = row['opend']
    
    print(f"Debugging Case {caseid}, Waveform {waveform_key}")
    print(f"OpStart: {opstart}, OpEnd: {opend}")
    
    # Resolve spec key
    ID_TO_SPEC_KEY = {spec['id']: key for key, spec in WAVEFORM_SPECS.items()}
    spec_key = ID_TO_SPEC_KEY.get(waveform_key)
    if not spec_key and waveform_key in WAVEFORM_SPECS:
        spec_key = waveform_key
        
    if not spec_key:
        print(f"FAIL: Could not resolve spec key for {waveform_key}")
        return

    spec = WAVEFORM_SPECS[spec_key]
    native_sr = spec['native_sr']
    print(f"Native SR: {native_sr}")
    
    # 1. Load
    print("Loading data from VitalDB...")
    wave = vitaldb.load_case(caseid, spec['id'], interval=1/native_sr)
    
    if wave is None:
        print("FAIL: Wave is None")
        return
    
    print(f"Loaded shape: {wave.shape}")
    
    # 2. Slice
    start_idx = int(opstart * native_sr)
    end_idx = int(opend * native_sr)
    print(f"Slicing indices: {start_idx} to {end_idx}")
    
    seg = wave[start_idx:end_idx]
    seg = seg.squeeze()
    print(f"Segment shape: {seg.shape}")
    
    # 3. Raw Stats
    print(f"Raw Min: {np.nanmin(seg)}, Max: {np.nanmax(seg)}")
    print(f"Raw Mean: {np.nanmean(seg)}, Std: {np.nanstd(seg)}")
    print(f"Raw NaNs: {np.isnan(seg).sum()} ({np.mean(np.isnan(seg))*100:.2f}%)")
    
    # 4. Interpolate NaNs (step_02 logic)
    nan_mask = np.isnan(seg)
    if np.mean(nan_mask) > 0:
        print("Interpolating NaNs...")
        x = np.arange(seg.size)
        seg = np.interp(x=x, xp=x[~nan_mask], fp=seg[~nan_mask])
    
    print(f"After Interp NaNs: {np.isnan(seg).sum()}")
    print(f"After Interp Infs: {np.isinf(seg).sum()}")
    
    # 5. Process Signal - Step by Step
    print("Running process_signal (manual breakdown)...")
    
    # Filter
    try:
        filt_type, *params = spec['filter']
        print(f"Applying filter: {filt_type}, {params}")
        if filt_type == 'bpf':
            # TEST SOS FILTERING
            from scipy.signal import butter, sosfiltfilt
            def bpf_sos(x, fs, lo, hi, order=4):
                nyquist = fs / 2
                sos = butter(order, [lo / nyquist, hi / nyquist], btype='band', output='sos')
                return sosfiltfilt(sos, x)
            
            print("Using SOS filtering...")
            seg_filt = bpf_sos(seg, fs=native_sr, lo=params[0], hi=params[1])
            
        elif filt_type == 'lpf':
            from data_preparation.waveform_processing import lpf
            seg_filt = lpf(seg, fs=native_sr, fc=params[0])
        elif filt_type == 'hpf':
            from data_preparation.waveform_processing import hpf
            seg_filt = hpf(seg, fs=native_sr, fc=params[0])
            
        print(f"Filtered NaNs: {np.isnan(seg_filt).sum()}")
        print(f"Filtered Infs: {np.isinf(seg_filt).sum()}")
        print(f"Filtered Min/Max: {np.nanmin(seg_filt)}, {np.nanmax(seg_filt)}")
        
        seg = seg_filt
    except Exception as e:
        print(f"Filter failed: {e}")
        
    # Resample
    if spec['up'] != 1 or spec['down'] != 1:
        try:
            from data_preparation.waveform_processing import aa_downsample
            print(f"Resampling up={spec['up']}, down={spec['down']}")
            seg_res = aa_downsample(seg, spec['up'], spec['down'])
            print(f"Resampled NaNs: {np.isnan(seg_res).sum()}")
            seg = seg_res
        except Exception as e:
            print(f"Resample failed: {e}")

    seg_proc = seg
        
    print(f"Processed shape: {seg_proc.shape}")
    print(f"Processed Min: {np.nanmin(seg_proc)}, Max: {np.nanmax(seg_proc)}")
    print(f"Processed Mean: {np.nanmean(seg_proc)}, Std: {np.nanstd(seg_proc)}")
    print(f"Processed NaNs: {np.isnan(seg_proc).sum()}")
    
    # 6. Final Check
    if np.isnan(seg_proc).all() or np.nanstd(seg_proc) < 1e-6:
        print("FAIL: Final check failed (Flatline or All NaN)")
    else:
        print("PASS: Signal is valid")

if __name__ == "__main__":
    debug_case()
