import sys
import os
import vitaldb
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preparation.waveform_processing import WAVEFORM_SPECS

def test_vitaldb_loading():
    print("Testing vitaldb loading with WAVEFORM_SPECS...")
    caseid = 1
    waveform_key = 'PLETH'
    
    if waveform_key not in WAVEFORM_SPECS:
        print(f"Error: {waveform_key} not in WAVEFORM_SPECS")
        return

    spec = WAVEFORM_SPECS[waveform_key]
    print(f"Spec for {waveform_key}: {spec}")
    
    try:
        # Use explicit interval to avoid numpy casting error in vitaldb library
        interval = 1/spec['native_sr']
        print(f"Attempting load with interval={interval}")
        wave = vitaldb.load_case(caseid, spec['id'], interval=interval)
        
        if wave is None:
            print("Result is None")
        else:
            print(f"Loaded shape: {wave.shape}")
            print(f"Sample data: {wave[:5]}")
            print("SUCCESS: VitalDB reading works.")
            
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vitaldb_loading()
