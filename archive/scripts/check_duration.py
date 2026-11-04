import pandas as pd
try:
    df = pd.read_csv('/home/exouser/js2-volume/projects/aki_prediction_project/data/processed/aki_pleth_ecg_co2_awp.csv')
    df['duration'] = df['opend'] - df['opstart']
    print(f"Max duration (s): {df['duration'].max()}")
    print(f"Max duration (h): {df['duration'].max()/3600}")
except Exception as e:
    print(e)
