import pandas as pd
import numpy as np
def split_by_time(df, col_mapping):
    ts_name = col_mapping['timestamp_col_name']
    # Преобразование столбца с датой и временем в формат datetime

    ts = pd.to_datetime(df[ts_name]).astype(int) // 10 ** 9
    condition_music = lambda A: A > 100
    result = (ts[1:].values - ts[:-1].values).astype(int)
    indx = np.where(condition_music(result))
    if len(indx[0]) == 0:
        return [0, -1]
    return indx[0]