import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error)

from paths import *


def get_evaluation_metrics(inv_test_Y: np.array, inv_pred_Y: np.array, prnt: bool = False, name:str = ''):
    """Calculates and optionally prints metrics given a set of real values and predictions
    
        Args:
            inv_test_Y:
            inv_pred_Y:
            prnt: Indicates whether the results should be printed on screen
            name:
    """
    mae   = mean_absolute_error(inv_test_Y, inv_pred_Y)
    rmse  = math.sqrt(mean_squared_error(inv_test_Y, inv_pred_Y))
    mape  = mean_absolute_percentage_error(inv_test_Y, inv_pred_Y)
    stdev = statistics.stdev(inv_test_Y-inv_pred_Y)
    mean  = statistics.mean(inv_test_Y-inv_pred_Y)
    sample_size = len(inv_test_Y)

    if prnt:
        print(f'{str.title(name)+" set":18}| MAE:     {mae  :>10.3f}s')
        print(f'{                    "":18}| RMSE:    {rmse :>10.3f}s')
        print(f'{                    "":18}| StDev:   {stdev:>10.3f}s')
        print(f'{                    "":18}| Mean:    {mean :>10.3f}s')
        print(f'{                    "":18}| MAPE:    {mape :>10.3f}')
        print(f'{                    "":18}| Muestra: {sample_size:>10,}')

    metrics = dict(
        mae=mae,
        rmse=rmse,
        mape=mape,
        stdev=stdev,
        mean=mean,
        sample_size=sample_size
    )

    return metrics


def display_errors(test_Y: np.array, pred_Y: np.array, title: str = None):
    fig,ax    = plt.subplots(1,1, figsize = (14,5))
    error     = pd.DataFrame(test_Y - pred_Y, columns=['error'])

    pepe = sns.histplot(error, ax=ax, binwidth=50, log_scale=(False, True))
    ax.set_title(title)
    ax.set(xlabel='Error (real-predicho)', 
           ylabel='Frecuencia')

### Deprecated?
def load_from_dataset(path: Path, date: str, airport: str, randomize:bool = False) -> pd.DataFrame:
    # print(airport, date)
    datasets = [tf.data.Dataset.load(str(ds)) 
                for ds in sorted(path.glob(f'*'))]
    
    if randomize:
        probs = [ds.cardinality().numpy() for ds in datasets]
        probs = [x/sum(probs) for x in probs]
        dataset = tf.data.Dataset.sample_from_datasets(datasets, weights=probs)
    else:
        dataset  = tf.data.Dataset.from_tensor_slices(datasets)
        dataset  = dataset.interleave(lambda x: x, cycle_length=1, 
                                    num_parallel_calls=tf.data.AUTOTUNE)
    
    return dataset