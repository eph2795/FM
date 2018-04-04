from json import load
from collections import defaultdict

import pandas as pd

import numpy as np
from scipy.sparse import hstack

from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression


def main(config):
    fname = config['file_name']
    if config['has_header']:
        df = pd.read_csv(config['file_name'], delimiter=config['delimiter'])
    else:
        df = pd.read_csv(config['file_name'], delimiter=config['delimiter'], header=None)

    def process_columns(l):
        if l is None:
            return []
        new_l = []
        for item in l:
            if isinstance(item, int):
                new_l.append(df.columns[item])
            else:
                new_l.append(item)
        return new_l

    to_throw = process_columns(config['throw_features'])
    target_col = process_columns(config['target_col'])
    f_col = process_columns(config['float_features'])
    cat_col = process_columns(config['categorical_features'])

    Y = df[target_col]

    cols = [i for i in range(df.shape[1])]
    cols = list(set(cols).difference(set(to_throw + target_col)))
    
    df = df.iloc[:, cols]
    df.to_csv(config['csv_name'], index=False)
    
    X_cat = df[cat_col] if cat_col else df.select_dtypes(exclude='floating')
    X_f = df[f_col] if f_col else df.select_dtypes(include='floating'))
    
    X_cat = np.array(X_cat, dtype=np.str)
    X_f = np.array(X_f)
    Y = np.array(Y).flatten()
    
    with open(config['vw_name'], 'w') as f:
        for i in range(Y.shape[0]):
            l = []
            l.append(str(Y[i]))
            l.append(' | ')
            for j in range(X_f.shape[1]):
                l.append(str(j))
                l.append(':')
                l.append(str(X_f[i, j]))
                l.append(' ')
            shift = X_f.shape[1]
            for j in range(X_cat.shape[1]):
                l.append(str(j + shift))
                l.append(':')
                l.append(X_cat[i, j])
                l.append(' ')
            l.append('\n')
            f.write(''.join(l))
    
    
if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = load(f)
    default_config = defaultdict(lambda: None)
    default_config.update(config)
    main(default_config)