from sys import argv

import pandas as pd

from sklearn.preprocessing import OneHotEncoder


def main():
    if len(argv) == 1:
        print("No files found!")
        return
    
    ohe = OneHotEncoder(sparse=False)
    for i, item in enumerate(argv[1:]):
        df = pd.DataFrame.from_csv(item, columns=[])
        print(df)
        
if __name__ == '__main__':
    main()
