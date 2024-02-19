import pandas as pd


def pre_process():
    df = pd.read_csv('australian.csv', delimiter=' ')
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(r'\d+:', '', regex=True)
    df = df.iloc[:, :-1]
    df = df.iloc[1:]
    df.to_csv('new_australian.csv', index=False)


if __name__ == '__main__':
    pre_process()
