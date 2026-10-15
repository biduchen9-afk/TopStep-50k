import pandas as pd, numpy as np
SCR = '/tmp/claude-0/-home-user-TopStep-50k/6aef8a09-9112-51df-af8b-197cbd9daa4c/scratchpad'


def load(mkt='nq'):
    import os
    p = f'{SCR}/{mkt}_raw.pkl'
    if os.path.exists(p):
        return pd.read_pickle(p)
    df = pd.read_csv(f'/home/user/data_raw/{mkt}_databento.txt', sep=';', header=None,
                     names=['ts', 'o', 'h', 'l', 'c', 'v'], dtype={'ts': str})
    df['ts'] = pd.to_datetime(df['ts'], format='%Y%m%d %H%M%S')
    df = df.set_index('ts')
    df.to_pickle(p)
    return df


if __name__ == '__main__':
    df = load('nq')
    print(df.shape, df.index.min(), df.index.max(), df.index.is_monotonic_increasing, df.index.duplicated().sum())
