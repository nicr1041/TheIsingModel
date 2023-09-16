# Script that combines all the raw data csv files (as yielded by the Monte Carlo simulations) in one csv file.
# It also cuts off the thermalization.
# The path has to be a directory in which all the raw data files of all the necessary Monte Carlo simulations are stored. Furthermore, it is important to notice
# that all the csv files starting with "lattice" are considered to be raw data csv files. Therefore, it must be checked that all the files in the directory starting with "lattice" are raw data files.

import os
import sys
import pandas as pd



def write_files(sources, combined):
    with open(sources[0], 'r') as first:
        combined.write(first.read())

    for i in range(1, len(sources)):
        with open(sources[i], 'r') as s:
            # Ignore the rest of the headers
            next(s, None)
            for line in s:
                combined.write(line)

def concatenate_csvs(root_path):
    filenames=[]
    for root, sub, files in os.walk(root_path):
        for filename in files:
            if filename.endswith('.csv') and filename.startswith('lattice'):
                filenames.append(os.path.join(root, filename))

    print(filenames)

    combined_path = os.path.join(root_path, 'reprocessed_ising_wolf.csv')
    with open(combined_path, 'w+') as combined:
        write_files(filenames, combined)

def load_and_cutoff_thermalization(path):
    df=pd.read_csv(path+'/reprocessed_ising_wolf.csv', sep=';')
    _beta=df['beta'].unique()
    _N=df['N'].unique()
    df_save=pd.DataFrame(columns=['n', 'interpol_E', 'interpol_M','beta','J','B','N'])
    for beta in _beta:
        for N in _N:
            tmp_df=df.loc[(df['beta']==beta) & (df['N']==N) & (df['n']>199)] #cuts off the thermalzation of 200 steps
            df_save=pd.concat([df_save, tmp_df])
    df_save.to_csv(path+'/_reprocessed_ising_wolf.csv', sep=';', encoding='utf-8', header=True)
    print(df_save)


if __name__ == '__main__':
    path = sys.argv[1]
    print(path)
    concatenate_csvs(path)
    load_and_cutoff_thermalization(path)
