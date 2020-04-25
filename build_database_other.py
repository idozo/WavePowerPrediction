import random

import pandas as pd
import numpy as np

hours_before = 48
hours_ahead = hours_before
cols = ["VHM0", "VTM10", "VTM02", "VMDR", "VSDX", "VSDY", "VHM0_WW", "VTM01_WW", "VMDR_WW", "VHM0_SW1",
        "VTM01_SW1", "VMDR_SW1", "VHM0_SW2", "VTM01_SW2", "VMDR_SW2", "VPED", "VTPK", "wavepower"]

def build_input_for_year(year):
    df = pd.read_csv("data/" + location + "/" + location + "_" + str(year) + ".csv")
    new_rows = []

    dupcols = []
    for i in range(hours_before, 0, -1):
        for col_name in cols:
            dupcols.append(str(i) + "AGO " + col_name)
    for i in range(1, hours_ahead + 1):
        dupcols.append(str(i) + "AHEAD wavepower")

    for i in range(hours_before, len(df.index) - hours_ahead):
        pre_data = pd.DataFrame(df, index=list(range(i - hours_before, i)), columns=cols)
        ahead_data = pd.DataFrame(df, index=list(range(i, i + hours_ahead)), columns=["wavepower"])
        tmp = pd.DataFrame(np.concatenate(([pre_data.values.flatten()], [ahead_data.values.flatten()]), axis=1), columns=dupcols)
        new_rows.append(tmp)

    new_df = pd.concat(new_rows, axis=0, sort=False)
    new_df.to_csv('data/' + location + "/" + location + "_"+ str(year) + '_input.csv', index=False)


def check_file(year):
    database = pd.read_csv("data/" + location + "/" + location + "_" + str(year) + ".csv")
    input_file = pd.read_csv('data/' + location + "/" + location + "_"+ str(year) + '_input.csv')

    for _ in range(100):
        j = random.randrange(1, hours_before)  # offset AGO to check
        k = random.randrange(1, hours_ahead)  # offset AHEAD to check
        i = random.randrange(0, len(input_file) - k - hours_before)  # row in input to check
        col = random.choice(cols)  # col to check

        if not (np.isclose(database[col][i + hours_before - j], input_file[str(j) + 'AGO ' + col][i])
                and np.isclose(database['wavepower'][i + hours_before - 1 + k], input_file[str(k) + 'AHEAD wavepower'][i])):
            print('\nERROR!! The file was not built properly', i, j, k, col)
            exit(1)


if __name__ == '__main__':
    location = '19-35'
    year = 2017
    print("BUILD " + str(year) + " FILE", end="\t")
    build_input_for_year(year)
    print("- DONE", end="\t")
    check_file(year)
    print('- CHECKED')

