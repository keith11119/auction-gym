import pandas as pd
import os

def split_csv(file, save_dir, row_limit = 100):
    for i, chunk in enumerate(pd.read_csv(file, chunksize=row_limit, header=None, skip_blank_lines=False)):
        chunk.to_csv(f'{save_dir}{i}.csv', index=False, header=False)

def split_data(estimator, competition, data_set, agent_num, row_num, data_dir=None, special=''):
    if data_dir is None:
        data_dir = f"../data_{estimator}_{competition}/"
    data_file = f"{data_dir}{data_set}_{agent_num}.csv"
    save_dir = f"../data_{estimator}_{competition}_split_{row_num}{special}/agent_{agent_num}/{data_set}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    split_csv(data_file, save_dir, row_limit=row_num)

if __name__ == "__main__":
    agent_num = 0
    estimator = "DR"
    competition = "L"
    data_set = "test"
    row_num = 100
    split_data(estimator=estimator, competition=competition, data_set=data_set, agent_num=agent_num, row_num=100)