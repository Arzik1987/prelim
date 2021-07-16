import os
from experiment_dt import *

os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

print(SPLITNS)

import pandas as pd
from multiprocessing import Pool, cpu_count
from itertools import product


def exp_not_done():
    allexp = pd.DataFrame(product(SPLITNS, DNAMES, DSIZES))
    allexp.columns = ['splitn','dname', 'dsize']
    allexp = allexp.astype({'splitn': 'int32', 'dname' : 'string', 'dsize' : 'int32'})
    WHERE = 'registrydt/'
    _, _, filenames = next(os.walk(WHERE))
    terminated = []
    for i in filenames:
        if not "times" in i:
            extra = i.split(".")[0].split("_")
            terminated.append([extra[1], extra[0], extra[2]])
    terminated = pd.DataFrame(terminated, columns=['splitn','dname', 'dsize'])
    terminated = terminated.astype({'splitn': 'int32', 'dname' : 'string', 'dsize' : 'int32'})
    notdone = allexp.merge(terminated, how = 'outer', indicator = True).\
    loc[lambda x : x['_merge'] == 'left_only'][['splitn','dname', 'dsize']]
    print(notdone)
    pool = Pool(32)
    pool.starmap_async(experiment_dt, notdone.itertuples(index = False, name = None))
    pool.close()
    pool.join()

if __name__ == "__main__":
    exp_not_done()

