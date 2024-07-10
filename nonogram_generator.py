#!/usr/bin/env python3
"""
Created on Sun Jul  7 16:37:46 2024
"""

import numpy as np
from itertools import product,groupby
from os import chdir
from random import choice

chdir("data")



FILLED = 1
NOT_FILLED = 0

for n in range(1_000):
    
    rows = np.random.randint(5,6)
    columns = np.random.randint(5,6)

    nonogram = np.full((rows,columns),0)

    for i,j in product(range(rows),range(columns)):
        nonogram[i,j] = choice([FILLED,NOT_FILLED])
        
    rows_out = []
    for row in nonogram:
        groups = groupby(row)
        result = [sum(1 for _ in group) for label, group in groups if label == FILLED]
        rows_out.append(result)
        
    columns_out = []
    for column in nonogram.T:
        groups = groupby(row)
        result = [sum(1 for _ in group) for label, group in groups if label == FILLED]
        columns_out.append(result)
    
    with open(f"{n+1}.dat","w") as out:
        print(f'{rows},{columns}',file=out)
        print('-',file=out)
        for row in rows_out:
            print(*row,sep=',',file=out)
        print('-',file=out)
        for col in columns_out:
            print(*col,sep=',',file=out)
        print('-',file=out)
    
    with open(f'{n+1}.target',"w") as out:
        np.savetxt(out,nonogram,fmt="%i")
