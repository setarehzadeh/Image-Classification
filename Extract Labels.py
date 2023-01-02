import numpy as np
import pandas as pd
import pathlib

data = pd.read_excel('Output.xlsx')

Labels = []


for i in range(len(data)):
    
    if data.loc[i, 'CLASS'] == 'NORM' :
        Labels.append(0)
    else:
        if data.loc[i, 'SEVERITY'] == 'B' :
            Labels.append(1)
        elif  data.loc[i, 'SEVERITY'] == 'M' :
            Labels.append(2)
        else:
            pass
   

Labels = np.array(Labels)

np.save('Output_data.npy', Labels)
