import numpy as np
import pandas as pd

All_Data = []
for i in range(322):
    print(i)
    name = 'im_' + str(i+1)+'.npy'
    data = np.load(name)
    All_Data.append([data])
    
    
All_Data = np.array(All_Data)
All_Data = All_Data.reshape(322,1024,1024)
np.save('Input_data.npy', All_Data)


#All_Data = np.array(All_Data)
'''    
All_Data = pd.concat([All_Data, pd.DataFrame(data)])
A = np.array(All_Data)
A = A.reshape(1024, 1024, 322)
np.save('Input_data.npy', A)
'''