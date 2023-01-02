import pandas as pd
import numpy as np
import random

def rotate_matrix(mat):
    rand = random.randint(1,3)
    for i in range(rand):
        mat = np.rot90(mat)
    return mat


data = np.load('Output_data.npy')

Class_1 = np.where(data == 0)[0]
Class_2 = np.where(data == 1)[0]
Class_3 = np.where(data == 2)[0]
New_Output_For_Balance = []

##### Class 2
Len_2 = len(Class_1)-len(Class_2)
index_2 = []
for i in range(Len_2):
    index_2.append(random.randint(0, len(Class_2)-1))
    
for i in range(len(index_2)):
    New_Output_For_Balance.append(1)


##### Class 3
Len_3 = len(Class_1)-len(Class_3)
index_3 = []
for i in range(Len_3):
    index_3.append(random.randint(0, len(Class_3)-1))

for i in range(len(index_3)):
    New_Output_For_Balance.append(2)

New_Output_For_Balance = np.array(New_Output_For_Balance)
Balanced_Output = np.concatenate((data, New_Output_For_Balance))
np.save('Balanced_Output.npy', Balanced_Output)


Input_data = np.load('Input_data.npy')

All_Data_Class_2 = []
for i in range(len(index_2)):
    name = 'im_' + str(Class_2[index_2[i]]+1)+'.npy'
    data1 = np.load(name)
    data1 = rotate_matrix(data1) 
    All_Data_Class_2.append([data1])
    
A_2 = np.array(All_Data_Class_2)
A_2 = A_2.reshape(len(index_2), 1024, 1024)

Balanced_Input = np.concatenate((Input_data, A_2))


All_Data_Class_3 = []
for i in range(len(index_3)):
    name = 'im_' + str(Class_3[index_3[i]]+1)+'.npy'
    data2 = np.load(name)
    data2 = rotate_matrix(data2)
    All_Data_Class_3.append([data2])
    
A_3 = np.array(All_Data_Class_3)
A_3 = A_3.reshape(len(index_3), 1024, 1024)    

Balanced_Input = np.concatenate((Balanced_Input, A_3))
np.save('Balanced_Input.npy', Balanced_Input)





