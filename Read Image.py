import cv2
import numpy as np

for i in range(1, 323):
    if i < 10:
        name =  'mdb00' + str(i)+ '.pgm'
    elif i>= 10 and i < 100:
        name =  'mdb0' + str(i)+ '.pgm'
    else:
        name =  'mdb' + str(i)+ '.pgm'

    data = cv2.imread(name,-1) 
    
    output_name = 'im_'+ str(i) + '.npy'
    
    np.save(output_name, data)
