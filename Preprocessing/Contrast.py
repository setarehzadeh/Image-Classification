from PIL import Image, ImageEnhance
import cv2
import numpy as np


for i in range(1, 323):
    if i < 10:
        name =  'mdb00' + str(i)+ '.pgm'
    elif i>= 10 and i < 100:
        name =  'mdb0' + str(i)+ '.pgm'
    else:
        name =  'mdb' + str(i)+ '.pgm'

    #data = cv2.imread(name,-1) 
    data =  Image.open(name)
    
    enhancer = ImageEnhance.Contrast(data)
    factor = 1.25 #increase contrast
    im_output = enhancer.enhance(factor)
    
    name_output = 'im_'+str(i)+'.png'
    im_output.save(name_output)
        
    
'''    
    
#read the image
im = Image.open("mdb001.pgm")

#image brightness enhancer
enhancer = ImageEnhance.Contrast(im)

factor = 1 #gives original image
im_output = enhancer.enhance(factor)
im_output.save('original-image.png')

factor = 0.5 #decrease constrast
im_output = enhancer.enhance(factor)
im_output.save('less-contrast-image.png')

factor = 1.25 #increase contrast
im_output = enhancer.enhance(factor)
im_output.save('more-contrast-image.png')





    
    output_name = 'im_'+ str(i) + '.npy'
    
    np.save(output_name, data)
'''