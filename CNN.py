# Import Libraries
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, GlobalAveragePooling1D, MaxPooling1D
import matplotlib.pyplot as plt
import time
#from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="3"
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from keras.regularizers import l2
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from sklearn.model_selection import KFold


####################################################################
def One_Hot(y_Train):
    ohe = OneHotEncoder()
    y_Train1 = ohe.fit_transform(y_Train).toarray()
    return y_Train1

####################################################################
def Conv2_2C_1D(X_Train, X_Test, y_Train, y_Test, batch_size, epochs, Num_Class, count):
    start = time.time()
    
    X_Train = X_Train.reshape(X_Train.shape[0], 1024, 1024,1)
    X_Test = X_Test.reshape(X_Test.shape[0], 1024, 1024,1)
    
    
    model = Sequential()
    # Convolution Layer 1 (Filter size 2, Number of filters 128)
    model.add(Conv2D(64, (20, 20), input_shape=(1024, 1024, 1), kernel_initializer='he_uniform', activation='relu'))
    # Max Pooling Layer 1  (Filter size 2, Number of filters 128)
    model.add(MaxPooling2D((10, 10)))
    # Convolution Layer 2  (Filter size 2, Number of filters 64)
    model.add(Conv2D(32, (20, 20), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu'))
    # Max Pooling Layer 2  (Filter size 2, Number of filters 64)
    model.add(MaxPooling2D((10, 10)))

    #model.add(Conv2D(32, (3, 3), input_shape=(X_Train.shape[1],1), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu'))
    #model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    # Fully connected Layer 1, 128 Neurons
    #model.add(Dense(128, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu'))
    # Fully connected Layer 1, 64 Neurons
    model.add(Dense(64, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu'))
    # Fully connected Layer 1, 32 Neurons
    model.add(Dense(32, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu'))
    # Output Layer Number of Activity
    model.add(Dense(Num_Class, activation='softmax'))
    
    model.summary()
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam', metrics=['accuracy'])
    
    y_Train1 = One_Hot(y_Train)
    y_Test1 =  One_Hot(y_Test)
    
    history = model.fit(X_Train, y_Train1, validation_data = (X_Test, y_Test1) ,batch_size = batch_size,
                        epochs = epochs, verbose=1, shuffle = True)
    
    
    
    
    
    
    P1 = 'CNN_Output_Train.png'
    P1_1 = 'CNN_Output_Test.png'
    '''
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.legend(('train', 'test'), loc='lower right')
    plt.savefig(P1_1)
    plt.close()
    '''
    # Predict Test Dataset
    y_Pred = model.predict(X_Test)
    
    
    name = 'y_Test'+str(count)+'.npy'
    np.save(name, y_Test)
    
    name1 = 'y_pred'+str(count)+'.npy'
    np.save(name, y_Pred)
    #############*******************##################     
    # Save Trained Model 
    
    #filename = 'CNN_Model.sav'
    #joblib.dump(model, filename)
    
    #############*******************##################     
    '''
    In prediction vector, we want to find the labels, so 
    we use max of elements in each row 
    index of that element shows the label for that record
    '''
    '''
    y_Pred1 = np.zeros((y_Pred.shape[0], 1))
    for j in range(y_Pred.shape[0]):
        Count = np.argmax(y_Pred[j])
        y_Pred1[j] = Count+1
        
    '''    
    #############*******************##################     
    '''
    Confusion Matrix for Test Dataset
    '''    
    '''
    confusion = confusion_matrix(y_Test,y_Pred1)
    confusion1 = pd.DataFrame(confusion)
    P2 = 'CNN_Confusion'+ str(count)+'.xlsx'
    confusion1.to_excel(P2, header = False, index = False)
    
    
    target_names = ['Class_1', 'Class_2', 'Class_3']
    
    
    A = classification_report(y_Test, y_Pred1, target_names=target_names, output_dict=True)
    Report = pd.DataFrame(A).transpose()
    
    P3 = '1. '+'_CNN_Report_Criteria'+ str(count)+'.xlsx'
    Report.to_excel(P3)
    #############*******************##################
    
    # Write real label and predicted label for test dataset
    y_Test2 = pd.DataFrame(y_Test)
    y_Pred2 = pd.DataFrame(y_Pred1)
    
    P4 = 'CNN_Y_Test_Real'+ str(count)+'.xlsx'
    P5 = 'CNN_Y_Test_Pred'+ str(count)+'.xlsx'
    
    y_Test2.to_excel(P4, header = False, index = False)
    y_Pred2.to_excel(P5, header = False, index = False)
    
    #############*******************##################
    # Calculate Time for each Part and write it 
    end = time.time()
    Time_Taken = end - start
    Time = pd.DataFrame([np.array([Time_Taken])])
    P6 = 'CNN_Time'+ str(count)+'.xlsx'
    Time.to_excel(P6, header = False, index = False)

    '''



#####################################################################################
    
# Number f+of Activity
Num_Class = 3


# Other Hyperparameters for Neural Network

batch_size = 4
epochs = 100
  


Input = np.load('Input_data.npy')
Output = np.load('Output_data.npy')
Output = Output.reshape(-1,1)


kf = KFold(n_splits=10, random_state=None, shuffle=True)



Output_0 = np.where(Output == 0)[0]
Output_1 = np.where(Output == 1)[0]
Output_2 = np.where(Output == 2)[0]


Class_0_X_Train = []
Class_0_X_Test = []
Class_0_Y_Train = []
Class_0_Y_Test = []

for train_index, test_index in kf.split(Output_0):
    Class_0_X_Train.append(Input[Output_0[train_index]])
    Class_0_X_Test.append(Input[Output_0[test_index]])
    Class_0_Y_Train.append(Output[Output_0[train_index]])
    Class_0_Y_Test.append(Output[Output_0[test_index]])
    

Class_1_X_Train = []
Class_1_X_Test = []
Class_1_Y_Train = []
Class_1_Y_Test = []

for train_index, test_index in kf.split(Output_1):
    Class_1_X_Train.append(Input[Output_1[train_index]])
    Class_1_X_Test.append(Input[Output_1[test_index]])
    Class_1_Y_Train.append(Output[Output_1[train_index]])
    Class_1_Y_Test.append(Output[Output_1[test_index]])
    

Class_2_X_Train = []
Class_2_X_Test = []
Class_2_Y_Train = []
Class_2_Y_Test = []

for train_index, test_index in kf.split(Output_2):
    Class_2_X_Train.append(Input[Output_2[train_index]])
    Class_2_X_Test.append(Input[Output_2[test_index]])
    Class_2_Y_Train.append(Output[Output_2[train_index]])
    Class_2_Y_Test.append(Output[Output_2[test_index]])
    

count = 1
for i in range(10):
    
    X_Train = np.concatenate((Class_0_X_Train[i], Class_1_X_Train[i], Class_2_X_Train[i])) 
    X_Test = np.concatenate((Class_0_X_Test[i], Class_1_X_Test[i], Class_2_X_Test[i])) 
    y_Train = np.concatenate((Class_0_Y_Train[i], Class_1_Y_Train[i], Class_2_Y_Train[i])) 
    y_Test = np.concatenate((Class_0_Y_Test[i], Class_1_Y_Test[i], Class_2_Y_Test[i])) 


    print(X_Train.shape, X_Test.shape, y_Train.shape, y_Test.shape)
    Conv2_2C_1D(X_Train, X_Test, y_Train, y_Test, batch_size, epochs, Num_Class, count)
    count += 1


































