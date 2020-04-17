from python_speech_features import mfcc
import glob
from scipy.io import wavfile
import numpy as np
import pickle
import copy
import os

def mfcc_to_pickle():

    #reading wave files from path and adding to list
    file_list = glob.glob("D:/Inne Projekty z Programowania/TMProject1/train/*.wav")
    only_name = []
    for filename in file_list:
        temp = filename.split("\\")
        only_name.append(temp[1])

    #iterating through list
    mfcc_file_dict = {}
    for wave_name in only_name:
        fs, data = wavfile.read("D:/Inne Projekty z Programowania/TMProject1/train/"+wave_name)
        #creating MFCC matrix

        temp_name = wave_name.split('.')#cutting off extension
        mfcc_matrix = mfcc(signal=data, samplerate=fs)
        mfcc_file_dict[temp_name[0]] = mfcc_matrix

    #serialisation and saving

    pickle_out = open('train.pickle', 'wb')
    pickle.dump(mfcc_file_dict, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
    pickle_out.close()

def load_mfcc_from_pickle():
    #loading training data
    pickle_in = open('train.pickle', 'rb')
    mfcc_file_dict = pickle.load(pickle_in)
    #print(mfcc_file_dict['AO1M1_0_'])

    #Spliting on smaller boards

    #key=number value=list of mfcc
    samples_dict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}



    for key in mfcc_file_dict.keys():
        number = key.split('_')[1]
        samples_dict[int(number)].append(mfcc_file_dict[key])
    return samples_dict

#mfcc_to_pickle()
main_dict=load_mfcc_from_pickle()
#keras model :)
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization

NB_OUTPUTS=10
NB_EPOCH = 250
BATCH_SIZE =70
OPTIMIZER = Adam() # optimizer
N_HIDDEN = 128

model=Sequential()
#add input layer with 32 neuron an 13 inputs with randomised weights at start with range from -0.05 to 0,05
model.add(Dense(32, input_shape=(13,), kernel_initializer='random_uniform'))
model.add(BatchNormalization(momentum=0.01))
model.add(Activation('relu'))
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
#model.add(Dense(N_HIDDEN))
#model.add(Activation('relu'))
model.add(Dense(NB_OUTPUTS))
model.add(Activation('softmax'))
model.summary()
#compiling model so it can be executed by Tensorflow backend
#We use Adam optimiser and categorical_crossentropy as an objective function
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

def split_speakers(dictionary, number):
    test_dict={0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    fit_dict = copy.deepcopy(dictionary)
    for x in range(10):
        test_dict[x]=fit_dict[x][4*number:4*number+4]
        k=4*number
        del fit_dict[x][k:k+4]
    return fit_dict, test_dict

for test_group in range(5):
    XTRAIN=[]
    YTRAIN=[]

    XTEST=[]
    YTEST=[]
    fit_d,test_d=split_speakers(main_dict,test_group)
    for fit_number in range(10):
        for speakers in range(18):
            fitData=fit_d[fit_number][speakers]
            for k in range(len(fitData)):
                #print(fitData[k])
                XTRAIN.append(fitData[k].tolist())
                #print(XTRAIN)
                y=[]
                for s in range(10):
                    if s==fit_number:
                        y.append(1)
                    else:
                        y.append(0)
                YTRAIN.append(y)
    XTRAIN=np.array(XTRAIN)
    YTRAIN=np.array(YTRAIN)
    #print(np.shape(XTRAIN))
    #print(np.shape(YTRAIN))
    for fit_number in range(10):
        accuracy_total = np.zeros((4,10)) # dla calosci
        for speakers in range(4):
            testData=test_d[fit_number][speakers]
            for k in range(len(testData)):
                XTEST.append(testData[k].tolist())
                y=[]
                for s in range(10):
                    if s==fit_number:
                        y.append(1)
                    else:
                        y.append(0)
                YTEST.append(y)
    XTEST=np.array(XTEST)
    YTEST=np.array(YTEST)
    #print(np.shape(XTEST))
    #print(np.shape(YTEST))
    model.fit(XTRAIN,YTRAIN,verbose=0)
    score = model.evaluate(XTEST, YTEST, verbose=0)
    print("Test score:", score[0])
    print('Test accuracy:', score[1])
    print("\n")
    #wait=input()