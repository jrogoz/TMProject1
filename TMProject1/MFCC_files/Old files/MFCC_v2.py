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
for k in range(10):
    print(len(main_dict[k]))

#keras model :)
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.utils import to_categorical

NB_OUTPUTS=10
NB_EPOCH = 20
BATCH_SIZE = 15
OPTIMIZER = Adam() # optimizer
#N_HIDDEN = 128

model=Sequential()
#add input layer with 13 neuron an 13 inputs with randomised weights at start with range from -0.05 to 0,05
model.add(Dense(13, input_shape=(13,), kernel_initializer='random_uniform'))
model.add(Activation('relu'))
model.add(Dense(NB_OUTPUTS))
model.add(Activation('softmax'))
model.summary()
#compiling model so it can be executed by Tensorflow backend
#We use Adam optimiser and categorical_crossentropy as an objective function
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

def split_speakers(dictionary, number):
    test_dict={0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    #fit_dict={0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    fit_dict = copy.deepcopy(dictionary)
    print("Fit size: "+str(len(fit_dict[0])))
    for x in range(10):
        test_dict[x]=fit_dict[x][4*number:4*number+4]
        k=4*number

        del fit_dict[x][k:k+4]

       #print("fit dict size: " + str(len(fit_dict[x])))
        # for z in range(4):
        #     print("k+z:"+str(k+z))
        #     print("fit dict size: "+str(len(fit_dict[x])))
        #     del fit_dict[x][k+z]
    print("Group Number: "+str(number))
    print("fit dict len for 0: "+str(len(fit_dict[0])))
    print("test dict len for 0: " + str(len(test_dict[0])))
    return fit_dict, test_dict

#n1,n2=split_speakers(main_dict,2)

for test_group in range(5):
    fit_d,test_d=split_speakers(main_dict,test_group)
    for fit_number in range(10):
        for speakers in range(18):
            #print("Test group: "+str(test_group))
            #print("Fit number: "+str(fit_number))
            #print("Speaker number (should be 18): "+str(len(fit_d[fit_number])))
            fitData=np.array(fit_d[fit_number][speakers])
            m,n=np.shape(fitData)
            #print("m: "+str(m))
            y_train =np.zeros((m,10))
            for k in range(m):
                y_train[k,fit_number]=1
            print(np.ndim(y_train))
            #y_train = to_categorical(y_train)
            model.fit(fitData,y_train)
    for fit_number in range(10):
        for speakers in range(4):
            testData=np.array(test_d[fit_number][speakers])
            m,n=np.shape(testData)
            y_test=np.zeros((m,10))
            for k in range(m):
                y_test[k,fit_number]=1
            score = model.evaluate(testData, y_test, verbose=1)
            print("\n")
            print("Test group: "+str(test_group+1)+" number: "+str(fit_number)+" speaker: "+str(speakers))
            print("Test score:", score[0])
            print('Test accuracy:', score[1])
            print("\n")
    clear = lambda: os.system('cls')
