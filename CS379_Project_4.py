###########################
#Adam Mick CS379 Project 4#
###########################
###########################################################################################
#References & Sources                                                                     #
#Text Generation with LSTM Recurrent Neural Networks in Python with Keras - Jason Brownlee#
#Text Generation with an RNN - Tensorflow tutorial                                        #
#Keras API reference - Keras website                                                      #
#The above sources were all heavily referenced in the creation of this project            #
###########################################################################################

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils



def main():

    #########################
    #load & process the data#
    #########################

    text = open("alice_in_wonderland.txt", 'r', encoding = 'utf-8').read() #opens and assigns the book text
    text = text.lower()#converts all characters to lowercase for processing.

    character = sorted(list(set(text))) #create a list of each character in the data
    char_to_int = dict((c, i) for i, c in enumerate(character)) #assing each character to a dictionary with an associated number for processing
    int_to_char = dict((i, c) for i, c in enumerate(character)) #create a reverse map of ints to characters 

    character_total = len(text)
    total_words = len(character)

    print("The total character count is: ", character_total)
    print("The number of words in the vocabulary is:", total_words)



    ########################
    #creating training data#
    ########################

    length = 150 #this is the length of data that will be used to create sequences of text for it to look at. 
    dataX = []
    dataY = []
    for i in range(0, character_total - length, 1):
        seq_in = text[i:i + length]
        seq_out = text[i + length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])

    n_patterns = len(dataX)

    print("Total patterns: ", n_patterns)


    ##############################
    #Formatting the training data#
    ##############################

    X = np.reshape(dataX, (n_patterns, length, 1)) #This converts the data to a formatting to fit the keras LSTM model 
    X = X / float(total_words)
    Y = np_utils.to_categorical(dataY)

    ########################
    #creating the algorithm#
    #########################

    seqModel = Sequential()
    seqModel.add(LSTM(256, input_shape = (X.shape[1], X.shape[2])))
    seqModel.add(Dropout(0.2))
    seqModel.add(Dense(Y.shape[1], activation = 'softmax'))
    seqModel.compile(loss='categorical_crossentropy', optimizer='adam')
    
    # define the checkpoint
    filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    seqModel.fit(X, Y, epochs=20, batch_size=128, callbacks=callbacks_list)

    #################
    #Predicting text#
    #################

    weightFile = "weights-improvement-19-1.9435.hdf5"

    seqModel.load_weights(weightFile)
    seqModel.compile(loss = 'categorical_crossentropy', optimizar = 'adam')

    #pick a random seed
    start = numpy.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    print "Seed:"
    print "\"", ''.join([int_to_char[value] for value in pattern]), "\""
    #generate characters
    for i in range(1000):
	    x = numpy.reshape(pattern, (1, len(pattern), 1))
	    x = x / float(n_vocab)
	    prediction = model.predict(x, verbose=0)
	    index = numpy.argmax(prediction)
	    result = int_to_char[index]
	    seq_in = [int_to_char[value] for value in pattern]
	    sys.stdout.write(result)
	    pattern.append(index)
	    pattern = pattern[1:len(pattern)]
    print "\nDone."

    return


main()
