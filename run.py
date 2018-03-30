#!/usr/bin/env python2.7

import numpy as np
import re
import itertools
from collections import Counter
import csv

from keras.models import Model
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.optimizers import Adam


train_FP = "/Users/calvin-is-seksy/Desktop/myProjects/CS291A/data/liar_dataset/train.tsv"
valid_FP = "/Users/calvin-is-seksy/Desktop/myProjects/CS291A/data/liar_dataset/valid.tsv"
test_FP = "/Users/calvin-is-seksy/Desktop/myProjects/CS291A/data/liar_dataset/test.tsv"


def clean(my_str):
	# function taken from Yoon Kim's CNN for Sentence Classification. github: https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    my_str = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", my_str)
    my_str = re.sub(r"\'s", " \'s", my_str)
    my_str = re.sub(r"\'ve", " \'ve", my_str)
    my_str = re.sub(r"n\'t", " n\'t", my_str)
    my_str = re.sub(r"\'re", " \'re", my_str)
    my_str = re.sub(r"\'d", " \'d", my_str)
    my_str = re.sub(r"\'ll", " \'ll", my_str)
    my_str = re.sub(r",", " , ", my_str)
    my_str = re.sub(r"!", " ! ", my_str)
    my_str = re.sub(r"\(", " \( ", my_str)
    my_str = re.sub(r"\)", " \) ", my_str)
    my_str = re.sub(r"\?", " \? ", my_str)
    my_str = re.sub(r"\s{2,}", " ", my_str)
    return my_str.strip().lower()


def file_to_train_data(filename):
    labels = []
    statements = []
    with open(filename, "r") as infile: 
        LoL = [x.strip().split('\t') for x in infile]

    for row in LoL: 
        label = row[0] # label 
        statement = row[1] # statment 
        statements.append(statement)
        
        if label == "pants-fire":
            labels.append([1,0,0,0,0,0])
        elif label == "false":
            labels.append([0,1,0,0,0,0])
        elif label == "barely-true":
            labels.append([0,0,1,0,0,0])
        elif label == "half-true":
            labels.append([0,0,0,1,0,0])
        elif label == "mostly-true":
            labels.append([0,0,0,0,1,0])
        elif label == "true":
            labels.append([0,0,0,0,0,1])
        else: 
            print "invalid label"
    
    data_words = [clean(sentence) for sentence in statements]
    data_words = [sentence.split(" ") for sentence in data_words]
    
    return [data_words, labels]


def file_to_test_data(filename):
    statements = []
    with open(filename, "rb") as infile: 
        LoL = [x.strip().split('\t') for x in infile]
    
    for row in LoL: 
        statement = row[0]
        statements.append(statement)
    
    data_words = [clean(sentence) for sentence in statements]
    data_words = [sentence.split(" ") for sentence in data_words]
    
    unknown_labels = [[0] for x in range(len(statement)) ]
    return [data_words, unknown_labels]


def load_train(filename): # for both training set and validation set 
    sentences, labels = file_to_train_data(filename)

    max_len_of_rows = 489
    sentences_padded = []
    for i in range(len(sentences)):
        row = sentences[i]
        pad_nums = max_len_of_rows - len(row)
        new_row = row + ["<PAD/>"] * pad_nums
        sentences_padded.append(new_row)

    word_counts = Counter(itertools.chain(*sentences_padded))
    vocab_pop_to_not = [word[0] for word in word_counts.most_common()]
    vocab_pop_to_not = list(sorted(vocab_pop_to_not))
    vocab = {word: index for index, word in enumerate(vocab_pop_to_not)}

    x = np.array([[vocab[word] for word in sentence] for sentence in sentences_padded])
    y = np.array(labels)

    return [x, y, vocab, vocab_pop_to_not]

def load_test(filename): # for test set 
    sentences, labels = file_to_test_data(filename)

    max_len_of_rows = 489
    sentences_padded = []
    for i in range(len(sentences)):
        row = sentences[i]
        pad_nums = max_len_of_rows - len(row)
        new_row = row + ["<PAD/>"] * pad_nums
        sentences_padded.append(new_row)

    word_counts = Counter(itertools.chain(*sentences_padded))
    vocab_pop_to_not = [word[0] for word in word_counts.most_common()]
    vocab_pop_to_not = list(sorted(vocab_pop_to_not))
    vocab = {word: index for index, word in enumerate(vocab_pop_to_not)}

    x = np.array([[vocab[word] for word in sentence] for sentence in sentences_padded])
    y = np.array(labels)

    return [x, y, vocab, vocab_pop_to_not]

def run(): 
	X_train, y_train, train_vocab, train_vocab_pop_to_not = load_train(train_FP)	
	X_valid,y_valid, valid_vocab, valid_vocab_pop_to_not = load_train(valid_FP)	
	X_test, y_test, _, _ = load_test(test_FP)							 

	row_len = X_valid.shape[1]
	vocab_size = len(train_vocab_pop_to_not) + len(valid_vocab_pop_to_not) # ??? Add validations size too? 
	embedding_dim = 256
	filters = [3,4,5]
	n_filters = 512 
	d = 0.5 
	epo = 1
	batch_size = 20 
	LR = 1e-4

	print("Creating Model...")
	# model 1: CNN for statements 
	model_1_inputs = Input(shape=(row_len,), dtype='int32')
	embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=row_len)(model_1_inputs)
	reshape = Reshape((row_len,embedding_dim,1))(embedding)

	conv_0 = Conv2D(n_filters, kernel_size=(filters[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
	conv_1 = Conv2D(n_filters, kernel_size=(filters[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
	conv_2 = Conv2D(n_filters, kernel_size=(filters[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

	maxpool_0 = MaxPool2D(pool_size=(row_len - filters[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
	maxpool_1 = MaxPool2D(pool_size=(row_len - filters[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
	maxpool_2 = MaxPool2D(pool_size=(row_len - filters[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

	concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
	flatten = Flatten()(concatenated_tensor)
	dropout = Dropout(d)(flatten)
	model_1_output = Dense(units=6, activation='softmax')(dropout)

	model_1 = Model(inputs=model_1_inputs, outputs=model_1_output)

	ckpt = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
	# adam = Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	adam = Adam(lr=LR, epsilon=1e-8)

	model_1.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
	print("Training Model...")

	model_1.fit(X_train, y_train, batch_size=batch_size, epochs=epo, verbose=1, callbacks=[ckpt], validation_data=(X_valid, y_valid))

	raw_model_output = model_1.predict(X_test, batch_size=batch_size, verbose=1) 

	output_labels = ["pants-fire", "false", "barely-true", 
	                 "half-true", "mostly-true", "true"]

	with open('predictions.txt', 'w') as outfile: 
		for i in range(len(raw_model_output)):
		    temp_raw_model_output = raw_model_output[i]
		    output_label_index = np.nonzero(temp_raw_model_output == max(temp_raw_model_output))[0][0]
		    outfile.write(output_labels[output_label_index] + "\n")
		    print output_labels[output_label_index]

run()


















