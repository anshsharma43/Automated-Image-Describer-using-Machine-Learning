#!/usr/bin/env python
# coding: utf-8

# In[52]:


import numpy as np
# from numpy import array
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

import string
import os
import glob
from PIL import Image
from time import time
import collections
import random
import numpy as np
import json
import pickle
import glob
from gtts import gTTS
from playsound import playsound


# from keras import Input, layers
# from keras import optimizers
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.layers import LSTM, Embedding, Dense, Activation, Flatten, Reshape, Dropout
# from tensorflow.keras.layers import Bidirectional
# from tensorflow.keras.layers import add
from tensorflow.keras.applications.inception_v3 import InceptionV3
# from tensorflow.keras.applications.vgg19 import VGG19
#from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

# from tqdm import tqdm


# In[53]:


model_temp = InceptionV3(weights='imagenet')
model_new = Model(model_temp.input, model_temp.layers[-2].output)


# In[54]:


# model = InceptionV3(weights='imagenet')
# model_new = Model(model.input, model.layers[-2].output)


# In[55]:


from tensorflow.keras.utils import load_img,img_to_array
def preprocess(image_path):
    img = load_img(image_path, target_size=(299, 299))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


# In[56]:


def encode(image):
    image = preprocess(image) 
    fea_vec = model_new.predict(image) 
    print(fea_vec.shape)
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    print(fea_vec.shape)
    return fea_vec


# In[ ]:





# In[57]:


from tensorflow.keras.models import load_model
model=load_model('model-input/image_describer_40 (2).h5')
# model.summary()


# In[58]:


# glove_path = '../input/glove6b/glove.6B.200d.txt'
# embeddings_index = {} 
# f = open(glove_path, encoding="utf-8")
# for line in f:   
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs


# In[59]:


# import json

# # Load JSON string from file
# with open('model-input/train_descriptions (1).json', 'r') as f:
#     json_train_desc = f.read()

# # Convert JSON string to dictionary
# train_descriptions = json.loads(json_train_desc)

# # Get all captions from dictionary
# all_train_captions = []
# for key, val in train_descriptions.items():
#     for cap in val:
#         all_train_captions.append(cap)


# In[60]:


# print(len(all_train_captions)) # contain all the captions   5*11858 = 59173
# print(all_train_captions[:5])


# In[61]:


# word_count_threshold = 10
# word_counts = {}
# nsents = 0
# for sent in all_train_captions:
#     nsents += 1
#     for w in sent.split(' '):
#         word_counts[w] = word_counts.get(w, 0) + 1
# vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

# print('Vocabulary = %d' % (len(vocab)))


# In[62]:


# import json

# # Assuming that you have already created the ixtoword and wordtoix dictionaries
# # ...

# # Save ixtoword dictionary to file
# with open('ixtoword.json', 'w') as f:
#     json.dump(ixtoword, f)

# # Save wordtoix dictionary to file
# with open('wordtoix.json', 'w') as f:
#     json.dump(wordtoix, f)


# In[63]:


# # Load ixtoword dictionary from file
# with open('ixtoword.json', 'r') as f:
#     ixtoword = json.load(f)

# # Load wordtoix dictionary from file
# with open('wordtoix.json', 'r') as f:
#     wordtoix = json.load(f)


# In[64]:





# In[65]:


# all_desc = list()
# for key in train_descriptions.keys():
#     [all_desc.append(d) for d in train_descriptions[key]]
# lines = all_desc
# max_length = max(len(d.split()) for d in lines)
max_length=51
print('Description Length: %d' % max_length)


# In[66]:


# import pickle

# # Assuming that you have already created the vocab list
# # ...

# # Save vocab list to file
# with open('vocab.pkl', 'wb') as f:
#     pickle.dump(vocab, f)



# In[67]:


# Load vocab list from file
with open('model-input/vocabulary_40 (1).pkl', 'rb') as f:
    vocab = pickle.load(f)


# In[68]:


ixtoword = {}
wordtoix = {}
ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1


vocab_size = len(ixtoword) + 1
print(vocab_size)
# In[69]:


# embedding_dim = 200
# embedding_matrix = np.zeros((vocab_size, embedding_dim))
# for word, i in wordtoix.items():
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         embedding_matrix[i] = embedding_vector


# In[70]:


# model.layers[2].set_weights([embedding_matrix])
# model.layers[2].trainable = False
# model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])


# In[71]:


def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break

    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


# In[ ]:





# In[72]:


def beam_search_predictions(image, beam_index = 3):
    start = [wordtoix["startseq"]]
    start_word = [[start, 0.0]]
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_length, padding='post')
            preds = model.predict([image,par_caps], verbose=0)
            word_preds = np.argsort(preds[0])[-beam_index:]
            # Getting the top <beam_index>(n) predictions and creating a 
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [ixtoword[i] for i in start_word]
    final_caption = []
    
    
    
    for i in intermediate_caption:
        if i != 'endseq':
            final_caption.append(i)
        else:
            break

    final_caption = ' '.join(final_caption[1:])
    return final_caption


# In[73]:


def generate_caption(image_path):
    image = encode(image_path)
    image = image.reshape((1, 2048))
    #x=plt.imread(image_path)
    #plt.imshow(x)
    #plt.show()
    pred_caption=beam_search_predictions(image, beam_index = 7)
    speech = gTTS('Predicted Caption : ' + pred_caption, lang = 'en', slow = False)
    speech.save('static/voice.mp3')
    audio_file = 'voice.mp3'
    return pred_caption

#     print("Greedy:",greedySearch(image))
#     print("Beam Search, K = 3:",beam_search_predictions(image, beam_index = 3))
#     print("Beam Search, K = 5:",beam_search_predictions(image, beam_index = 5))
      

# In[74]:


# generate_caption('coco2017/train2017/000000000009.jpg')


# In[76]:


#generate_caption('coco2017/train2017/000000000030.jpg')


# In[ ]:




