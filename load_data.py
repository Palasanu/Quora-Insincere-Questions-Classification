#!/usr/bin/env python
# coding: utf-8

# In[70]:


import os
import random
import pickle
import zipfile
from tqdm import tqdm
import tensorflow as tf      #the progress bar
import en_core_web_sm as en  #from the spaCy library, https://spacy.io/usage/


# In[16]:


filename = './data/text8.zip'


# In[20]:


def read_data(filename):
  with zipfile.ZipFile(filename) as f:
    text = tf.compat.as_str(f.read(f.namelist()[0]))
  return text

text = read_data(filename)
print("Len of data:", len(text))


# In[55]:


def tokenize_text(text, nr_to_delete = 10):
    nlp = en.load()      #load the tokenizer
    
    batch_lenght = nlp.max_length
    nlp.max_length += 1
    text_tokenized = []  #every token in the original order
    tokens_count = {}    #a dict with everry word with his count
    vocab = []           #every unique word in the appearance order  

    batches = [text[x:x+batch_lenght] for x in range(0, len(text), batch_lenght)]  #place the text in n batchez of size batch_lenght

    del text  #delete the original text so we save memory

    for batch in tqdm(batches):  #tqdm is the progress bar
        tokens = nlp(batch)         
        for token in tokens:
            text_tokenized.append(token.string.strip())
            if text_tokenized[-1] in tokens_count:
                tokens_count[text_tokenized[-1]] += 1
            else:
                tokens_count[text_tokenized[-1]] = 1

    vocab = sorted(tokens_count, key=tokens_count.__getitem__) #get a list with the keys of the dict sorted by the nr of appearances
    
    sum = 0
    unk = vocab[:nr_to_delete]   #save a nr of words with the least appearances 
    for el in unk:               # delete the saved words and sum their appearence in sum
        sum += tokens_count[el]
        del tokens_count[el]
    tokens_count['UNK'] = sum    #replace the words with the "UNK" key
    
    vocab = sorted(tokens_count, key=tokens_count.__getitem__)  #resort he dict so we get the vocab without the words with the least appearences
    
    for idx,word in enumerate(text_tokenized): #replace the saved words with the "UNK" in the text
        for w in unk:
            if w == word:
                text_tokenized[idx] = 'UNK'             
    
    return text_tokenized,vocab       
    
tokenized_text, vocab = tokenize_text(text[:2500000])    


# In[57]:


def tokens_to_index(tokenized_text,vocab):
    word_to_index = {}
    index_to_word = {}
    
    for idx, word in enumerate(vocab): #create the two dictionaries
        word_to_index[word] = idx
        index_to_word[idx] = word
    
    for idx,token in enumerate(tokenized_text): #replace the text with the word indexes
        tokenized_text[idx] = word_to_index[token]
    
    return tokenized_text, word_to_index, index_to_word 
tokenized_text, word_to_index, index_to_word = tokens_to_index(tokenized_text,vocab)


# In[58]:


tokenized_text[:10]


# In[66]:


def idx_to_text(tokenized_text):
    str_text = ''
    for index in tokenized_text:
        str_text += str(index_to_word[index])+' '
    return str_text


# In[68]:


idx_to_text(tokenized_text[:10])


# In[ ]:




