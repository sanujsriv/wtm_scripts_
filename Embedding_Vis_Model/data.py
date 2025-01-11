#@title Download Data Function
import gc,torch
from sklearn.feature_extraction.text import CountVectorizer
import math,os
import numpy as np
import nltk
from nltk.corpus import stopwords
from utils import load_obj_pkl5,load_obj,save_obj
from nltk import word_tokenize
# nltk.download('punkt')

if 'grad16' in os.getcwd(): local=True
else: local = False

def download_data(data):
  data = data.lower()
  
  if data=='wos':
    os.system('wget -N https://www.dropbox.com/s/f3qw75jmsdlyk8j/wos_4000_20.zip')
    os.system('unzip wos_4000_20.zip')
  
  elif data=='bbc':
     os.system('wget -N https://www.dropbox.com/s/4mmiaed6rg5lpb1/bbc_2000_20.zip')
     os.system('unzip bbc_2000_20.zip')

  elif data=='searchsnippet':
      os.system('wget -N https://www.dropbox.com/s/prk65wqfgvajhd2/searchsnippet_3000_.zip')
      os.system('unzip searchsnippet_3000_.zip')

  elif data=='agnews120k':
    os.system('wget -N https://www.dropbox.com/s/p7b2msz5fzqvc1q/agnews120k_4000_.zip')
    os.system('unzip agnews120k_4000_.zip')

  elif data=='yahooanswers':
      os.system('wget -N https://www.dropbox.com/s/ggg0ustpn13m7z5/yahooanswers_4000_.zip')
      os.system('unzip yahooanswers_4000_.zip')
  
  else: print('"data" is entered incorrectly ') 


#@title Data Loading Functions

def load_data(data,dtype,cur_dir,skipgram_embeddings):

  data = data.lower()
  dtype = dtype.lower()

  sample_size = 20
  small_data_no_of_ids = 1

  if local==True:
    if dtype == "small":
      dir ='/content/data_'+data+'/'+"full"
    else:
      dir ='/content/data_'+data+'/'+dtype
    os.chdir(cur_dir+dir)

  elif local==False:
    if dtype == "small":
      dir ='/content/content/data_'+data+'/'+"full"
    else:
      dir ='/content/content/data_'+data+'/'+dtype
    os.chdir(dir)

  if dtype=='small':
    data_preprocessed=[]
    data_preprocessed_labels=[]

    for i in range(small_data_no_of_ids):
      id = str(i+1)
      data_preprocessed.append(load_obj_pkl5(id+'_docs_sampled_'+dtype+'_'+str(sample_size)+data+'full'))
      data_preprocessed_labels.append(load_obj_pkl5(id+'_labels_sampled_'+dtype+'_'+str(sample_size)+data+'full'))
    embeddings = load_obj_pkl5('embeddings'+'_'+data+'_'+'full')

  else:
    data_preprocessed=load_obj_pkl5("data_preprocessed_"+data+"_"+dtype)
    data_preprocessed_labels=load_obj_pkl5("data_preprocessed_labels_"+data+"_"+dtype)
    if skipgram_embeddings: embeddings=load_obj_pkl5("generated_embeddings_"+data+"_"+dtype)
    else: embeddings=load_obj_pkl5("embeddings_"+data+"_"+dtype)

  if local==True: os.chdir(cur_dir)
  if local==False: os.chdir('/content/') 
  
  return data_preprocessed,data_preprocessed_labels,embeddings,data
  

def get_data_label_vocab_for_large_data(data,lables,max_features):
  min_df=0
  preprossed_data = data
  train_label = lables
  vectorizer = CountVectorizer(min_df=min_df,max_features=max_features,dtype=np.float32)
  count_vec = vectorizer.fit_transform(preprossed_data)
  vocab = vectorizer.vocabulary_
  id_vocab = dict(map(reversed, vocab.items()))
  print(count_vec.shape,count_vec.__class__.__name__,len(vocab))
  print(vocab)
  train_label = np.asarray(train_label)

  return count_vec,train_label,id_vocab

def get_data_label_vocab_normal(data,lables,max_features):
  min_df=0
  preprossed_data = data
  train_label = lables
  vectorizer = CountVectorizer(min_df=min_df,max_features=max_features,dtype=np.float32)
  train_vec = vectorizer.fit_transform(preprossed_data).toarray()
  vocab = vectorizer.vocabulary_
  id_vocab = dict(map(reversed, vocab.items()))
  print(train_vec.shape,train_vec.__class__.__name__,len(vocab))
  print(vocab)
  
  tensor_train_w = (torch.from_numpy(np.array(train_vec)).float())
  train_label = np.asarray(train_label)

  return tensor_train_w,train_label,id_vocab