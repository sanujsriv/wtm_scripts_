import seaborn as sb
import numpy as np  
from sklearn.neighbors import KNeighborsClassifier
import torch.nn as nn
import pickle5
import pickle
import torch
import math
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import matplotlib.pyplot as plt
from time import time
import gc
# nltk.download('punkt')

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_obj_pkl5(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle5.load(f)

  
def doc_remove_extra_stopwords(doc):
  extra_stopwords = ['la','wa','will','fa','ha','pa','co','v','said']
  doc = word_tokenize(doc)
  doc = filter(lambda x:x not in extra_stopwords, doc)
  doc = ' '.join(e for e in doc)
  return doc

cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-8)

def flatten_list(user_list): return [item for sublist in user_list for item in sublist]
def get_embedding_tensor(word_list,embeddings): return torch.tensor([embeddings[w] for w in word_list])
def get_doc_word_embeddings(id_vocab,embeddings):
  sorted_id_word_vocab = sorted(id_vocab.items(), key=lambda x: x[1]) ### alphabetically sorted
  word_list = [s[1] for s in sorted_id_word_vocab]
  words_tensor = get_embedding_tensor(word_list,embeddings)
  embedding_tensor_sorted_alp = get_embedding_tensor(word_list,embeddings)
  emb_size = embedding_tensor_sorted_alp.shape[1]
  return embedding_tensor_sorted_alp,emb_size

def get_topwords(beta,id_vocab,no_of_topwords):
    topic_indx = 0
    topwords_topic = []
    for i in range(len(beta)):      
        topwords_topic.append(str(topic_indx)+": "+ " ".join([id_vocab[j] for j in beta[i].argsort()[:-no_of_topwords - 1:-1]]))
        topic_indx+=1
    return topwords_topic

import plotly.graph_objects as go

def plot_loss(y,name):
  figure = go.Figure()
  figure.add_trace(go.Scatter(x=[i for i in range(1,epochs+1)], y=y,mode='lines',name=name))
  figure.show(renderer='colab')


def plot_fig(zx, labels_list, zphi,lim,contour='No'):
    labels = []
    for i in range(len(labels_list)):
        labels.append('C'+str(labels_list[i]))
    fig, ax = plt.subplots( figsize=(20, 20))
    if contour=='yes':
       get_Contour(ax,zx,lim)
    
    sb.scatterplot(ax=ax,x=zx[:,0],y=zx[:,1],hue=labels_list,alpha=0.8,palette='deep')
    ax.set(ylim=(-lim,lim))
    ax.set(xlim=(-lim,lim))
    ax.text(0,0, 'X' ,c='black')
    ax.scatter(zphi[:, 0], zphi[:, 1], alpha=1.0,  edgecolors='black', facecolors='none', s=30)
   
    for indx, topic in enumerate(zphi):
        ax.text(zphi[indx, 0], zphi[indx, 1], 'topic'+str(indx))
  
def print_Topics(beta,id_vocab,no_of_topwords):
  print("---"*10)
  topword_topics = get_topwords(beta, id_vocab,no_of_topwords)
  topword_topics_list=[]
  for topwords in topword_topics:
      topword_topics_list.append(topwords.split())
      print(topwords)
  print("---"*10)

def cal_knn(coordinate, label):
    output = []
    for n_neighbors in [10, 20, 30, 40, 50]:
        neigh = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        neigh.fit(coordinate, label)
        output.append(neigh.score(coordinate, label))
    return output

def getall_tensor_size(): 
  for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            print(type(obj), obj.size(),get_mem_size(obj)) 
    except:
        pass