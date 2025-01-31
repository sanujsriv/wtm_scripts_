import argparse
import torch
import os
import numpy as np
from data import download_data,load_data,get_data_label_vocab_for_large_data,get_data_label_vocab_normal
from utils import load_obj_pkl5,load_obj,save_obj,get_doc_word_embeddings,get_topwords,cal_knn,print_Topics,plot_fig
from wtm import Model
# from wtm_vd import ModelVD as vd_wtm_Model
from train_evaluation import train_for_large,test_for_large,train,test

import pickle5
import bz2
import pickle
import _pickle as cPickle
from time import time

def compressed_pickle(data,title):
  with bz2.BZ2File(title + '.pbz2', 'w') as f:
    cPickle.dump(data, f)

def decompress_pickle(file):
 data = bz2.BZ2File(file+".pbz2", 'rb')
 data = cPickle.load(data)
 return data

parser = argparse.ArgumentParser(description='Word Embeddings Topic Model (WTM)')

## data arguments
parser.add_argument('-d','--dataset', type=str, default='bbc', help='name of corpus')
parser.add_argument('-dt','--dtype', type=str, default='short', help='short/full/small corpus')
parser.add_argument('-path','--data_path', type=str, default='./content', help='directory containing data')
parser.add_argument('-bs','--batch_size', type=int, default=256, help='batch size for training')
parser.add_argument('-r','--run', type=int, default=1, help='run')

## model arguments
parser.add_argument('-k','--num_topics', type=int, default=10, help='number of topics')
parser.add_argument('-sg_emb','--skipgram_embeddings', type=int, default=1, help='whether use of skipgram embeddings or any other embeddings')
parser.add_argument('-emb_sz','--emb_size', type=int, default=300, help='dimension of embeddings')
parser.add_argument('-h1','--hidden1', type=int, default=100, help='dim of hidden layer 1')
parser.add_argument('-h2','--hidden2', type=int, default=100, help='dim of hidden layer 2')
parser.add_argument('-varx', '--variance_x', type=float, default=1.0, help='variance of x {doc coord}')
parser.add_argument('-s','--smoothen', type=int, default=1e-6, help='error/term-smoothening constant')
parser.add_argument('-isbn_muz','--is_batch_norm_muz', type=bool, default=True, help='batch norm mu_z')
parser.add_argument('-isbn_x','--is_batch_norm_x', type=bool, default=True, help='batch norm x')
parser.add_argument('-isbn_phi','--is_batch_norm_phi', type=bool, default=True, help='batch norm phi')
parser.add_argument('-KLD','--have_KLD', type=str, default='True', help='KLD yes or no')
# parser.add_argument('-esam','--eps_samples', type=int, default=1, help='how many eps samples to take')

## optimization arguments
parser.add_argument('-lr','--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('-e','--epochs', type=int, default=1000, help='number of epochs')
parser.add_argument('-ncoord','--num_coordinates', type=int, default=2, help='num of coordinates')
parser.add_argument('-drop','--dropout', type=float, default=0.5, help='dropout rate on the encoder')
parser.add_argument('-wdecay','--weight_decay', type=float, default=0.0, help='some l2 regularization')

### evaluation / visualization arguments
parser.add_argument('--num_words', type=int, default=10, help='number of words for topic viz')
parser.add_argument('--visualize', type=bool, default=False, help='produce visualization')
parser.add_argument('--show_knn', type=bool, default=True, help='Show KNN score{k = 10,20,30,40,50}')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

  torch.cuda.empty_cache()

  bs = args.batch_size
  epochs = args.epochs

  en1_units_x = args.hidden1
  en2_units_x = args.hidden2
  dropout = args.dropout
  num_coordinate = args.num_coordinates
  variance_x = args.variance_x
  num_topic = args.num_topics
  smoothen = args.smoothen 

  isbn_muz = args.is_batch_norm_muz
  isbn_x = args.is_batch_norm_x
  isbn_phi = args.is_batch_norm_phi
  skipgram_embeddings = args.skipgram_embeddings
  have_KLD = args.have_KLD

  # eps_samples = args.eps_samples

  visualize = args.visualize
  show_knn = args.show_knn
  num_words = args.num_words

  emb_size = args.emb_size
  learning_rate = args.learning_rate
  weight_decay = args.weight_decay
  beta1 = 0.99
  beta2 = 0.999

  model_name = 'Embedding_Vis_Model'
  home_dir='/home/grad16/sakumar/CIKM_Experiments_2021/'+model_name
 
  #### Data Downloading ####
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  data_name= args.dataset # wos,bbc,searchsnippet,stackoverflow,agnews120k
  dtype=args.dtype # full, short,small

  if data_name == 'bbc': 
    max_features = 2000
    bs = 256
  elif data_name == 'searchsnippet': 
    max_features = 3000
    bs = 1000
  elif data_name == 'yahooanswers': 
    max_features = 4000
    bs = 5000
  elif data_name == 'agnews120k':
    max_features = 8000
    bs = 10000
  else: 
    max_features = 4000
    bs = 5000

  # download_data(d_data) ###

  # ##### Data loading #####
  loaded_data = load_data(data_name,dtype,home_dir,skipgram_embeddings)
  data_preprocessed , data_preprocessed_labels , embeddings, name = loaded_data
  print(name,len(data_preprocessed_labels),len(data_preprocessed),len(embeddings),dtype)

  if dtype == 'small':
    data_preprocessed = data_preprocessed[0]
    data_preprocessed_labels = data_preprocessed_labels[0]

  else:  
    len_docs = [len(d.split(" ")) for d in data_preprocessed]
    print(np.min(len_docs),np.mean(len_docs).round(2),np.max(len_docs))

  torch.cuda.empty_cache()

  if(len(data_preprocessed)>200000):
    get_data_label_fn = get_data_label_vocab_for_large_data
    train_fn = train_for_large
    test_fn = test_for_large
  else:
    get_data_label_fn = get_data_label_vocab_normal 
    train_fn = train
    test_fn = test

  count_vec,train_label,id_vocab = get_data_label_fn(data_preprocessed,data_preprocessed_labels,max_features)
  print('data_shape: ',count_vec.shape,"\n")
  print('args: '+str(args)+"\n\n")
  print("kld: ",str(have_KLD),"dropout:",dropout,"\n\n")
  print(en1_units_x,en2_units_x,dropout,learning_rate,variance_x,bs,'\n\n')
  all_indices = torch.randperm(count_vec.shape[0]).split(bs)
  embedding_tensor_sorted_alp = get_doc_word_embeddings(id_vocab,embeddings)

  embedding_tensor_words = embedding_tensor_sorted_alp[0]# norm_emb_alp 
  embedding_tensor_words = embedding_tensor_words.to(device)

  num_input = count_vec.shape[1]
  model = Model(num_input, en1_units_x, en2_units_x, num_coordinate, num_topic, dropout, 
  variance_x, bs, embedding_tensor_words,emb_size,smoothen,isbn_muz,isbn_x,isbn_phi,have_KLD, "gaussian")
  
  tstart = time()
  trained_model = train_fn(model,count_vec,train_label,num_input,num_topic,learning_rate,beta1,
                          beta2,all_indices,epochs,weight_decay,device)

  tstop = time()              
  x_list,labels_list,zphi,doc_ids,beta = test_fn(trained_model,all_indices,count_vec,num_topic,train_label,id_vocab,device)
  
  #*********** print topics ***********#
  print_Topics(beta,id_vocab,num_words)

  #*********** visualize ***********#
  if visualize:
    plot_fig(x_list, labels_list,zphi,lim =10,contour='no')

  #*********** KNN ***********#
  if show_knn:
    print('KNN:- ',cal_knn(x_list,labels_list),np.mean(cal_knn(x_list,labels_list)))

  all_results = {}
  all_results['X'] = x_list
  all_results['phi'] = zphi
  ail=np.asarray([t.item() for t in doc_ids])
  all_results['doc_ids'] = ail
  all_results['KNN'] = cal_knn(x_list,labels_list)

  os.chdir(home_dir)
  save_dir = home_dir+"/SavedOutput/"+data_name+"/"+dtype+"/topics_"+str(args.num_topics)+"/runs_"+str(args.run)+"_drop_"+str(dropout)
  os.makedirs(save_dir,exist_ok=True)
  os.chdir(save_dir)

  with open("results_"+data_name+"_"+str(args.num_topics)+".txt","w") as f:
      f.write(str(args)+"\n\n")
      f.write("Dropout: "+str(dropout)+", topics: "+str(args.num_topics))
      f.write('---'*30+'\n\n')
      f.write('runtime: - '+str(tstop-tstart)+'s\n\n')
      f.write('---------------Printing the Topics------------------\n')
      topword_topics = get_topwords(beta,id_vocab,num_words)
      topword_topics_list=[]
      for topwords in topword_topics:
          topword_topics_list.append(topwords.split())
          f.write(topwords+'\n')
      f.write('---------------End of Topics---------------------\n')
      f.write('KNN:- '+str(cal_knn(x_list,labels_list)))
      f.write('---'*30+'\n\n')

  model_signature=data_name+'_'+dtype+'_'+str(args.num_topics)+'_'+str(args.run)
  torch.save(trained_model.state_dict(), model_signature+'.pt')
  compressed_pickle(all_results,model_signature+'_all_results')
  os.chdir(home_dir)
