from MulticoreTSNE import MulticoreTSNE as TSNE
import os
from time import time
import argparse
import numpy as np  
from sklearn.neighbors import KNeighborsClassifier

import pickle5
import bz2
import pickle
import _pickle as cPickle

local =True

def compressed_pickle(data,title):
  with bz2.BZ2File(title + '.pbz2', 'w') as f:
    cPickle.dump(data, f)

def decompress_pickle(file):
 data = bz2.BZ2File(file+".pbz2", 'rb')
 data = cPickle.load(data)
 return data

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_obj_pkl5(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle5.load(f)

home_dir = '/home/grad16/sakumar/CIKM_Experiments_2021/tsne/PRODLDA'
results_dir =  '/home/grad16/sakumar/CIKM_Experiments_2021/prodLDA/pytorch-avitm'
local =True

def download_data(data):
  data = data.lower()
  os.chdir(home_dir)  

  if data=='agnews120k' and not os.path.exists("agnews120k_4000_.zip"):
    os.system('wget -N -c https://www.dropbox.com/s/p7b2msz5fzqvc1q/agnews120k_4000_.zip')
    os.system('unzip agnews120k_4000_.zip')
  
  elif data=='bbc' and not os.path.exists("bbc_2000_20.zip"):
     os.system('wget -N https://www.dropbox.com/s/4mmiaed6rg5lpb1/bbc_2000_20.zip')
     os.system('unzip bbc_2000_20.zip')

  elif data=='searchsnippet' and not os.path.exists("searchsnippet_3000_20.zip"):
      os.system('wget -N https://www.dropbox.com/s/tq31csl55b3oc09/searchsnippet_3000_20.zip')
      os.system('unzip searchsnippet_3000_20.zip')
    
  elif data=='yahooanswers' and not os.path.exists("yahooanswers_sampled_5000.zip"):
       os.system('wget -N https://www.dropbox.com/s/hlxlhyzivg3kfbj/yahooanswers_sampled_5000.zip')
       os.system('unzip yahooanswers_sampled_5000.zip')
  
  # else: print('"data" is entered incorrectly ')  


def load_data(data,dtype,dtype2):
  dir ='/content/data_'+data+'/'+dtype
  os.chdir(home_dir+dir)

  if dtype2=='normal':
    data_preprocessed=load_obj_pkl5("data_preprocessed_"+data+"_"+dtype)
    data_preprocessed_labels=load_obj_pkl5("data_preprocessed_labels_"+data+"_"+dtype)
    embeddings=load_obj_pkl5("embeddings_"+data+"_"+dtype)

    if local==True:
      os.chdir(home_dir)
    if local==False:
      os.chdir('/content/') 

    return data_preprocessed,data_preprocessed_labels,embeddings,data

  elif dtype2=='small':
    data_preprocessed=[]
    data_preprocessed_labels=[]
  
    id = '1'
    data_preprocessed.append(load_obj(id+'_docs_sampled_'+dtype2+'_'+str(sample_size)+data+dtype))
    data_preprocessed_labels.append(load_obj(id+'_labels_sampled_'+dtype2+'_'+str(sample_size)+data+dtype))
    embeddings = load_obj('embeddings'+'_'+data+'_'+dtype)

    if local==True:
      os.chdir(home_dir+dir)
    if local==False:
      os.chdir('/content/')  

    return data_preprocessed,data_preprocessed_labels,embeddings,data


def cal_knn(coordinate, label):
    output = []
    for n_neighbors in [10, 20, 30, 40, 50]:
        neigh = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        neigh.fit(coordinate, label)
        output.append(neigh.score(coordinate, label))
    return output


parser = argparse.ArgumentParser(description='PRODLDA')
parser.add_argument('--model_name', type=str, default='PRODLDA', help='model Name')
parser.add_argument('--dataset', type=str, default='yahooanswers', help='data Name')
parser.add_argument('--dtype', type=str, default='short', help='model Name')
parser.add_argument('--num_topics', type=str, default='10', help='model Name')
parser.add_argument('--num_runs', type=str, default='1', help='model Name')

args = parser.parse_args()

model_name = args.model_name
all_data_names = [args.dataset]
dtypes = [args.dtype]
num_topics = [args.num_topics]
num_runs = [args.num_runs]

# model_name = 'PRODLDA'
# all_data_names = ['yahooanswers']
# dtypes = ['short']
# num_topics = ['10']
# num_runs = ['1']

# all_indices_l = []
# all_thetas = []


for data_name in all_data_names:
  for dtype in dtypes:
    for num_topic in num_topics:
      for i in num_runs:
        os.chdir(results_dir+"/"+"SavedOutput/"+data_name+"/"+dtype+"/topics_"+str(num_topic)+"/runs_"+str(i))
        all_results = decompress_pickle(data_name+'_'+dtype+"_"+str(num_topic)+"_"+str(i)+"_all_results")
        # all_indices = load_obj(data_name+'_'+dtype+"_"+str(num_topic)+"_"+str(i)+"_all_indices")
        # all_indices_l.append(all_indices)
        # all_thetas.append(theta)

if dtypes[0]=="small":
  data_name = all_data_names[0]
  dtype="full"
  dtype2=dtypes[0]
else:
    data_name = all_data_names[0]
    dtype=dtypes[0]
    dtype2 = "normal"
#### Input Data Downloading ##

os.chdir(home_dir)
download_data(data_name) ###
# ##### Data loading #####
loaded_data = load_data(data_name,dtype,dtype2)
data_preprocessed , data_preprocessed_labels , embeddings, name = loaded_data
print(name,len(data_preprocessed_labels),len(data_preprocessed),len(embeddings),dtype,dtype2)

os.chdir(home_dir)
if 'small'==dtypes[0]:
  labels = np.array(data_preprocessed_labels[0])
else:
  labels = np.array(data_preprocessed_labels)
  os.chdir(home_dir)

theta = all_results['theta']
beta = all_results['beta']
doc_ids = np.asarray(all_results['doc_ids'])

all_KNNs = []

tsne = TSNE(n_jobs=16,n_iter=1000)
all_X_coord = tsne.fit_transform(theta)
all_phi_coord = tsne.fit_transform(beta)
all_KNNs = np.append(all_KNNs,cal_knn(all_X_coord,labels[doc_ids]))

all_results['X'] = all_X_coord
all_results['phi'] = all_phi_coord
all_results['KNN'] = all_KNNs

save_dir = home_dir+"/"+str(model_name)+"/tsne_results/"+data_name+"/"+dtype+"/topics_"+str(num_topic)+"/runs_"+str(i)
os.makedirs(save_dir,exist_ok=True)
os.chdir(save_dir)

compressed_pickle(all_results,str(model_name)+'_all_results_'+str(data_name)+"_"+str(dtypes[0])+"_"+str(num_topic)+"_"+str(i))
with open(str(model_name)+"_tsne_etm_"+str(data_name)+"_"+str(dtypes[0])+"_"+str(num_topic)+"_"+str(i)+".txt","w") as f:
   f.write("**"*50)
   f.write("\n\n")
   f.write("TSNE : "+str(model_name)+"_"+str(data_name)+"_"+str(dtypes[0])+"_"+str(num_topic)+"\n\n")
   f.write("KNN: \n\n"+str(all_KNNs)+'\n\n')
   f.write("Coordinates: X \n\n"+str(all_X_coord)+'\n\n')
   f.write("**"*50+'\n\n')
   np.save('X_coordinates.npy', all_X_coord)
   np.save('phi_coordinates.npy', all_phi_coord)
   f.write("Coordinates: phi \n\n"+str(all_phi_coord)+'\n\n')
os.chdir(home_dir)