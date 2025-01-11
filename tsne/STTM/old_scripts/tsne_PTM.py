from MulticoreTSNE import MulticoreTSNE as TSNE
import os
from time import time
import argparse
import numpy as np  
from sklearn.neighbors import KNeighborsClassifier

local =True

def load_data(data,dtype,dtype2):
  dir ='/content/data_'+data+'/'+dtype
  os.chdir('/home/grad16/sakumar/CIKM_Experiments_2021/PLSV-VAE'+dir)

  if dtype2=='normal':
    data_preprocessed=load_obj("data_preprocessed_"+data+"_"+dtype)
    data_preprocessed_labels=load_obj("data_preprocessed_labels_"+data+"_"+dtype)
    embeddings=load_obj("embeddings_"+data+"_"+dtype)

    if local==True:
      os.chdir('/home/grad16/sakumar/CIKM_Experiments_2021/PLSV-VAE')
    if local==False:
      os.chdir('/content/') 
    return data_preprocessed,data_preprocessed_labels,embeddings,data

  elif dtype2=='small':
    data_preprocessed=[]
    data_preprocessed_labels=[]
    
    if local==True:
      os.chdir('/home/grad16/sakumar/CIKM_Experiments_2021/PLSV-VAE'+dir)
    if local==False:
      os.chdir('/content/')  
    
    print(os.getcwd())
    id = '1'
    data_preprocessed.append(load_obj(id+'_docs_sampled_'+dtype2+'_'+str(sample_size)+data+dtype))
    data_preprocessed_labels.append(load_obj(id+'_labels_sampled_'+dtype2+'_'+str(sample_size)+data+dtype))
    embeddings = load_obj('embeddings'+'_'+data+'_'+dtype)

    return data_preprocessed,data_preprocessed_labels,embeddings,data


def cal_knn(coordinate, label):
    output = []
    for n_neighbors in [10, 20, 30, 40, 50]:
        neigh = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        neigh.fit(coordinate, label)
        output.append(neigh.score(coordinate, label))
    return output

home_dir = '/home/grad16/sakumar/CIKM_Experiments_2021/tsne/PTM'

import pickle
local =True
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_obj_pkl5(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle5.load(f)

parser = argparse.ArgumentParser(description='STTM')
parser.add_argument('--model_name', type=str, default='PTM', help='model Name')
parser.add_argument('--dataset', type=str, default='bbc', help='model Name')
parser.add_argument('--dtype', type=str, default='short', help='model Name')
parser.add_argument('--num_topics', type=str, default='10', help='model Name')

args = parser.parse_args()

model = args.model_name
all_data_names = [args.dataset]
dtypes = [args.dtype]
num_topics = [args.num_topics]

# model = 'PTM'
# all_data_names = ['twentynews']
# dtypes = ['small']
# num_topics = ['10']

sample_size = 20

# num_topics = [10,20,30,40,50]
# all_data_names = ['wos']
# dtypes = ['short']

# all_topics_theta = []

all_runs_theta = []
for data_name in all_data_names:
  for dtype in dtypes:
    for num_topic in num_topics:
      for i in range(5):
        os.chdir(home_dir+"/"+model+"/"+dtype+"/topic_"+str(num_topic)+"/results/")
        with open(model+"_"+data_name+"_"+"numtopic_"+str(num_topic)+"_run_"+str(i+1)+"_"+dtype+".theta") as f:
          content = f.read()
          content = content.split('\n')
          run_theta = []
          for x in content:
            theta_i = []
            vals = x.split(' ')
            for v in vals[:-1]:
              theta_i.append(float(v))
            run_theta.append(theta_i) 
          all_runs_theta.append(np.asarray(run_theta[:-1]))

if dtypes[0] == "small":
  dtype="full"
  dtype2 ="small"

elif dtypes[0] =="short" or dtypes[0] =="full":
   dtype2="normal"

#### Input Data Downloading ##
os.chdir(home_dir)
# download_data(data_name) ###
# ##### Data loading #####
loaded_data = load_data(data_name,dtype,dtype2)
data_preprocessed , data_preprocessed_labels , embeddings, name = loaded_data
if dtypes[0]!='small':
  print(name,len(data_preprocessed_labels),len(data_preprocessed),len(embeddings),dtype,dtype2)
if dtypes[0]=="small":
  print(name,len(data_preprocessed_labels[0]),len(data_preprocessed[0]),len(embeddings),dtype,dtype2)
os.chdir(home_dir)

if 'small'==dtypes[0]:
  labels = np.array(data_preprocessed_labels[0])
  # dtype=dtypes[0]
  # os.chdir(home_dir+'/'+data_name+'_ETM/content/'+data_name+'/'+dtype+'/'+" /")

else:
  labels = np.array(data_preprocessed_labels) 

tsne = TSNE(n_jobs=16)
all_X_coord = []
for theta in all_runs_theta:
  X = tsne.fit_transform(theta)
  all_X_coord.append(X)

all_KNNs = []
for x in all_X_coord:
  all_KNNs = np.append(all_KNNs,cal_knn(x,labels))
  
with open("tsne_"+str(data_name)+"_"+str(dtypes[0])+"_"+str(num_topic)+".txt","w") as f:
   f.write("**"*50)
   f.write("\n\n")
   f.write("TSNE : "+str(data_name)+"_"+str(dtypes[0])+"_"+str(num_topic)+"\n\n")
   f.write(str(all_KNNs.reshape(1,5,5)))