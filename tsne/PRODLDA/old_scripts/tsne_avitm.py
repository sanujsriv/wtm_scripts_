from MulticoreTSNE import MulticoreTSNE as TSNE
import os
from time import time
import argparse
import numpy as np  
from sklearn.neighbors import KNeighborsClassifier
import pickle

def cal_knn(coordinate, label):
    output = []
    for n_neighbors in [10, 20, 30, 40, 50]:
        neigh = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        neigh.fit(coordinate, label)
        output.append(neigh.score(coordinate, label))
    return output

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
local =True

def load_data(data,dtype,dtype2):
  data = data.lower()
  dtype = dtype.lower()
  dtype2 = dtype2.lower() 
  d,l,e,fn = load_d_data(data,dtype,dtype2)
  return d,l,e,fn

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
    sample_size = 20
    if local:
      os.chdir('/home/grad16/sakumar/CIKM_Experiments_2021/PLSV-VAE'+dir)
    if local=='False':
      os.chdir(dir)

    
    if (dtype == 'full' or dtype == 'short') and dtype2 == 'small':
      for i in range(1):
        id = str(i+1)
        data_preprocessed.append(load_obj(id+'_docs_sampled_'+dtype2+'_'+str(sample_size)+data+dtype))
        data_preprocessed_labels.append(load_obj(id+'_labels_sampled_'+dtype2+'_'+str(sample_size)+data+dtype))
        embeddings = load_obj('embeddings'+'_'+data+'_'+dtype)
    if local==True:
      os.chdir('/home/grad16/sakumar/CIKM_Experiments_2021/PLSV-VAE')
    if local==False:
      os.chdir('/content/')  
    # os.system("rm -r *")
    return data_preprocessed,data_preprocessed_labels,embeddings,data



parser = argparse.ArgumentParser(description='PRODLDA')
parser.add_argument('--model_name', type=str, default='PRODLDA', help='model Name')
parser.add_argument('--dataset', type=str, default='agnews120k', help='data Name')
parser.add_argument('--dtype', type=str, default='short', help='model Name')
parser.add_argument('--num_topics', type=str, default='10', help='model Name')

args = parser.parse_args()

model_name = args.model_name
all_data_names = [args.dataset]
dtypes = [args.dtype]
num_topics = [args.num_topics]

# num_topics = [10]
# all_data_names = ['agnews120k']
# dtypes = ['short']

all_indices_l = []
all_thetas = []


for data_name in all_data_names:
  for dtype in dtypes:
    for num_topic in num_topics:
      for i in range(5):
        os.chdir(home_dir+"/"+"SavedOutput/"+data_name+"/"+dtype+"/topics_"+str(num_topic)+"/runs_"+str(i+1))
        theta = load_obj(data_name+'_'+dtype+"_"+str(num_topic)+"_"+str(i+1)+"_theta")
        all_indices = load_obj(data_name+'_'+dtype+"_"+str(num_topic)+"_"+str(i+1)+"_all_indices")
        all_indices_l.append(all_indices)
        all_thetas.append(theta)

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
# download_data(data_name) ###
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

tsne = TSNE(n_jobs=16)
all_X_coord = []
for theta in all_thetas:
  X = tsne.fit_transform(theta)
  all_X_coord.append(X)

all_KNNs = []
all_indx_np = np.asarray(all_indices_l)
for x,idx_permute in zip(all_X_coord,all_indx_np):
  all_KNNs = np.append(all_KNNs,cal_knn(x,labels[idx_permute.astype('int')]))

with open(model_name+"_tsne_"+str(data_name)+"_"+str(dtypes[0])+"_"+str(num_topic)+".txt","w") as f:
   f.write("**"*50)
   f.write("\n\n")
   f.write("TSNE : "+str(data_name)+"_"+str(dtypes[0])+"_"+str(num_topic)+"\n\n")
   f.write(str(all_KNNs.reshape(len(num_topics),5,5)))