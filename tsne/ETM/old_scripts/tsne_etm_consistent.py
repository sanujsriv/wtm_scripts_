# num_topics = [10,20,30,40,50]
# all_data_names = ['bbc','wos','searchsnippet']
# dtypes = ['full','short','small']

from MulticoreTSNE import MulticoreTSNE as TSNE
import os
from time import time
import argparse
import numpy as np  
from sklearn.neighbors import KNeighborsClassifier

def cal_knn(coordinate, label):
    output = []
    for n_neighbors in [10, 20, 30, 40, 50]:
        neigh = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        neigh.fit(coordinate, label)
        output.append(neigh.score(coordinate, label))
    return output

# ptm_dir = '/home/grad16/sakumar/CIKM_Experiments_2021/tsne/PTM'
home_dir = '/home/grad16/sakumar/CIKM_Experiments_2021/tsne/ETM'

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
parser.add_argument('--num_runs', type=str, default='1', help='run Name')

args = parser.parse_args()

model = args.model_name
all_data_names = [args.dataset]
dtypes = [args.dtype]
num_topics = [args.num_topics]
num_runs = [args.num_runs]

sample_size = 20

def load_data(data,dtype,dtype2):
  dir ='/content/data_'+data+'/'+dtype
  if local: 
    os.chdir('/home/grad16/sakumar/CIKM_Experiments_2021/PLSV-VAE'+dir)
  else:
    os.chdir(dir)

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
    if local:
      os.chdir(cur_dir+dir)
    if local=='False':
      os.chdir(dir)

    if (dtype == 'full' or dtype == 'short') and dtype2 == 'small':
      for i in range(small_data_no_of_ids):
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

all_thetas = []
for data_name in all_data_names:
  for dtype in dtypes:
    for num_topic in num_topics:
      for i in num_runs:
        os.chdir(home_dir+"/"+"SavedOutput/"+data_name+"/"+dtype+"/topics_"+str(num_topic)+"/runs_"+str(i))
        theta = load_obj(str(model)+'_theta_all_'+str(i)+'_'+data_name+'_'+dtype+'_'+str(num_topic))
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

# if data_name=="agnews120k":
#   os.system("wget -N https://www.dropbox.com/s/lf4imup7h7yl2hx/agnews120k_ETM.zip")
#   os.system("unzip agnews120k_ETM.zip")

# elif data_name=="twentynews":
#   os.system("wget -N https://www.dropbox.com/s/f9x1mdnd4scfe3b/twentynews_ETM.zip")
#   os.system("unzip twentynews_ETM.zip")

if 'small'==dtypes[0]:
  # os.system('unzip '+data_name+'_ETM.zip -d '+data_name+'_ETM')
  labels = np.array(data_preprocessed_labels[0])
  dtype=dtypes[0]
  os.chdir(home_dir+'/'+data_name+'_ETM/content/'+data_name+'/'+dtype+'/'+"sample1/")
  idx_permute = load_obj(data_name+"_"+dtype+"_idx_permute")
else:
  # os.system('unzip '+data_name+'_ETM.zip -d '+data_name+'_ETM')
  os.chdir(home_dir+'/'+data_name+'_ETM/content/'+data_name+'/'+dtype+'/')
  idx_permute = load_obj(data_name+"_"+dtype+"_idx_permute")
  labels = np.array(data_preprocessed_labels)
  os.chdir(home_dir)
  
tsne = TSNE(n_jobs=16,n_iter=1000)
all_X_coord = []
for theta in all_thetas:
  X = tsne.fit_transform(theta)
  all_X_coord.append(X)

all_KNNs = []
for x in all_X_coord:
  all_KNNs = np.append(all_KNNs,cal_knn(x,labels[idx_permute]))

with open("tsne_etm_"+str(data_name)+str(dtypes[0])+str(num_topic)+str(i)+".txt","w") as f:
   f.write("**"*50)
   f.write("\n\n")
   f.write("TSNE : "+str(data_name)+str(dtypes[0])+str(num_topic)+"\n\n")
   f.write(str(all_KNNs))