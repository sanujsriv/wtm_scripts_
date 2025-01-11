# num_topics = [10,20,30,40,50]
# all_data_names = ['bbc','wos','searchsnippet']
# dtypes = ['full','short','small']

import bz2
import pickle
import _pickle as cPickle
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
data_dir = '/home/grad16/sakumar/CIKM_Experiments_2021/ETM'

import pickle
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

parser = argparse.ArgumentParser(description='ETM')
parser.add_argument('--model_name', type=str, default='ETM', help='model Name')
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

# model = 'ETM'
# all_data_names = ['bbc']
# dtypes = ['short']
# num_topics = ['10']
# num_runs = [1]

sample_size = 20
small_data_no_of_ids = 1

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

all_runs_thetas = []
all_runs_betas = []

for data_name in all_data_names:
  for dtype in dtypes:
    for num_topic in num_topics:
      for i in num_runs:
        os.chdir(data_dir+"/"+"SavedOutput/"+data_name+"/"+dtype+"/topics_"+str(num_topic)+"/runs_"+str(i))
        all_results = decompress_pickle(str(model)+'_all_results_'+str(i)+'_'+data_name+'_'+dtype+'_'+str(num_topic))
#         all_runs_thetas = np.append(all_runs_thetas,all_results['all_theta'])
#         all_runs_betas = np.append(all_runs_betas, all_results['beta'])

# all_runs_thetas = all_runs_thetas.reshape(all_results['all_theta'].shape + (5,))
# all_runs_betas = all_runs_thetas.reshape(all_results['beta'].shape + (5,))

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



if 'small'==dtypes[0]:
  # os.system('unzip '+data_name+'_ETM.zip -d '+data_name+'_ETM')
  labels = np.array(data_preprocessed_labels[0])
  dtype=dtypes[0]
  os.chdir(data_dir+'/'+'data/content/'+data_name+'/'+dtype+'/'+"sample1/")
  idx_permute = load_obj(data_name+"_"+dtype+"_idx_permute")
else:
  # os.system('unzip '+data_name+'_ETM.zip -d '+data_name+'_ETM')
  os.chdir(data_dir+'/'+'data/content/'+data_name+'/'+dtype+'/')
  idx_permute = load_obj(data_name+"_"+dtype+"_idx_permute")
  labels = np.array(data_preprocessed_labels)
  os.chdir(home_dir)


all_KNNs = []
tsne = TSNE(n_jobs=-1,n_iter=1000,verbose=1)
all_X_coord = tsne.fit_transform(all_results['all_theta'])
all_phi_coord = tsne.fit_transform(all_results['beta'])
all_KNNs = np.append(all_KNNs,cal_knn(all_X_coord,labels[idx_permute]))

all_results['X'] = all_X_coord
all_results['phi'] = all_phi_coord
all_results['KNN'] = all_KNNs

save_dir = home_dir+"/tsne_results/"+data_name+"/"+dtype+"/topics_"+str(num_topic)+"/runs_"+str(i)
os.makedirs(save_dir,exist_ok=True)
os.chdir(save_dir)

compressed_pickle(all_results,'all_results_'+str(data_name)+"_"+str(dtypes[0])+"_"+str(num_topic)+"_"+str(i))
with open("tsne_etm_"+str(data_name)+"_"+str(dtypes[0])+"_"+str(num_topic)+"_"+str(i)+".txt","w") as f:
   f.write("**"*50)
   f.write("\n\n")
   f.write("TSNE : "+str(data_name)+str(dtypes[0])+str(num_topic)+"\n\n")
   f.write("KNN: \n\n"+str(all_KNNs)+'\n\n')
   f.write("Coordinates: X \n\n"+str(all_X_coord)+'\n\n')
   f.write("**"*50+'\n\n')
   np.save('X_coordinates.npy', all_X_coord)
   np.save('phi_coordinates.npy', all_phi_coord)
   f.write("Coordinates: phi \n\n"+str(all_phi_coord)+'\n\n')
os.chdir(home_dir)

