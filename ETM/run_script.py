import os
from time import time
import argparse

parser = argparse.ArgumentParser(description='ETM')
parser.add_argument('--data_name', type=str, default='bbc', help='data Name')
parser.add_argument('--dtype', type=str, default='short', help='type of data')
parser.add_argument('--num_topics', type=str, default='10', help='topics Name')
parser.add_argument('--num_runs',type=str, default='1' , help='runs')

args = parser.parse_args()

# data_name = 'bbc' # bbc,searchsnippet, wos
# dtype = 'short' # full,short,small

data_name = args.data_name
dtype = args.dtype
num_topics =args.num_topics

ETM_dir = '/home/grad16/sakumar/CIKM_Experiments_2021/ETM'
os.chdir(ETM_dir)

if data_name == 'bbc':
  vocab_size = 2000
elif data_name == 'searchsnippet':
  vocab_size = 3000
elif data_name == 'wos':
  vocab_size = 4000
else:
  vocab_size = 4000

embeddings_path_ETM = ETM_dir+"/embeddings_glove_/"+"embeddings_"+data_name+"_"+str(vocab_size)+".txt"

if dtype =="small":
  data_path_ETM = ETM_dir+'/data/content/'+data_name+"/"+dtype+"/"+"sample1"+"/"
else:
  data_path_ETM = ETM_dir+'/data/content/'+data_name+"/"+dtype

num_runs=[args.num_runs]

## script
for r in num_runs:
  dataset_name_ETM = str(r)+"_"+data_name+"_"+dtype
  save_dir = ETM_dir+"/SavedOutput/"+data_name+"/"+dtype+"/topics_"+str(num_topics)+"/runs_"+str(r)
  os.makedirs(save_dir,exist_ok=True)
  os.system("nohup python3 main.py \
  --mode train \
  --dataset "+dataset_name_ETM+" --data_path "+data_path_ETM+" \
  --emb_path "+embeddings_path_ETM+" \
  --num_topics "+str(num_topics)+" --train_embeddings 0 --epochs 1000 > "+save_dir+"/"+"output_"+dataset_name_ETM+"_"+str(num_topics)+".txt")
