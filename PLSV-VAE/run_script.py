import os
from time import time
import argparse

parser = argparse.ArgumentParser(description='PLSV-VAE')
parser.add_argument('--data_name', type=str, default='bbc', help='bbc,searchsnippet, wos')
parser.add_argument('--dtype', type=str, default='short', help='full,short,small')
parser.add_argument('--num_topic', type=str, default='10', help='number of topic')
parser.add_argument('--num_runs', type=str, default='1', help='num of runs')
args = parser.parse_args()

# data_name = 'bbc' # bbc,searchsnippet, wos
# dtype = 'short' # full,short,small

data_name = args.data_name
dtype = args.dtype
num_topic =args.num_topic
num_runs = [args.num_runs]

# data_name = 'bbc'
# dtype = 'short'
# num_topic = '10'
# num_runs = ['1']

model_name = 'PLSV-VAE'
home_dir='/home/grad16/sakumar/CIKM_Experiments_2021/'+model_name
os.chdir(home_dir)

#num_runs=5

## script
for r in num_runs:
  os.chdir(home_dir)
  save_dir = home_dir+"/SavedOutput/"+data_name+"/"+dtype+"/topics_"+str(num_topic)+"/runs_"+str(r)
  os.makedirs(save_dir,exist_ok=True)
  os.system("nohup python3 train.py \
   --dataset "+data_name+" --dtype "+dtype+" --topics "+num_topic+" --run "+str(r)+" -e 1000 \
    > "+save_dir+"/"+"output_"+data_name+"_"+str(num_topic)+".txt")