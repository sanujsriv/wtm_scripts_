import os
from time import time
import argparse

parser = argparse.ArgumentParser(description='WTM')
parser.add_argument('--data_name', type=str, default='bbc', help='bbc,searchsnippet, wos')
parser.add_argument('--dtype', type=str, default='short', help='full,short,small')
parser.add_argument('--num_topic', type=str, default='10', help='number of topic')
parser.add_argument('--num_runs', type=str, default='1', help='num of runs')
parser.add_argument('--kld', type=str, default='True', help='KLD yes or no')
parser.add_argument('--eps_samples', type=int, default=1, help='eps samples')
parser.add_argument('--drop', type=float, default=0.25, help='drop out value')
parser.add_argument('--var_drop', type=str, default='True', help='var dropout yes or no')

args = parser.parse_args()

# data_name = 'bbc' # bbc,searchsnippet, wos
# dtype = 'short' # full,short,small

data_name = args.data_name
dtype = args.dtype
num_topic =args.num_topic
num_runs = [args.num_runs]
kld = args.kld
eps_samples = args.eps_samples
var_drop = args.var_drop
drop = args.drop

# data_name = 'bbc'
# dtype = 'short'
# num_topic = '10'
# num_runs = ['1']

model_name = 'Embedding_Vis_Model'
home_dir='/home/grad16/sakumar/CIKM_Experiments_2021/'+model_name
os.chdir(home_dir)

#num_runs=5

if str(var_drop)=='True':
  var_folder = 'var_dropout'
  # Model = Model
else:
  var_folder = 'no_var_dropout'
  # Model = ModelVD

if str(kld) == "True": kld_folder = "kld"
else: kld_folder = "no_kld"

## script
for r in num_runs:
  os.chdir(home_dir)
  save_dir = home_dir+"/SavedOutput/"+var_folder+"/"+kld_folder+"/"+data_name+"/"+dtype+"/topics_"+str(num_topic)+"/runs_"+str(r)
  os.makedirs(save_dir,exist_ok=True)
  os.system("nohup python3 main.py \
   --dataset "+data_name+" --dtype "+dtype+" --num_topics "+num_topic+" --run "+str(r)+" -e 1000 -drop "+str(drop)+" --have_KLD "+str(kld)+" --eps_samples "+str(eps_samples)+" --var_dropout "+str(var_drop)+" \
    > "+save_dir+"/"+"output_"+data_name+"_"+str(num_topic)+".txt")