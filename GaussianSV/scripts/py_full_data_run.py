import os
from time import time

## full
os.chdir("/home/grad16/sakumar/CIKM_Experiments_2021/GaussianSV")
topics=[str(i) for i in range(10,60,10)]
datasets = ['bbc','searchsnippet']
dtype='full'
# samples=5
no_of_runs = 5

def savetime_to_str(start,stop,msg,msg_val):
  str_time = str(stop - start)
  message = "\nTime for "+msg+" : "+str(msg_val)+" - "+str_time
  return message

for dataset in datasets:
  with open(dataset+"_All_Runtime_Results_"+dtype+".txt", "a") as f:
    start_total_time_for_all_topics = time()
    for num_topic in topics:
      start_total_time_for_all_runs = time()
      for r in range(no_of_runs):	
        start_time_for_each_run = time()	
        os.system("perl gaussiansv.pl \
        --data ./input_data/"+dataset+"/content/GaussianSV_data_"+dataset+"/"+dtype+"/input_data_"+dataset+"_"+dtype+".txt \
        --word_vectors ./input_data/"+dataset+"/content/GaussianSV_data_"+dataset+"/"+dtype+"/unit_len_embeddings"+dataset+"_"+dtype+".txt \
        --num_topics "+num_topic+"\
        --output_file ./Output/"+dataset+"/"+dtype+"/output_"+dtype+"_topics_"+num_topic+"_runs_"+str(r+1)+".txt")
        stop_time_for_each_run = time()
        f.write(savetime_to_str(start_time_for_each_run,stop_time_for_each_run,"num_run",str(r+1)))

      stop_total_time_for_all_runs = time()
      f.write("\n\n"+"**"*20+savetime_to_str(start_total_time_for_all_runs,stop_total_time_for_all_runs,"num_topics",str(num_topic)))

    stop_total_time_for_all_topics = time()
    f.write("\n\n"+"**"*20+savetime_to_str(start_total_time_for_all_topics,stop_total_time_for_all_topics,"dataset",str(dataset)))
