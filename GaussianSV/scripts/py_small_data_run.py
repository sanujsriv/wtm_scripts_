import os
from time import time

## Small
os.chdir("/home/grad16/sakumar/CIKM_Experiments_2021/GaussianSV")
topics=[str(t) for t in range(10,60,10)]
datasets = ['bbc','searchsnippet','wos']
# dtypes=['full','short','small']
samples=5
no_of_runs = 5

def savetime_to_str(start,stop,msg,msg_val):
  str_time = str(stop - start)
  message = "\nTime for "+msg+" : "+str(msg_val)+" - "+str_time
  return message

with open("All_Runtime_Results_Small.txt", "w") as f:
  start_total_time_for_all_dataset = time()
  for dataset in datasets:
    start_total_time_for_all_topics = time()
    for num_topic in topics:
      start_total_time_for_all_samples = time()
      for i in range(samples):	
        start_total_time_for_all_runs = time()
        for r in range(no_of_runs):	
          start_time_for_each_run = time()	
          os.system("perl gaussiansv.pl \
        --data ./input_data/"+dataset+"/content/GaussianSV_data_"+dataset+"/small/input_data_"+dataset+"_smallsample_"+str(i+1)+".txt \
        --word_vectors ./input_data/"+dataset+"/content/GaussianSV_data_"+dataset+"/small/unit_len_embeddings"+dataset+"_smallsample_"+str(i+1)+".txt \
        --num_topics "+num_topic+"\
        --output_file ./Output/"+dataset+"/small/output_small_sample_"+str(i+1)+"_topics_"+num_topic+"_runs_"+str(r+1)+".txt")
          stop_time_for_each_run = time()
          f.write(savetime_to_str(start_time_for_each_run,stop_time_for_each_run,"num_run",str(r+1)))

        stop_total_time_for_all_runs = time()
        f.write("\n\n"+"**"*20+savetime_to_str(start_total_time_for_all_runs,stop_total_time_for_all_runs,"num_sample",str(i+1)))

      stop_total_time_for_all_samples = time()
      f.write("\n\n"+"**"*20+savetime_to_str(start_total_time_for_all_samples,stop_total_time_for_all_samples,"num_topics",str(num_topic)))

    stop_total_time_for_all_topics = time()
    f.write("\n\n"+"**"*20+savetime_to_str(start_total_time_for_all_topics,stop_total_time_for_all_topics,"dataset",str(dataset)))

  stop_total_time_for_all_dataset = time()
  f.write("\n\n"+"**"*20+savetime_to_str(start_total_time_for_all_dataset,stop_total_time_for_all_dataset,"total time","For all datasets"))