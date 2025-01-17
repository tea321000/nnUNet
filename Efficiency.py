import os, time
import json,glob
import argparse
from pynvml.smi import nvidia_smi
from multiprocessing import Process


def daemon_process(time_interval, json_path, gpu_index=0):
    gpu_memory_max = 0
    while True:
        nvsmi = nvidia_smi.getInstance()
        dictm = nvsmi.DeviceQuery('memory.free, memory.total')
        gpu_memory = dictm['gpu'][gpu_index]['fb_memory_usage']['total'] - dictm['gpu'][gpu_index]['fb_memory_usage'][
            'free']
        print("gpu_memory", gpu_memory)
        # if os.path.exists(json_path):
        #     with open(json_path)as f:
        #         js = json.load(f)
        # else:
        #     js = {
        #     'gpu_memory':[]
        #     }
        # with open(json_path, 'w')as f:
        #     #js['gpu_memory'] = gpu_memory_max
        #     js['gpu_memory'].append(gpu_memory)
        #     json.dump(js, f, indent=4)
        time.sleep(time_interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-time_interval", default=0.1, help="time_interval")
    parser.add_argument("-shell_path", default='predict.sh', help="time_interval")
    parser.add_argument("-gpus", default=0, help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("-docker_input_file", default='./inputs/', help="docker input folder") 
    parser.add_argument("-docker_name", default='nnunet', help="docker output folder") 
    args = parser.parse_args()
    print('We are evaluating', args.docker_name)
    json_dir = './data_all/{}'.format(args.docker_name)
    json_path = os.path.join(json_dir,glob.glob(args.docker_input_file+'/*')[0].split('/')[-1].split('.')[0]+'.json')
    #print(json_path)
    p1 = Process(target=daemon_process, args=(args.time_interval, json_path, args.gpus,))
    p1.daemon = True
    p1.start()
    t0 = time.time()
    cmd = 'docker container run --gpus all --name ttime --rm -v $PWD/inputs/:/workspace/inputs/  -v $PWD/ttime_outputs/:/workspace/outputs/  ttime:latest /bin/bash -c "sh predict.sh"'
    # cmd = 'docker container run  --gpus="device=1"  --name {} --rm -v $PWD/inputs/:/workspace/inputs/ -v $PWD/outputs/:/workspace/outputs/ {}:latest /bin/bash -c "sh {}"'.format(
    # args.docker_name, args.docker_name, args.shell_path)
    print(cmd) 
    os.system(cmd)

    infer_time = time.time() - t0
    print("infer_time", infer_time)
    # with open(json_path, 'r')as f:
    #     js = json.load(f)
    # with open(json_path, 'w')as f:
    #     js['time'] = infer_time
    #     json.dump(js, f, indent=4)

###rep_fold_2_running_mean_disable_tta:  3823.18 87.75 3840 86.70 3812 85.27
###rep_fold_2_running_mean:  4313 94.25      4290 96.09    4289 88.6
# fold_2_running_mean_disable_tta: 4002 93.13 3999 89.58  4001 88.82
### fold_2_running_mean:  4049 90.093    4025 91.31    4029 88.64
###         fold_3:  4046 90.168322  4026 103.7424 4024 94.33