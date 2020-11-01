import os
import torch

from collections import defaultdict

from edgify.utils import WriterQMNIST

if __name__ == "__main__":
    for writer_id in range(131):
        with open(os.path.join('data', 'QMNIST', 'models-subject-out', str(writer_id), 'accuracy_before.txt'), 'r') as f:
            accuracy_before = f.readline().strip()
        with open(os.path.join('data', 'QMNIST', 'models-subject-out', str(writer_id), 'accuracy_after.txt'), 'r') as f:
            accuracy_after = f.readline().strip()
        print(accuracy_before, ', ', accuracy_after)
        
    print()
    batch_sizes = [64, 128, 256, 512, 1024]
    for batch_size in batch_sizes:
        gpu_utils = []
        mem_utils = []
        active_pwr = []
        idle_pwr = []
        temp = []
        for writer_id in range(131):
            with open(os.path.join('data', 'QMNIST', 'models-subject-out', str(writer_id), 'personalize-server-gpu-' + str(batch_size) + '.csv'), 'r') as f:
                _ = f.readline() # discard header
                active_samples = 0
                idle_samples = 0
                writer_idle_power = 0
                writer_active_power = 0
                writer_gpu_util = 0
                writer_mem_util = 0
                writer_temp = 0

                for sample in f:
                    ts, pstate, util_gpu, util_mem, temp_gpu, temp_mem, power = sample.strip().split(',')
                    if util_gpu.strip() == '0 %' and util_mem.strip() == '0 %':
                        # idle state
                        writer_idle_power += float(power.strip().split(' ')[0])
                        idle_samples += 1
                    else:
                        writer_active_power += float(power.strip().split(' ')[0])
                        writer_gpu_util += float(util_gpu.strip().split(' ')[0])
                        writer_mem_util += float(util_mem.strip().split(' ')[0])
                        writer_temp += float(temp_gpu.strip())
                        active_samples += 1
                
                gpu_utils.append(writer_gpu_util / active_samples)
                mem_utils.append(writer_mem_util / active_samples)
                active_pwr.append(writer_active_power / active_samples)
                idle_pwr.append(writer_idle_power / idle_samples)
                temp.append(writer_temp / active_samples)
        print(batch_size, sum(gpu_utils) / len(gpu_utils), \
            sum(mem_utils) / len(mem_utils), \
                sum(temp) / len(temp), \
                    sum(active_pwr) / len(active_pwr), \
                        sum(idle_pwr) / len(idle_pwr))
        