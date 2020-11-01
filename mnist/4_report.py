import torch

from collections import defaultdict

from edgify.utils import WriterQMNIST

if __name__ == "__main__":
    for writer_id in range(131):
        with open(os.path.join('data', 'QMNIST', 'model-subjet-out', str(writer_id), 'accuracy_before.txt'), 'r') as f:
            accuracy_before = f.readline().strip()
        with open(os.path.join('data', 'QMNIST', 'model-subjet-out', str(writer_id), 'accuracy_after.txt'), 'r') as f:
            accuracy_after = f.readline().strip()
        print(accuracy_before, accuracy_after)
        
    print()
    batch_sizes = [64, 128, 256, 512, 1024]
    for batch_size in batch_sizes:
        gpu_utils = []
        mem_utils = []
        active_pwr = []
        idle_pwr = []
        temp = []
        for writer_id in range(131):
            with open(os.path.join('data', 'QMNIST', 'model-subjet-out', str(writer_id), 'personalize-server-gpu-' + str(batch_size) + '.csv'), 'r') as f:
                _ = f.readline() # discard header
                samples = 0
                writer_idle_power = 0
                writer_active_power = 0
                writer_gpu_util = 0
                writer_mem_util = 0
                writer_temp = 0

                for sample in f:
                    samples += 1
                    ts, pstate, util_gpu, util_mem, temp_gpu, temp_mem, power = list(filter(lambda  x: x.strip(), sample.strip().split(',')))
                    if util_gpu == '0 %' and util_mem == '0 %':
                        # idle state
                        writer_idle_power.append(float(power.split(' ')[0]))
                    else:
                        writer_active_power.append(float(power.split(' ')[0]))
                        writer_gpu_util.append(float(util_gpu.split(' ')[0]))
                        writer_mem_util.append(float(util_mem.split(' ')[0]))
                        writer_temp.append(float(temp_gpu))
                
                gpu_utils.append(sum(writer_gpu_util) / samples)
                mem_utils.append(sum(writer_mem_util) / samples)
                active_pwr.append(sum(writer_active_power) / samples)
                idle_pwr.append(sum(writer_idle_power) / samples)
                temp.append(sum(writer_temp) / samples)
        print(batch_size, sum(gpu_utils) / len(gpu_utils), \
            sum(mem_utils) / len(mem_utils), \
                sum(temp) / len(temp), \
                    sum(active_pwr) / len(active_pwr), \
                        sum(idle_pwr) / len(idle_pwr))
        