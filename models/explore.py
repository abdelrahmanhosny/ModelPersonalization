import os
import argparse
import re
import statistics
import matplotlib.pyplot as plt

from collections import defaultdict

def parse_tegrastats_file(file_path):
    avg_gpu_power = None
    avg_cpu_power = None
    
    gpu_utils = []
    cpu_utils = []
    mem_utils = []

    total_time = 0
    
    with open(file_path, 'r') as f:
        for line in f:
            total_time += 1     # add 1 second

            match = re.search('RAM (?P<mem_util>[0-9]+)/[0-9]+MB' ,line)
            mem_utils.append(float(match.group('mem_util')))

            match = re.search('GR3D_FREQ (?P<gpu_util>[0-9]+)\%' ,line)
            u = float(match.group('gpu_util'))
            if u != 0:
                gpu_utils.append(u)

            match = re.search('CPU \[(?P<cpu1_util>[0-9]+)\%\@[0-9]+,(?P<cpu2_util>[0-9]+)\%\@[0-9]+,(?P<cpu3_util>[0-9]+)\%\@[0-9]+,(?P<cpu4_util>[0-9]+)\%\@[0-9]+\] ' ,line)
            u = (float(match.group('cpu1_util')) + float(match.group('cpu2_util')) + float(match.group('cpu3_util')) + float(match.group('cpu4_util'))) / 4
            if u != 0:
                cpu_utils.append(u)

            match = re.search('POM_5V_GPU [0-9]+\/(?P<gpu_power>[0-9]+)', line)
            avg_gpu_power = float(match.group('gpu_power'))

            match = re.search('POM_5V_CPU [0-9]+\/(?P<cpu_power>[0-9]+)', line)
            avg_cpu_power = float(match.group('cpu_power'))
    
    return avg_cpu_power, avg_gpu_power, statistics.mean(gpu_utils), statistics.mean(cpu_utils), statistics.mean(mem_utils) / 3956 * 100., total_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Report Stats from Training on Jetson Nano')
    parser.add_argument('--mode', type=str, required=True,
                        help='Choose from [gpu, gpu_freeze, cpu, cpu_freeze]')
    parser.add_argument('--report', type=str, required=True,
                        help='Choose from [gpu_power, cpu_power, gpu_energy, cpu_energy, gpu_util, cpu_util, mem_util, time]')
    args = parser.parse_args()

    plt.rcParams.update({'font.size': 14})
    plt.rcParams['axes.labelweight'] = 'bold'

    models = ['MOBILE_NET', 'GOOGLENET', 'RESNET18', 'RESNET50']
    legend_names = {
        'MOBILE_NET': 'MobileNet (3.5M)',
        'GOOGLENET': 'GoogLeNet (6.6M)',
        'RESNET18': 'ResNet18 (11.6M)',
        'RESNET50': 'ResNet50 (25.5M)'
    }
    y_labels = {
        'gpu_power': 'Avg. GPU Power (mW)',
        'cpu_power': 'Avg. CPU Power (mW)',
        'gpu_energy': 'GPU Total Energy (J)',
        'cpu_energy': 'CPU Total Energy (J)',
        'gpu_util': 'Avg. GPU Utilization (%)',
        'cpu_util': 'Avg. CPU Utilization (%)',
        'mem_util': 'Avg. Memory Utilization (%)',
        'time': 'Forward + Backprob Time (s)',
    }
    mode_paths = {
        'gpu': '',
        'gpu_freeze': 'FREEZE',
        'cpu': 'NO_CUDA',
        'cpu_freeze': 'NO_CUDA_FREEZE'
    }
    batch_size = [2, 4, 8, 16, 32, 64, 128, 256]
    markers = iter(['o', 'x', '+', 'd'])

    for m in models:
        metrics = defaultdict(lambda: [])
        for s in batch_size:
            report = os.path.join('experiments', 'CIFAR10', m, mode_paths[args.mode], str(s) + '.txt')
            avg_cpu_power, avg_gpu_power, avg_gpu_util, avg_cpu_util, avg_mem_util, total_time = parse_tegrastats_file(report)
            metrics['gpu_power'].append(avg_gpu_power)
            metrics['cpu_power'].append(avg_cpu_power)
            metrics['gpu_energy'].append(avg_gpu_power * total_time / 1000)
            metrics['cpu_energy'].append(avg_cpu_power * total_time / 1000)
            metrics['gpu_util'].append(avg_gpu_util)
            metrics['cpu_util'].append(avg_cpu_util)
            metrics['mem_util'].append(avg_mem_util)
            metrics['time'].append(total_time)
        
        plt.plot(metrics[args.report], marker=next(markers), markersize=8, label=legend_names[m])
    
    plt.xticks(range(len(batch_size)), batch_size, size='large')
    plt.ylabel(y_labels[args.report])
    plt.legend()
    plt.tight_layout()

    save_dir = os.path.join('viz', args.mode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, args.report + '.png'))
