import os
import torch
import argparse
import re
import statistics
import matplotlib.pyplot as plt

from collections import defaultdict

from edgify.utils import WriterQMNIST

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Report Stats from Training on Jetson Nano')
    parser.add_argument('--file', type=str, required=True,
                        help='Open file')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N', \
        help='input batch size for training (default: 64)')
    
    
    args = parser.parse_args()
    gpu_utils = []
    cpu_utils = []
    mem_utils = []
    gpu_power = []
    cpu_power = []
    gpu_energy = 0
    cpu_energy = 0
    with open(args.file, 'r') as f:
        for line in f:
            match = re.search('RAM (?P<mem_util>[0-9]+)/[0-9]+MB' ,line)
            mem_utils.append(float(match.group('mem_util')))

            match = re.search('GR3D_FREQ (?P<gpu_util>[0-9]+)\%' ,line)
            gpu_utils.append(float(match.group('gpu_util')))

            match = re.search('CPU \[(?P<cpu1_util>[0-9]+)\%\@[0-9]+,(?P<cpu2_util>[0-9]+)\%\@[0-9]+,off,off\] ' ,line)
            cpu_utils.append( (float(match.group('cpu1_util')) + float(match.group('cpu2_util'))) / 2 )

            match = re.search('POM_5V_GPU (?P<gpu_power>[0-9]+)\/[0-9]+', line)
            gpu_power.append(float(match.group('gpu_power')))
            gpu_energy += float(match.group('gpu_power'))

            match = re.search('POM_5V_CPU (?P<cpu_power>[0-9]+)\/[0-9]+', line)
            cpu_power.append(float(match.group('cpu_power')))
            cpu_energy += float(match.group('cpu_power'))

    print('Average GPU Util (%): ', statistics.mean(gpu_utils))
    print('Average CPU Util (%): ', statistics.mean(cpu_utils))
    print('Average Mem Util (%): ', 100. * statistics.mean(mem_utils) / 3956)
    
    print('Average GPU Power (mW): ', statistics.mean(gpu_power))
    print('Average CPU Power (mW): ', statistics.mean(cpu_power))

    print('GPU Energy (J): ', gpu_energy / 1000.0)
    print('CPU Energy (J): ', cpu_energy / 1000.0)

    plt.plot(gpu_utils, color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('GPU Utilization (%)', color='red')
    
    ax2 = plt.twinx()
    ax2.plot(cpu_utils, color='blue')
    ax2.set_ylabel('CPU Utilization (%)', color='blue')

    plt.xlim([0, len(gpu_utils)])
    plt.title('Batch Size: {}'.format(args.batch_size))
    plt.show()

    plt.clf()

    plt.plot(mem_utils)
    plt.xlabel('Time (s)')
    plt.ylabel('Memory Utilization (%)')
    plt.title('Batch Size: {}'.format(args.batch_size))
    plt.show()

    plt.clf()

    plt.plot(gpu_power, color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('GPU Power (mW)', color='red')
    
    ax2 = plt.twinx()
    ax2.plot(cpu_power, color='blue')
    ax2.set_ylabel('CPU Power (mW)', color='blue')

    plt.xlim([0, len(cpu_power)])
    plt.ylim([0, max(cpu_power) + 50])
    plt.title('Batch Size: {}'.format(args.batch_size))
    plt.show()
    

    
    