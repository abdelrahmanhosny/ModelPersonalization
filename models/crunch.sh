python3 models/explore.py --mode gpu --report=gpu_power
python3 models/explore.py --mode gpu --report=gpu_energy
python3 models/explore.py --mode gpu --report=gpu_util
python3 models/explore.py --mode gpu --report=cpu_power
python3 models/explore.py --mode gpu --report=cpu_energy
python3 models/explore.py --mode gpu --report=cpu_util
python3 models/explore.py --mode gpu --report=mem_util
python3 models/explore.py --mode gpu --report=time

python3 models/explore.py --mode gpu_freeze --report=gpu_power
python3 models/explore.py --mode gpu_freeze --report=gpu_energy
python3 models/explore.py --mode gpu_freeze --report=gpu_util
python3 models/explore.py --mode gpu_freeze --report=cpu_power
python3 models/explore.py --mode gpu_freeze --report=cpu_energy
python3 models/explore.py --mode gpu_freeze --report=cpu_util
python3 models/explore.py --mode gpu_freeze --report=mem_util
python3 models/explore.py --mode gpu_freeze --report=time

python3 models/explore.py --mode cpu --report=gpu_power
python3 models/explore.py --mode cpu --report=gpu_energy
python3 models/explore.py --mode cpu --report=gpu_util
python3 models/explore.py --mode cpu --report=cpu_power
python3 models/explore.py --mode cpu --report=cpu_energy
python3 models/explore.py --mode cpu --report=cpu_util
python3 models/explore.py --mode cpu --report=mem_util
python3 models/explore.py --mode cpu --report=time

exit
python3 models/explore.py --mode cpu_freeze --report=gpu_power
python3 models/explore.py --mode cpu_freeze --report=gpu_energy
python3 models/explore.py --mode cpu_freeze --report=gpu_util
python3 models/explore.py --mode cpu_freeze --report=cpu_power
python3 models/explore.py --mode cpu_freeze --report=cpu_energy
python3 models/explore.py --mode cpu_freeze --report=cpu_util
python3 models/explore.py --mode cpu_freeze --report=mem_util
python3 models/explore.py --mode cpu_freeze --report=time
