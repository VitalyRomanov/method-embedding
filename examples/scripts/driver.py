import subprocess
from subprocess import Popen, PIPE
import numpy as np


for i in np.linspace(0.01,1,10):
    # subprocess.check_call(['python', './mm1.py', str(i)])
    subprocess.check_call(['python', './main_attention_pool.py', f'{i}'])


