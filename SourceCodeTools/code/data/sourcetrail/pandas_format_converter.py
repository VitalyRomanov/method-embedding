import sys
from SourceCodeTools.code.data.sourcetrail.file_utils import *

input_path = sys.argv[1]
target_format = sys.argv[2]

base_dir = os.path.dirname(input_path)
filename = os.path.basename(input_path)
output_path = os.path.join(base_dir, ".".join(filename.split(".")[:-1]) + "." + target_format)

data = unpersist(input_path)
persist(data, output_path)