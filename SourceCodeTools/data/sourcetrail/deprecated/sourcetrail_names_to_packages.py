import sys, os
import pandas as pd
from csv import QUOTE_NONNUMERIC

nodes = pd.read_csv(sys.argv[1])

nodes['serialized_name'] = nodes['serialized_name'].apply(lambda x: x.split(".")[0])

nodes.to_csv(os.path.join(os.path.dirname(sys.argv[1]), "package_names.csv"), index=False, quoting=QUOTE_NONNUMERIC)