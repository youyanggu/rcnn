
import sys
import gzip

import numpy as np

def say(s, stream=sys.stderr):
    stream.write("{}".format(s))
    stream.flush()

def load_embedding_iterator(path):
    file_open = gzip.open if path.endswith(".gz") else open
    count = 0
    with file_open(path) as fin:
        for line in fin:
            line = line.strip()
            if line:
                if count == 0:
                    count = 1
                    continue
                count += 1
                parts = line.split()
                word = parts[0]
                vals = np.array([ float(x) for x in parts[1:] ])
                yield word, vals

