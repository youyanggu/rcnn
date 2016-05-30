import numpy as np

def get_similar_reps(reps, similar_qs):
    return np.array([reps.get(i, np.zeros(400)) for i in similar_qs])

def load_q2q(reps_fname, similar_fname):
    """
    Load askubuntu question data.

    reps_fname - file name of question representations
    similar_fname - file name of similar question ids
    """
    q_reps = {}
    with open(reps_fname, 'r') as f:
        for line in f:
            q_str, reps_str = line.strip().split('\t')
            q = int(q_str)
            q_reps[q] = [float(i) for i in reps_str.split()]
    similar_qs = {}
    q20s = {}
    bm25s = {}
    with open(similar_fname, 'r') as f:
        for line in f:
            q_str, similar_q_str, q20_str, bm25_str = line.strip().split('\t')
            q = int(q_str)
            if q not in q_reps:
                continue # no similar q's
            similar_qs[q] = [int(i) for i in similar_q_str.split()]
            q20s[q] = [int(i) for i in q20_str.split()]
            bm25s[q] = [float(i) for i in bm25_str.split()]
    return q_reps, similar_qs, q20s, bm25s

def get_counts(similar_qs, q20s):
    counts = np.zeros(20)
    for i, q in enumerate(q20s):
        if q in similar_qs:
            counts[i] = 1
    return counts

