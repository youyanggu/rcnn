import numpy as np
from sklearn.cross_validation import train_test_split
import sys

sys.path.append('../')
from main import get_ing_split

sys.path.append('../../../adulteration/model')
sys.path.append('../../../adulteration/wikipedia')
from hier_to_cat import test_model
from wikipedia import get_adulterants, get_ings_wiki_links, get_ings
wiki_path = '../../../adulteration/wikipedia/'

def softmax(w):
    w = np.array(w)
    maxes = np.amax(w, axis=1)
    maxes = maxes.reshape(maxes.shape[0], 1)
    e = np.exp(w - maxes)
    dist = e / np.sum(e, axis=1).reshape(-1,1)
    return dist

def p_y_given_x(X, w):
    return softmax(np.dot(X, w))

def unseen_wiki_articles(seed=42):
    ing_wiki_links = get_ings_wiki_links()
    ings_train, ings_dev, adulterants = get_ing_split(seed=seed)

    train_wiki_articles = set([ing_wiki_links[i][0] for i in ings_train])
    present = 0
    not_present = 0
    for i in ings_dev:
        if ing_wiki_links[i][0] in train_wiki_articles:
            present += 1
        else:
            not_present += 1
            print i, ing_wiki_links[i]
    print present, not_present


def load_reps(model_id, dataset):
    assert dataset in ['train', 'dev', 'test']
    fname = 'representations/{}_{}_ing_reps.npy'.format(model_id, dataset)
    reps = np.load(fname)
    reps_prod = np.load(fname.replace('_ing_', '_prod_'))
    #reps_prod=np.random.random((131,50))
    
    ing_wiki_links = get_ings_wiki_links()
    with open('../../../adulteration/ncim/idx_to_cat.pkl', 'rb') as f_in:
        idx_to_cat = pickle.load(f_in)
    ings_train, ings_dev, adulterants = get_ing_split(seed=42)
    ings = {'train':ings_train, 'dev':ings_dev, 'test':adulterants}[dataset]
    
    predictions = p_y_given_x(reps, reps_prod.T)

    test_model(predictions, ings, idx_to_cat, top_n=3, fname=None, ings_wiki_links=ing_wiki_links)

