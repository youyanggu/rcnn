import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn.cross_validation import train_test_split
import sys
import time

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

def p_y_given_x(a, b, w=None):
    if w is not None:
        w = np.diag(w)
        return softmax(np.dot(np.dot(a, w), b))
    else:
        return softmax(np.dot(a, b))

def gen_sequence(counts, num_in_sequence=None, random=True):
    """Generates random sequence of product categories from the distribution 
    in the input.

    Input:
        counts - (num_categories,) numpy vector of product category counts.
        num_in_sequence - if this is an int, then scale sum of counts to num_in_sequence.
    Output:
        sequence of product category indices
    """
    sequence = np.repeat(np.arange(len(counts)).reshape(-1,1), counts)
    if num_in_sequence:
        sequence = np.random.choice(sequence, num_in_sequence)
    if random:
        np.random.shuffle(sequence)
    return sequence

def get_baseline_loss(pred, sequence):
    """Compute the baseline loss (with w=eye(d)) given a constant prediction distribution 
    and a sequence of product categories."""
    probs = pred[sequence]
    loss = np.mean(-np.log(probs))
    return loss

def run_naive_online(reps, reps_prod, sequence, multiplier):
    """Increases the prob of a category by a multiplier at each step."""
    losses = []
    pred = p_y_given_x(reps, reps_prod.T)[0]
    for t, y in enumerate(sequence):
        prob = pred[y]
        loss = -np.log(prob)
        losses.append(loss)
        pred[y] *= 1 + (multiplier-1.)/np.sqrt(t+1)
        pred = pred / pred.sum()
        assert abs(pred.sum()-1) < 1e-3
    return np.mean(losses), pred

def run_online(reps, reps_prod, sequence, batch, l2_reg, step_size):
    def loss_func(w, print_prob=False):
        pred = p_y_given_x(reps, reps_prod.T, w)[0]
        prob = pred[batch_y]
        if print_prob:
            print y, pred, prob
        loss = -(np.sum(np.log(prob)) + l2_reg*np.sum(np.log(w)-w+1))
        return loss

    d = reps.shape[1]
    w = np.ones(d)
    eps = np.sqrt(np.finfo(float).eps)

    losses = []
    for t, y in enumerate(sequence):
        pred = p_y_given_x(reps, reps_prod.T, w)[0]
        prob = pred[y]
        loss = -np.log(prob)
        losses.append(loss)
        batch_y = sequence[t:max(0,t-batch):-1]
        #print "Pre:", w, loss_func(w, True)
        gradient = optimize.approx_fprime(w, loss_func, eps)
        #print "Post:", gradient, w - step_size * gradient
        w = w - step_size / max(1,np.sqrt((t+1)/100)) * gradient
        #w = w - step_size * gradient
        w = w.clip(min=1e-8, max=2)
        #w = gradient_update()
    """
    plt.xlabel('t')
    plt.ylabel('mean loss up to t')
    if batch == 1:
        label = 'online'
        linestyle = '-'
    elif batch == 10:
        label = 'mini-batch'
        linestyle = '--'
    else:
        label = 'batch all'
        linestyle = ':'
    plt.plot(np.cumsum(losses) / np.arange(len(losses)), label=label, linestyle=linestyle)
    plt.legend(loc='upper right')
    plt.grid()
    """
    return np.mean(losses), w

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


def plot_stuff():
    l = len(all_online_losses)
    sequence_lens = [input_to_outputs[i].sum() for i in indices[:l]]
    ratios = [all_online_losses[i] / all_baseline_losses[i] for i in range(l)]
    plt.xlabel('sequence length')
    plt.ylabel('online loss / baseline loss')
    #plt.title('Correlation between sequence length and improvement in loss')
    plt.xscale('log')
    plt.plot(sequence_lens, ratios, '.')
    plt.grid()
    plt.show()

    plt.xlabel('true loss')
    plt.ylabel('online loss / baseline loss')
    #plt.title('Correlation between true loss and improvement in loss')
    plt.plot(all_min_losses, ratios, '.')
    plt.grid()
    plt.show()

def run(model_id, dataset):
    assert dataset in ['train', 'dev', 'test']
    num_ingredients = 5000
    num_categories = 131
    fname = 'representations/{}_{}_ing_reps.npy'.format(model_id, dataset)
    reps = np.load(fname)
    reps_prod = np.load(fname.replace('_ing_', '_prod_'))
    assert reps.shape[1] == reps_prod.shape[1]
    assert reps_prod.shape[0] == num_categories
    print "Dim:", reps.shape[1]
    
    ings_wiki_links = get_ings_wiki_links()
    with open('../../../adulteration/ncim/idx_to_cat.pkl', 'rb') as f_in:
        idx_to_cat = pickle.load(f_in)
    
    # get data of all input ingredient to output product category distributions.
    if dataset == 'test':
        with open(wiki_path+'input_to_outputs_adulterants.pkl', 'r') as f_in:
            input_to_outputs = pickle.load(f_in)
    else:
        with open(wiki_path+'input_to_outputs.pkl', 'r') as f_in:
            input_to_outputs = pickle.load(f_in)

    # Get ingredient names
    #ings_train, ings_dev, adulterants = get_ing_split(seed=42)
    ings = {
            'train' : get_ings(num_ingredients), 
            'dev'   : get_ings(num_ingredients), 
            'test'  : get_adulterants()
    }[dataset]
    
    # Get indices
    seed = 42
    train_indices, dev_indices = train_test_split(
        range(num_ingredients), test_size=1/3., random_state=seed)
    test_indices = [k for k,v in input_to_outputs.items() if v.sum()>0]
    indices = {
                'train' : train_indices,
                'dev'   : dev_indices,
                'test'  : test_indices
    }[dataset]
    
    predictions = p_y_given_x(reps, reps_prod.T)
    assert len(predictions) == len(indices)

    #test_model(pred, ings, idx_to_cat, top_n=3, fname=None, ings_wiki_links=ings_wiki_links)

    run_test_model = True
    naive_mutiplier = 1.02
    l2_reg = 0.01
    step_size = 0.1
    num_in_sequence = None#1000
    batch_k = 10
    run_limit = None
    all_min_losses = [] # loss using true distrubution
    all_uniform_losses = [] # loss using uniform distribution
    all_baseline_losses = [] # loss using static batch prediction
    all_naive_online_losses = [] # loss that just increases the prob of that product with each update
    all_batchall_losses = [] # loss using bayesian update (batch=all)
    all_batchk_losses = [] # loss using bayesian update (batch=k)
    all_online_losses = [] # loss using online learning (batch=1)
    start_time = time.time()
    for i, ind in enumerate(indices[:run_limit]):
        ing = ings[ind]
        pred = predictions[i]
        counts = input_to_outputs[ind]
        true_dist = counts.astype(float) / counts.sum()
        sequence = gen_sequence(counts, num_in_sequence)
        print '=========================================='
        print i, ind, ing, len(sequence)

        min_loss = get_baseline_loss(true_dist, sequence)
        uniform_loss = get_baseline_loss(np.ones(num_categories, dtype=float)/num_categories, sequence)
        baseline_loss = get_baseline_loss(pred, sequence)
        naive_online_loss, naive_pred = run_naive_online(reps[i:i+1], reps_prod, sequence, naive_mutiplier)
        batchall_loss, w_batchall = run_online(reps[i:i+1], reps_prod, sequence, len(sequence), l2_reg, step_size)
        batchk_loss, w_batchk = run_online(reps[i:i+1], reps_prod, sequence, batch_k, l2_reg, step_size)
        online_loss, w_online = run_online(reps[i:i+1], reps_prod, sequence, 1, l2_reg, step_size)
        
        if run_test_model:
            # True distribution
            test_model(true_dist.reshape(1,-1), [ing], idx_to_cat, top_n=3, ings_wiki_links=ings_wiki_links)
            # Baseline prediction
            test_model(pred.reshape(1,-1), [ing], idx_to_cat, top_n=3, ings_wiki_links=ings_wiki_links)
            # Naive algo prediction
            test_model(naive_pred.reshape(1,-1), [ing], idx_to_cat, top_n=3, ings_wiki_links=ings_wiki_links)
            # After batchall update
            test_model(p_y_given_x(reps[i:i+1], reps_prod.T, w_batchall), [ing], idx_to_cat, top_n=3, ings_wiki_links=ings_wiki_links)
            # After batchk update
            test_model(p_y_given_x(reps[i:i+1], reps_prod.T, w_batchk), [ing], idx_to_cat, top_n=3, ings_wiki_links=ings_wiki_links)
            # After online update
            test_model(p_y_given_x(reps[i:i+1], reps_prod.T, w_online), [ing], idx_to_cat, top_n=3, ings_wiki_links=ings_wiki_links)
        print "True loss      :", min_loss
        print "Uniform loss   :", uniform_loss
        print "Baseline loss  :", baseline_loss
        print "Naive Algo loss:", naive_online_loss
        print "Batch all loss :", batchall_loss
        print "Batch k loss   :", batchk_loss
        print "Online loss    :", online_loss
        print w_batchall[:10], w_batchk[:10], w_online[:10]
        all_min_losses.append(min_loss)
        all_uniform_losses.append(uniform_loss)
        all_baseline_losses.append(baseline_loss)
        all_naive_online_losses.append(naive_online_loss)
        all_batchall_losses.append(batchall_loss)
        all_batchk_losses.append(batchk_loss)
        all_online_losses.append(online_loss)
        break
    print "-------------------------------------------------------"
    print "Mean True Loss      :", np.mean(all_min_losses)
    print "Mean Uniform Loss   :", np.mean(all_uniform_losses)
    print "Mean Baseline Loss  :", np.mean(all_baseline_losses)
    print "Mean Naive Algo Loss:", np.mean(all_naive_online_losses)
    print "Mean Batch all Loss :", np.mean(all_batchall_losses)
    print "Mean Batch k Loss   :", np.mean(all_batchk_losses)
    print "Mean Online Loss    :", np.mean(all_online_losses)
    print "Time elapsed: {:.1f}m".format((time.time()-start_time)/60)


