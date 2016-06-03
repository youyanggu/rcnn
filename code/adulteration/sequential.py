import argparse
import itertools
import numpy as np
import cPickle as pickle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
from scipy import optimize
from sklearn.cross_validation import train_test_split
import sys
import time

sys.path.append('../')
from main import get_ing_split, convert_to_zero_one

sys.path.append('../../../adulteration/model')
sys.path.append('../../../adulteration/wikipedia')
from askubuntu import get_similar_reps, load_q2q, get_counts
from hier_to_cat import test_model
from scoring import evaluate_map
from split_data import split_data_by_wiki
from wikipedia import get_adulterants, get_ings_wiki_links, get_ings
wiki_path = '../../../adulteration/wikipedia/'

def reshape(arr, iterations_per_ing):
    assert len(arr) % iterations_per_ing == 0
    n = len(arr)
    return np.mean(np.reshape(arr, (n/iterations_per_ing,iterations_per_ing)), axis=1)

def box_whisker_plot(save_id, maps_final_orig, map_improvements_final, 
    iterations_per_ing, split=[0.2, 0.4, 0.6, 0.8, 1]):
    assert len(maps_final_orig) == len(map_improvements_final)
    n = len(maps_final_orig)
    maps_final_orig_ = reshape(maps_final_orig, iterations_per_ing)
    map_improvements_final_ = reshape(map_improvements_final, iterations_per_ing)
    num_boxes = len(split)
    inds = np.digitize(maps_final_orig_, split, right=True)
    data = [[] for i in range(num_boxes)]
    for i,v in enumerate(map_improvements_final_):
        data[inds[i]].append(v)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bp = ax.boxplot(data)#, whis=[2,98])
    print "Boxplot quintile sizes:", [len(i) for i in data]
    for box in bp['boxes']:
        box.set(color='skyblue', linewidth=2)
    for median in bp['medians']:
        median.set(color='firebrick', linewidth=2)
    for whisker in bp['whiskers']:
        whisker.set(color='skyblue', linewidth=2)
    #plt.ylim((-1,1))
    #ax.set_yticks(np.arange(-1, 1.05, 0.1), minor=True)
    #ax.grid(which='minor', alpha=0.5)
    plt.xlabel('baseline MAP quintile')
    plt.ylabel('improvement over baseline MAP')
    plt.grid()
    ax.set_xticklabels(['[0,0.2]', '(0.2,0.4]', '(0.4,0.6]', '(0.6,0.8]', '(0.8,1]'])
    plt.savefig('boxplots/boxplot_{}.png'.format(save_id))
    

def load_saved_data(save_id):
    observed_ings = np.load('seq_results/observed_ings_{}.npy'.format(save_id))
    maps_final_orig = np.load('seq_results/maps_final_orig_{}.npy'.format(save_id))
    map_improvements_final = np.load('seq_results/map_improvements_final_{}.npy'.format(save_id))
    map_improvements_rand = np.load('seq_results/map_improvements_final_{}.npy'.format(save_id))
    num_pos = np.load('seq_results/num_pos_{}.npy'.format(save_id))
    all_l2_reg = np.load('seq_results/all_l2_reg_{}.npy'.format(save_id))
    return observed_ings, maps_final_orig, map_improvements_final, map_improvements_rand, num_pos, all_l2_reg

def combine_reps(reps1, reps2, reps3, indices1, indices2, indices3):
    reps_all = np.zeros((5439, reps1.shape[1]))
    for reps, indices in zip([reps1, reps2, reps3], [indices1, indices2, indices3]):
        for idx, j in enumerate(indices):
            #if j >= 1000:
            #    break
            assert reps_all[j].sum()==0
            reps_all[j] = reps[idx]
    return reps_all


def sigmoid(w):
    w = np.array(w)
    return 1 / (1 + np.exp(-w))

def softmax(w):
    w = np.array(w)
    maxes = np.amax(w, axis=1)
    maxes = maxes.reshape(maxes.shape[0], 1)
    e = np.exp(w - maxes)
    dist = e / np.sum(e, axis=1).reshape(-1,1)
    return dist

def p_y_given_x(a, b, binary, w=None):
    f = sigmoid if binary else softmax 
    if w is not None:
        w = np.diag(w)
        return f(np.dot(np.dot(a, w), b))
    else:
        return f(np.dot(a, b))

def gen_sequence(counts, add_negatives=False, max_positives=None, max_total=None, max_negatives_ratio=1, use_pair=False):
    """Generates random sequence of product categories from the distribution 
    in the input. First observation in sequence is always a positive.

    Input:
        counts - (num_categories,) numpy vector of product category counts.
        add_negatives - if category i is a negative instance (cannot occur), then 
            -(i+1) is added to the sequence.
        max_positives - max number of positive observations to use.
        max_total - max total observations.
        max_negatives_ratio - # of negatives observations <= # positives * max_negatives_ratio
    Output:
        sequence of product category indices. if use_pair, returns list. otherwise, returns numpy array.
    """
    assert not (use_pair and not add_negatives)
    num_categories = len(counts)
    all_positives = np.where(counts>0)[0]
    assert len(all_positives)>1
    np.random.shuffle(all_positives)
    positives = np.array_split(all_positives, 2)[0] # split positive instances into 2
    #positives = all_positives
    if max_positives:
        positives = positives[:max_positives]
    #first_positive = positives[0]
    if add_negatives:
        negatives = [-(i+1) for i in range(num_categories) if i not in all_positives]
        np.random.shuffle(negatives)
        if use_pair:
            max_len = min(len(positives), len(negatives))
            #sequence = zip(positives[:max_len], negatives[:max_len])
            sequence = list(itertools.product(positives[:max_len], negatives[:max_len]))
        else:
            sequence = np.append(positives, negatives[:int(len(positives)*max_negatives_ratio)])
    else:
        sequence = positives
    np.random.shuffle(sequence)
    if max_total:
        sequence = sequence[:max_total]
    #sequence = np.append(first_positive, sequence)
    return sequence

def get_baseline_loss(ing_idx, pred, sequence, ing_cat_pair_map):
    """Compute the baseline loss (with w=eye(d)) given a constant prediction distribution 
    and a sequence of product categories."""
    assert len(sequence)>0
    remaining_indices = range(len(pred))
    
    if type(sequence[0]) == tuple:
        sequence = list(sum(sequence, ())) # flatten sequence
    
    for t, y in enumerate(sequence):
        if y>=0 and y in remaining_indices:
            remaining_indices.remove(y)
        elif -(y+1) in remaining_indices:
            remaining_indices.remove(-(y+1))
        assert len(remaining_indices) > 0

    losses = []
    for i in remaining_indices:
        if (ing_idx, i) in ing_cat_pair_map:
            prob = pred[i]
            loss = -np.log(prob)
            losses.append(loss)
        else:
            prob = 1 - pred[i]
            loss = -np.log(prob)
            losses.append(loss)
    loss = np.mean(losses)
    return loss

def run_naive_online(reps, reps_prod, binary, sequence, multiplier):
    """Increases the prob of a category by a multiplier at each step."""
    losses = []
    pred = p_y_given_x(reps, reps_prod.T, binary)[0]
    for t, y in enumerate(sequence):
        if y >= 0:
            prob = pred[y]
            loss = -np.log(prob)
            losses.append(loss)
        else:
            prob = 1 - pred[-(y+1)]
            loss = -np.log(prob)
            losses.append(loss)
        if y >= 0:
            pred[y] *= 1 + (multiplier-1.)/np.sqrt(t+1)
        else:
            pred[y] *= 1 - (multiplier-1.)/np.sqrt(t+1)
        if binary:
            pred[y] = min(pred[y], 1)
        else:
            pred = pred / pred.sum()
            assert abs(pred.sum()-1) < 1e-3
    return np.mean(losses), pred

def run_online(ing_idx, reps, reps_prod, binary, skip_online_updates, use_pair, sequence, batch,
    l2_reg, learn_lambda_min_pos, maxiter, step_size, method, lower_bound, upper_bound, ing_cat_pair_map=None):
    def loss_func1(w):
        pred = p_y_given_x(reps, reps_prod.T, binary, w)[0]
        #prob = pred[batch_y]
        prob = pred[[i if i>=0 else -(i+1) for i in batch_y]]
        prob = [p if batch_y[i]>=0 else 1-p for i,p in enumerate(prob)]
        loss = -(np.sum(np.log(prob)) - l2_reg/2*np.sum((w-1)**2))#-np.sum(np.log(w)-w+1))
        return loss
    def loss_func2(w):
        prob = []
        for pos_idx, neg_idx in batch_y:
            neg_idx = -(neg_idx + 1)
            v = p_y_given_x(reps, (reps_prod[pos_idx]-reps_prod[neg_idx]).T, binary, w)[0]
            prob.append(v)
        loss = -(np.sum(np.log(prob)) - l2_reg/2*np.sum((w-1)**2))#-np.sum(np.log(w)-w+1))
        return loss

    sequence = np.array(sequence)
    d = reps.shape[1]
    w = np.ones(d)
    eps = np.sqrt(np.finfo(float).eps)
    num_categories = reps_prod.shape[0]
    seen_indices = set()
    remaining_indices = range(num_categories)
    orig_pred = p_y_given_x(reps, reps_prod.T, binary, w)[0]
    pred = orig_pred
    loss_func = loss_func2 if use_pair else loss_func1
    ops = {}
    if maxiter:
        ops['maxiter'] = maxiter
    if step_size:
        ops['eps'] = step_size
    if method is None:
        method = 'BFGS' if (lower_bound is None and upper_bound is None) else 'L-BFGS-B'
        
    # Find optimum lambda
    lambdas = np.array([0.085, 0.09, 0.095, 0.1, 0.105, 0.11, 0.115])
    #lambdas = np.array([0.01, 0.03, 0.05, 0.07, 0.1, 0.14, 0.18, 0.22])
    lambda_scores = []
    print len(sequence)
    num_pos = len(sequence) if use_pair else (sequence>=0).sum()
    if not use_pair and learn_lambda_min_pos and num_pos >= learn_lambda_min_pos and num_pos < 20:
        # Choose best lambda via cross-validation.
        for l2_reg in lambdas:
            map_left_outs = []
            for left_out_idx in range(len(sequence)):
                if use_pair:
                    left_out_pos_idx = sequence[left_out_idx][0]
                else:
                    left_out_pos_idx = sequence[left_out_idx]
                if not use_pair and left_out_pos_idx<0:
                    continue
                w = np.ones(d)
                new_sequence = np.delete(sequence, left_out_idx, axis=0)
                batch_y = new_sequence
                res = optimize.minimize(loss_func, w, 
                    method=method, 
                    options=ops,
                    bounds=[(lower_bound, upper_bound)]*len(w)
                )
                w = res.x
                pred_lambda = p_y_given_x(reps, reps_prod.T, binary, w)[0]
                map_left_out = 1./(np.where(np.argsort(-pred_lambda)==left_out_pos_idx)[0][0]+1)
                map_left_outs.append(map_left_out)
            print map_left_outs
            lambda_scores.append(np.mean(map_left_outs))
            #print l2_reg, map_left_outs, pred_lambda, w
        
        lambda_scores = np.array(lambda_scores)
        best_lambda = lambdas[np.where(lambda_scores==max(lambda_scores))[0][0]]
        #lambda_weights = np.exp2(-(len(lambda_scores)-1-np.argsort(np.argsort(lambda_scores))))
        lambda_weights = np.argsort(np.argsort(lambda_scores))
        #avg_lambda = (lambda_weights*lambdas).sum() / lambda_weights.sum()
        avg_lambda = lambdas[np.argsort(-lambda_scores)[:3]].mean()
        print best_lambda, avg_lambda, lambda_scores
        l2_reg = avg_lambda #best_lambda#orig_lambda

    map_1_orig = map_1_new = 0
    losses = []
    assert len(sequence) > 0
    for t, y in enumerate(sequence):
        if use_pair:
            pos_idx, neg_idx = y
        if not skip_online_updates:
            if use_pair:
                if pos_idx not in seen_indices:
                    losses.append(-np.log(pred[pos_idx]))
                    seen_indices.add(pos_idx)
                if neg_idx not in seen_indices:
                    losses.append(-np.log(1-pred[-(neg_idx+1)]))
                    seen_indices.add(neg_idx)
            else:
                if y not in seen_indices:
                    if y >= 0:
                        prob = pred[y]
                        loss = -np.log(prob)
                        losses.append(loss)
                    else:
                        prob = 1 - pred[-(y+1)]
                        loss = -np.log(prob)
                        losses.append(loss)
                    seen_indices.add(y)
        
        if use_pair:
            if pos_idx in remaining_indices:
                remaining_indices.remove(pos_idx)
            if -(neg_idx+1) in remaining_indices:
                remaining_indices.remove(-(neg_idx+1))
        else:

            if y>=0 and y in remaining_indices:
                remaining_indices.remove(y)
            elif -(y+1) in remaining_indices:
                remaining_indices.remove(-(y+1))
        assert len(remaining_indices) > 0

        if skip_online_updates and t != len(sequence)-1:
            continue

        if t in [1, len(sequence)-1]:
            print "Random"
            map_score_rand, prec_at_n_score_rand = evaluate_map(
                [ing_idx], [pred], ing_cat_pair_map, random=True, results_indices=remaining_indices)
            print "Orig Pred"
            map_score_orig, prec_at_n_score_orig = evaluate_map(
                [ing_idx], [orig_pred], ing_cat_pair_map, results_indices=remaining_indices)
            if map_score_orig - map_score_rand > 0.3:
                # if baseline prediction is good, set higher regularizer
                l2_reg *= min(5, map_score_orig / map_score_rand) 

        batch_y = sequence[t:max(0,t-batch):-1]
        res = optimize.minimize(loss_func, w, 
            method=method, 
            options=ops,
            bounds=[(lower_bound, upper_bound)]*len(w)
        )
        w = res.x

        pred = p_y_given_x(reps, reps_prod.T, binary, w)[0]
        if t in [1, len(sequence)-1]:
            print "Num observations:", t+1
            print "Remaining categories:", len(remaining_indices)
            #print "Random"
            #map_score_rand, prec_at_n_score_rand = evaluate_map(
            #    [ing_idx], [pred], ing_cat_pair_map, random=True, results_indices=remaining_indices)
            #print "Orig Pred"
            #map_score_orig, prec_at_n_score_orig = evaluate_map(
            #    [ing_idx], [orig_pred], ing_cat_pair_map, results_indices=remaining_indices)
            print "New Pred"
            map_score_new, prec_at_n_score_new = evaluate_map(
                [ing_idx], [pred], ing_cat_pair_map, results_indices=remaining_indices)
            #map_improvement = map_score_new - map_score_orig
            if t==1 or len(sequence)==1:
                map_1_orig = map_score_orig
                map_1_new = map_score_new
            if t == len(sequence) - 1:
                map_final_orig = map_score_orig
                map_final_new = map_score_new
    for i in remaining_indices:
        if (ing_idx, i) in ing_cat_pair_map:
            prob = pred[i]
            loss = -np.log(prob)
            losses.append(loss)
        else:
            prob = 1 - pred[i]
            loss = -np.log(prob)
            losses.append(loss)
    print "\nPrior"
    map_prior, prec_at_n_prior = evaluate_map(
                [ing_idx], [orig_pred], ing_cat_pair_map)
    print "Posterior"
    map_posterior, prec_at_n_posterior = evaluate_map(
                [ing_idx], [pred], ing_cat_pair_map)
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
    return np.mean(losses), w, l2_reg, map_1_orig, map_1_new, \
        map_final_orig, map_final_new, map_prior, map_posterior, map_score_rand

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

def main(args):
    print args
    use_args = True
    test_adulterants_only = args.test_adulterants_only
    add_adulterants = args.add_adulterants
    model_id = args.model_id
    dataset = args.dataset
    seed = args.seed
    use_askubuntu = args.use_askubuntu
    skip_online_updates = args.skip_online_updates
    iterations_per_ing = args.iterations_per_ing
    l2_reg = args.l2_reg
    learn_lambda_min_pos = args.learn_lambda_min_pos
    use_pair = args.use_pair
    lower_bound = args.lower_bound
    upper_bound = args.upper_bound
    max_positives = args.max_positives
    max_total = args.max_total
    max_negatives_ratio = args.max_negatives_ratio
    max_iterations = args.max_iterations
    save_id = args.save_id

    maxiter = None#100
    step_size = None#1e-5
    add_negatives = True
    batch_k = 10
    method = None
    binary = True

    assert dataset in ['train', 'dev', 'test']
    if use_askubuntu:
        num_categories = 20
        reps_fname = 'askubuntu_vectors/model_run{}.{}'.format(model_id, dataset)
        similar_fname = 'askubuntu_vectors/similar_{}.txt'.format(dataset)
        reps, similar_qs, q20s, bm25s = load_q2q(reps_fname, similar_fname)
        indices = q20s.keys()
        ing_cat_pair_map = {}
        for q_idx in indices:
            q_q20 = q20s[q_idx]
            similar_q = similar_qs[q_idx]
            for i, q in enumerate(q_q20):
                if q in similar_q:
                    ing_cat_pair_map[(q_idx, i)] = True
    else:
        # Ingredients
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
        with open(wiki_path+'input_to_outputs.pkl', 'r') as f_in:
            input_to_outputs = pickle.load(f_in)
        with open(wiki_path+'input_to_outputs_adulterants.pkl', 'r') as f_in:
            input_to_outputs_adulterants = pickle.load(f_in)
            idx = 5000
            for i in range(len(input_to_outputs_adulterants)):
                v = input_to_outputs_adulterants[i]
                if v.sum() > 0:
                    input_to_outputs[idx] = v
                    idx += 1
        
        ing_cat_pair_map = {}
        for inp, out in input_to_outputs.iteritems():
            for cat_idx in np.where(out>0)[0]:
                ing_cat_pair_map[(inp, cat_idx)] = True

        # Get ingredient names
        #ings_train, ings_dev, adulterants = get_ing_split(seed=42)
        if dataset == 'test' and test_adulterants_only:
            ings = get_adulterants()
        else:
            ings = get_ings(num_ingredients)
            if add_adulterants:
                ings = np.hstack([ings, get_adulterants()])
        
        # Get indices
        #train_indices, dev_indices = train_test_split(
        #    range(num_ingredients), test_size=1/3., random_state=seed)
        train_indices, dev_indices, test_indices = split_data_by_wiki(
            ings, seed)
        if dataset == 'test' and test_adulterants_only:
            test_indices = range(len(ings))
        indices = {
                    'train' : train_indices,
                    'dev'   : dev_indices,
                    'test'  : test_indices
        }[dataset]
        predictions = p_y_given_x(reps, reps_prod.T, binary)
        assert len(predictions) == len(indices)
    
    if not use_args:
        if use_askubuntu:
            skip_online_updates = True
            iterations_per_ing = 5
            l2_reg = 0.015#0.015
            learn_lambda_min_pos = 5
            maxiter = None
            step_size = None
            add_negatives = True
            use_pair = True
            batch_k = 10
            method = None
            lower_bound = 0
            upper_bound = 2
            max_positives = 20
            max_total = None
            max_negatives_ratio = 1
            max_iterations = 200
            save_id = 2
        else:
            #print_predictions = True
            skip_online_updates = True
            iterations_per_ing = 20
            l2_reg = 0.1 #0.5
            learn_lambda_min_pos = 9999
            maxiter = None#100
            step_size = None#1e-5
            add_negatives = True
            use_pair = False
            batch_k = 10
            method = None
            lower_bound = 0
            upper_bound = 2
            max_positives = 20
            max_total = 100
            max_negatives_ratio = 6
            max_iterations = 200
            save_id = 1
    all_min_losses = [] # loss using true distrubution
    all_uniform_losses = [] # loss using uniform distribution
    all_baseline_losses = [] # loss using static batch prediction
    #all_naive_online_losses = [] # loss that just increases the prob of that product with each update
    all_batchall_losses = [] # loss using bayesian update (batch=all)
    #all_batchk_losses = [] # loss using bayesian update (batch=k)
    #all_online_losses = [] # loss using online learning (batch=1)
    map_improvements_1 = []
    map_improvements_final = []
    map_improvements_post = []
    map_improvements_rand = [] # map_baseline - map_random
    maps_1_orig = []
    maps_final_orig = []
    maps_prior_orig = []
    sequence_lens = []
    num_pos = []
    observed_ings = []
    final_weights = []
    all_l2_reg = []
    start_time = time.time()
    cur_iteration = 0
    indices2 = list(enumerate(indices))[::-1]
    #np.random.shuffle(indices2)
    for i, ind in indices2:#enumerate(indices):
        if not use_askubuntu and ind >= 5000:
            # skip adulterants
            continue
        cur_iteration += 1
        if max_iterations and cur_iteration > max_iterations:
            break
        iter_start_time = time.time()
        if use_askubuntu:
            rep = np.array([reps[ind]])
            reps_prod = get_similar_reps(reps, q20s[ind])
            pred = p_y_given_x(rep, reps_prod.T, binary)[0]
            counts = get_counts(similar_qs[ind], q20s[ind])
            ing = ind
        else:
            rep = reps[i:i+1]
            pred = predictions[i]
            counts = input_to_outputs[ind]
            ing = ings[ind]
        if len(np.where(counts>0)[0]) <= 1:
            continue # need at least 2 observations
        if use_pair and len(np.where(counts==0)[0]) == 0:
            continue # need at least 1 negative
        observed_ings.append(ing)
        if binary:
            true_dist = convert_to_zero_one(counts)
        else:
            true_dist = counts.astype(float) / counts.sum()
        
        print '=========================================='
        print i, ind, ing
        for iteration in range(iterations_per_ing):
            print "Iteration:", iteration+1
            sequence = gen_sequence(counts, add_negatives, max_positives, max_total, max_negatives_ratio, use_pair)
            if use_pair and max_total:
                # filter by pair scores
                seq_scores = [pred[pos_idx]-pred[neg_idx] for pos_idx, neg_idx in sequence]
                sequence = [sequence[i] for i in sorted(np.argsort(seq_scores)[:max_total])]
            #if (counts>0).sum() > 5:
            #    l2_reg = 0.01
            if iteration == 0:
                if use_pair:
                    print "# pairs: {}, {} total Pos".format(len(sequence), (counts>0).sum())
                else:
                    print "# Pos in seq: {} / {}, {} total Pos".format((sequence>=0).sum(), len(sequence), (counts>0).sum())
                sequence_lens.append(len(sequence))
                num_pos.append((counts>0).sum())
                min_loss = get_baseline_loss(ind, true_dist, sequence, ing_cat_pair_map) # should be 0 for binary
                if binary:
                    uniform_loss = get_baseline_loss(ind, np.ones(num_categories)/2., sequence, ing_cat_pair_map)
                else:
                    uniform_loss = get_baseline_loss(ind, np.ones(num_categories, dtype=float)/num_categories, sequence, ing_cat_pair_map)
                baseline_loss = get_baseline_loss(ind, pred, sequence, ing_cat_pair_map)
            #naive_online_loss, naive_pred = run_naive_online(rep, reps_prod, binary, sequence, naive_mutiplier)
            
        
            batchall_loss, w_batchall, new_l2_reg, map_1_orig, map_1_new, map_final_orig, \
                map_final_new, map_prior, map_posterior, map_rand = run_online(
                ind, rep, reps_prod, binary, skip_online_updates, use_pair, sequence, len(sequence), l2_reg, \
                learn_lambda_min_pos, maxiter, step_size, method, lower_bound, upper_bound, ing_cat_pair_map)
            map_improvement_1 = map_1_new - map_1_orig
            map_improvement_final = map_final_new - map_final_orig
            map_improvement_post = map_posterior - map_prior
            #batchk_loss, w_batchk = run_online(rep, reps_prod, binary, use_pair, sequence, batch_k, l2_reg, step_size, method, lower_bound, upper_bound)
            #online_loss, w_online = run_online(rep, reps_prod, binary, use_pair, sequence, 1, l2_reg, step_size, method, lower_bound, upper_bound)
            all_l2_reg.append(new_l2_reg)
            map_improvements_1.append(map_improvement_1)
            map_improvements_final.append(map_improvement_final)
            map_improvements_post.append(map_improvement_post)
            map_improvements_rand.append(map_final_orig-map_rand)
            maps_1_orig.append(map_1_orig)
            maps_final_orig.append(map_final_orig)
            maps_prior_orig.append(map_prior)

            all_min_losses.append(min_loss)
            all_uniform_losses.append(uniform_loss)
            all_baseline_losses.append(baseline_loss)
            #all_naive_online_losses.append(naive_online_loss)
            all_batchall_losses.append(batchall_loss)
            #all_batchk_losses.append(batchk_loss)
            #all_online_losses.append(online_loss)

        if not use_askubuntu:# and print_predictions:
            pos_cats = [idx_to_cat[pos_idx] for pos_idx in sequence[sequence>=0]]
            print "Observed cats:", pos_cats
            # True distribution
            test_model(true_dist.reshape(1,-1), [ing], idx_to_cat, top_n=10, ings_wiki_links=ings_wiki_links)
            # Baseline prediction
            test_model(pred.reshape(1,-1), [ing], idx_to_cat, top_n=10, ings_wiki_links=ings_wiki_links)
            # Naive algo prediction
            #test_model(naive_pred.reshape(1,-1), [ing], idx_to_cat, top_n=10, ings_wiki_links=ings_wiki_links)
            # After batchall update
            test_model(p_y_given_x(rep, reps_prod.T, binary, w_batchall), [ing], idx_to_cat, top_n=10, ings_wiki_links=ings_wiki_links)
            # After batchk update
            #test_model(p_y_given_x(rep, reps_prod.T, binary, w_batchk), [ing], idx_to_cat, top_n=10, ings_wiki_links=ings_wiki_links)
            # After online update
            #test_model(p_y_given_x(rep, reps_prod.T, binary, w_online), [ing], idx_to_cat, top_n=10, ings_wiki_links=ings_wiki_links)
        print "True loss      :", min_loss
        print "Uniform loss   :", uniform_loss
        print "Baseline loss  :", baseline_loss
        #print "Naive Algo loss:", naive_online_loss
        print "Batch all loss :", batchall_loss
        #print "Batch k loss   :", batchk_loss
        #print "Online loss    :", online_loss
        print w_batchall[:10]#, w_batchk[:10], w_online[:10]
        final_weights.append(w_batchall)
        
        print "Iter time elapsed: {:.1f}s".format((time.time()-iter_start_time))
        print "-------------------------------------------------------"
        print "Mean True Loss      :", np.mean(all_min_losses)
        print "Mean Uniform Loss   :", np.mean(all_uniform_losses)
        print "Mean Baseline Loss  :", np.mean(all_baseline_losses)
        #print "Mean Naive Algo Loss:", np.mean(all_naive_online_losses)
        print "Mean Batch all Loss :", np.mean(all_batchall_losses)
        #print "Mean Batch k Loss   :", np.mean(all_batchk_losses)
        #print "Mean Online Loss    :", np.mean(all_online_losses)
        print "Mean seq len / Num positives: {:.1f} / {:.1f}".format(np.mean(sequence_lens), np.mean(num_pos))
        print ''
        print "Mean l2 reg          :", np.mean(all_l2_reg)
        print "Mean map_prior       :", np.mean(maps_prior_orig)
        print "Mean maps_1_orig     :", np.mean(maps_1_orig)
        print "Mean map_final_orig  :", np.mean(maps_final_orig)
        print "Mean map_improvements_1     :", np.mean(map_improvements_1)
        print "Mean map_improvements_final :", np.mean(map_improvements_final)
        print "Mean map_improvements_post :", np.mean(map_improvements_post)
        print "Mean map_improvements_rand :", np.mean(map_improvements_rand)
        print "% map_improvements_1 > 0    :", (np.array(map_improvements_1) > 0).mean(), (np.array(map_improvements_1) == 0).mean(), (np.array(map_improvements_1) < 0).mean()
        print "% map_improvements_final > 0:", (np.array(map_improvements_final) > 0).mean(), (np.array(map_improvements_final) == 0).mean(), (np.array(map_improvements_final) < 0).mean()
        print "% map_improvements_post > 0 :", (np.array(map_improvements_post) > 0).mean(), (np.array(map_improvements_post) == 0).mean(), (np.array(map_improvements_post) < 0).mean()
    print "Time elapsed: {:.1f}m".format((time.time()-start_time)/60)
    if save_id:
        np.save('seq_results/observed_ings_{}.npy'.format(save_id), np.array(observed_ings))
        np.save('seq_results/maps_final_orig_{}.npy'.format(save_id), np.array(maps_final_orig))
        np.save('seq_results/map_improvements_final_{}.npy'.format(save_id), np.array(map_improvements_final))
        np.save('seq_results/map_improvements_rand_{}.npy'.format(save_id), np.array(map_improvements_rand))
        np.save('seq_results/num_pos_{}.npy'.format(save_id), np.array(num_pos))
        np.save('seq_results/all_l2_reg_{}.npy'.format(save_id), np.array(all_l2_reg))
    os.system('say "Done done done."')
    box_whisker_plot(save_id, maps_final_orig, map_improvements_final, iterations_per_ing)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_id",
            type = int,
        )
    argparser.add_argument("--dataset",
            type = str,
            help = "'train', 'dev', or 'test'"
        )
    argparser.add_argument("--seed",
            type = int,
            default = 42,
            help = "random seed of the model"
        )
    argparser.add_argument("--add_adulterants",
            action='store_true',
            help = "add adulterants to training"
        )
    argparser.add_argument("--test_adulterants_only",
            action='store_true',
            help = "test using adulterants only"
        )
    argparser.add_argument("--use_askubuntu",
            action='store_true',
            help = "train on askubuntu data"
        )
    argparser.add_argument("--skip_online_updates",
            action='store_true',
        )
    argparser.add_argument("--iterations_per_ing",
            type = int,
        )
    argparser.add_argument("--l2_reg",
            type = float,
        )
    argparser.add_argument("--learn_lambda_min_pos",
            type = int,
        )
    argparser.add_argument("--use_pair",
            action='store_true',
        )
    argparser.add_argument("--lower_bound",
            type = float,
        )
    argparser.add_argument("--upper_bound",
            type = float,
        )
    argparser.add_argument("--max_positives",
            type = int,
        )
    argparser.add_argument("--max_total",
            type = int,
        )
    argparser.add_argument("--max_negatives_ratio",
            type = float,
        )
    argparser.add_argument("--max_iterations",
            type = int,
        )
    argparser.add_argument("--save_id",
            type = int,
        )
    args = argparser.parse_args()
    main(args)

