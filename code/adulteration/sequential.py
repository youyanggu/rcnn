import argparse
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
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

def gen_sequence(counts, add_negatives=False, max_positives=None, max_negatives_ratio=1):
    """Generates random sequence of product categories from the distribution 
    in the input. First observation in sequence is always a positive.

    Input:
        counts - (num_categories,) numpy vector of product category counts.
        add_negatives - if category i is a negative instance (cannot occur), then 
            -(i+1) is added to the sequence.
        max_positives - max number of positive observations to use.
        max_negatives_ratio - # of negatives observations <= # positives * max_negatives_ratio
    Output:
        sequence of product category indices
    """
    num_categories = len(counts)
    positives = np.where(counts>0)[0]
    assert len(positives)>1
    np.random.shuffle(positives)
    sequence = np.array_split(positives, 2)[0] # split positive instances into 2
    if max_positives:
        sequence = sequence[:max_positives]
    #first_positive = sequence[0]
    if add_negatives:
        negatives = [-(i+1) for i in range(num_categories) if i not in positives]
        sequence = np.append(sequence, negatives[:int(len(sequence)*max_negatives_ratio)])
    np.random.shuffle(sequence)
    #sequence = np.append(first_positive, sequence)
    return sequence

def get_baseline_loss(pred, sequence):
    """Compute the baseline loss (with w=eye(d)) given a constant prediction distribution 
    and a sequence of product categories."""
    probs = []
    for t, y in enumerate(sequence):
        if y >= 0:
            probs.append(pred[y])
        #else:
        #    probs.append(1 - pred[-(y+1)])
    loss = np.mean(-np.log(probs))
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
        #else:
        #    prob = 1 - pred[-(y+1)]
        #    loss = -np.log(prob)
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

def run_online(ing_idx, reps, reps_prod, binary, sequence, batch, l2_reg, maxiter, step_size, ing_cat_pair_map=None):
    def loss_func(w, print_prob=False):
        pred = p_y_given_x(reps, reps_prod.T, binary, w)[0]
        #prob = pred[batch_y]
        prob = pred[[i if i>=0 else -(i+1) for i in batch_y]]
        prob = [p if batch_y[i]>=0 else 1-p for i,p in enumerate(prob)]
        if print_prob:
            print y, pred, prob
        loss = -(np.sum(np.log(prob)) + l2_reg*np.sum(np.log(w)-w+1))
        return loss

    d = reps.shape[1]
    w = np.ones(d)
    eps = np.sqrt(np.finfo(float).eps)
    num_categories = reps_prod.shape[0]
    remaining_indices = range(num_categories)
    orig_pred = p_y_given_x(reps, reps_prod.T, binary, w)[0]
    pred = orig_pred

    losses = []
    assert len(sequence) > 0
    for t, y in enumerate(sequence):
        if y >= 0:
            prob = pred[y]
            loss = -np.log(prob)
            losses.append(loss)
        #else:
        #    prob = 1 - pred[-(y+1)]
        #    loss = -np.log(prob)
        batch_y = sequence[t:max(0,t-batch):-1]
        res = optimize.minimize(loss_func, w, 
            method='L-BFGS-B', 
            options={'maxiter': maxiter, 'eps':step_size}, 
            bounds=[(1e-8, 10)]*len(w)
        )
        w = res.x
        #print "Pre:", w, loss_func(w, True)
        #gradient = optimize.approx_fprime(w, loss_func, eps)
        #print "Post:", gradient, w - step_size * gradient
        #w = w - step_size / max(1,np.sqrt((t+1)/100)) * gradient
        #w = w - step_size * gradient
        w = w.clip(min=1e-8, max=10)
        #w = gradient_update()
        if y>=0:
            remaining_indices.remove(y)
        else:
            remaining_indices.remove(-(y+1))
        assert len(remaining_indices) > 0
        pred = p_y_given_x(reps, reps_prod.T, binary, w)[0]
        if t in [1, len(sequence)-1]:
            print "Num observations:", t+1
            print "Remaining categories:", len(remaining_indices)
            print "Random"
            map_score_rand, prec_at_n_score_rand = evaluate_map(
                [ing_idx], [pred], ing_cat_pair_map, random=True, results_indices=remaining_indices)
            print "Orig Pred"
            map_score_orig, prec_at_n_score_orig = evaluate_map(
                [ing_idx], [orig_pred], ing_cat_pair_map, random=False, results_indices=remaining_indices)
            print "New Pred"
            map_score_new, prec_at_n_score_new = evaluate_map(
                [ing_idx], [pred], ing_cat_pair_map, random=False, results_indices=remaining_indices)
            map_improvement = map_score_new - map_score_orig
            if t==1 or len(sequence)==1:
                map_improvement_1 = map_improvement
            if t == len(sequence) - 1:
                map_improvement_final = map_improvement
    for i in remaining_indices:
        if (ing_idx, i) in ing_cat_pair_map:
            prob = pred[i]
            loss = -np.log(prob)
            losses.append(loss)
        #else:
        #    prob = 1 - pred[i]
        #    loss = -np.log(prob)
        #    losses.append(loss)
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
    return np.mean(losses), w, map_improvement_1, map_improvement_final

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

def run(args):
    assert dataset in ['train', 'dev', 'test']
    test_adulterants_only = args.test_adulterants_only
    add_adulterants = args.add_adulterants
    model_id = args.model_id
    dataset = args.dataset
    seed = args.seed
    use_askubuntu = args.use_askubuntu
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
    
    binary = True
    print_predictions = True
    iterations = 1
    l2_reg = 0.01#2
    maxiter = 5#100
    step_size = 1e-3#1e-5
    add_negatives = True
    batch_k = 10
    run_limit = None
    max_positives = 20
    max_negatives_ratio = 2#1
    all_min_losses = [] # loss using true distrubution
    all_uniform_losses = [] # loss using uniform distribution
    all_baseline_losses = [] # loss using static batch prediction
    #all_naive_online_losses = [] # loss that just increases the prob of that product with each update
    all_batchall_losses = [] # loss using bayesian update (batch=all)
    #all_batchk_losses = [] # loss using bayesian update (batch=k)
    #all_online_losses = [] # loss using online learning (batch=1)
    map_improvements_1 = []
    map_improvements_final = []
    start_time = time.time()
    for i, ind in enumerate(indices):
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
        if binary:
            true_dist = convert_to_zero_one(counts)
        else:
            true_dist = counts.astype(float) / counts.sum()
        
        print '=========================================='
        print i, ind, ing
        for iteration in range(iterations):
            print "Iteration:", iteration+1
            sequence = gen_sequence(counts, add_negatives, max_positives, max_negatives_ratio)
            if iteration == 0:
                print len(sequence), (sequence>=0).sum(), (counts>0).sum()
                min_loss = get_baseline_loss(true_dist, sequence) # should be 0 for binary
                if binary:
                    uniform_loss = get_baseline_loss(np.ones(num_categories)/2., sequence)
                else:
                    uniform_loss = get_baseline_loss(np.ones(num_categories, dtype=float)/num_categories, sequence)
                baseline_loss = get_baseline_loss(pred, sequence)
            #naive_online_loss, naive_pred = run_naive_online(rep, reps_prod, binary, sequence, naive_mutiplier)
            batchall_loss, w_batchall, map_improvement_1, map_improvement_final = run_online(ind, rep, reps_prod, binary, sequence, len(sequence), l2_reg, maxiter, step_size, ing_cat_pair_map)
            #batchk_loss, w_batchk = run_online(rep, reps_prod, binary, sequence, batch_k, l2_reg, step_size)
            #online_loss, w_online = run_online(rep, reps_prod, binary, sequence, 1, l2_reg, step_size)
            map_improvements_1.append(map_improvement_1)
            map_improvements_final.append(map_improvement_final)

            all_min_losses.append(min_loss)
            all_uniform_losses.append(uniform_loss)
            all_baseline_losses.append(baseline_loss)
            #all_naive_online_losses.append(naive_online_loss)
            all_batchall_losses.append(batchall_loss)
            #all_batchk_losses.append(batchk_loss)
            #all_online_losses.append(online_loss)

        if print_predictions and not use_askubuntu:
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
        
        print "Iter time elapsed: {:.1f}m".format((time.time()-iter_start_time)/60.)
        print "-------------------------------------------------------"
        print "Mean True Loss      :", np.mean(all_min_losses)
        print "Mean Uniform Loss   :", np.mean(all_uniform_losses)
        print "Mean Baseline Loss  :", np.mean(all_baseline_losses)
        #print "Mean Naive Algo Loss:", np.mean(all_naive_online_losses)
        print "Mean Batch all Loss :", np.mean(all_batchall_losses)
        #print "Mean Batch k Loss   :", np.mean(all_batchk_losses)
        #print "Mean Online Loss    :", np.mean(all_online_losses)
        print "Mean map_improvements_1     :", np.mean(map_improvements_1)
        print "Mean map_improvements_final :", np.mean(map_improvements_final)
        print "% map_improvements_1 > 0    :", (np.array(map_improvements_1) > 0).mean(), (np.array(map_improvements_1) == 0).mean(), (np.array(map_improvements_1) < 0).mean()
        print "% map_improvements_final > 0:", (np.array(map_improvements_final) > 0).mean(), (np.array(map_improvements_final) == 0).mean(), (np.array(map_improvements_final) < 0).mean()
    print "Time elapsed: {:.1f}m".format((time.time()-start_time)/60)

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
    args = argparser.parse_args()
    main(args)

