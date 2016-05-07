import os, sys, random, argparse, time, math, gzip
import cPickle as pickle
from collections import Counter

import numpy as np
import scipy
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
import theano
import theano.tensor as T

sys.path.append('../')
sys.path.append('../../../adulteration/wikipedia')
sys.path.append('../../../adulteration/model')
from nn import get_activation_by_name, create_optimization_updates, softmax
from nn import Layer, EmbeddingLayer, LSTM, RCNN, StrCNN, Dropout, apply_dropout
from utils import say, load_embedding_iterator

import hier_to_cat
import scoring
from wikipedia import *

np.set_printoptions(precision=3)
wiki_path = '../../../adulteration/wikipedia/'

def reduce_dim(train_hier_x, n_components, saved=False):
    if saved:
        with open('pca.pkl', 'r') as f:
            pca = pickle.loads(f)
        train_hier_x_new = pca.transform(train_hier_x)
    else:
        pca = PCA(n_components=n_components)
        train_hier_x_new = pca.fit_transform(train_hier_x)
         with open('pca.pkl', 'w') as f:
            pickle.dump(pca, f)
    return train_hier_x_new

def create_product_mask(products_len, n_hidden):
    mask = []
    max_len = products_len.max()
    for l in products_len:
        row = np.pad(np.ones(l), pad_width=(0,max_len-l), mode='constant', constant_values=0)
        mask.append(np.tile(row, (n_hidden,1)).T)
    mask = np.array(mask)
    mask = np.swapaxes(mask, 0,1)
    return mask.astype('int32')

def read_corpus_products():
    with open('../../../adulteration/ncim/idx_to_cat.pkl', 'rb') as f_in:
        idx_to_cat = pickle.load(f_in)
    products = [idx_to_cat[i] for i in sorted(idx_to_cat.keys())]
    tokens = input_to_tokens(ings=products)
    # Add padding
    products_len = np.array([len(i) for i in tokens])
    max_len = products_len.max()
    new_tokens = []
    for t in tokens:
        num_pads = max_len - len(t)
        new_tokens.append(t+num_pads*['<pad>'])
    return new_tokens, products_len

def read_corpus_adulterants():
    with open(wiki_path+'input_to_outputs_adulterants.pkl', 'r') as f_in:
        input_to_outputs = pickle.load(f_in)
    corpus_x, corpus_y, hier_x = [], [], []
    input_keys = sorted(input_to_outputs.keys())
    adulterants = get_adulterants()
    input_tokens = input_to_tokens(input_keys, adulterants)
    ing_idx_to_hier_map = hier_to_cat.gen_ing_idx_to_hier_map(adulterants, adulterants=True)
    assert len(input_keys) == len(input_tokens)
    for i in range(len(input_keys)):
        inp = input_keys[i]
        tokens = input_tokens[i]
        hier = ing_idx_to_hier_map.get(i, np.zeros(3751))
        out = input_to_outputs[inp]
        if out.sum() <= 0:
            continue
        if len(tokens) > 5:
            corpus_x.append(tokens)
        else:
            corpus_x.append([])
        hier_x.append(hier)
        normalized = out*1. / out.sum()
        assert np.isclose(normalized.sum(), 1, atol=1e-5)
        corpus_y.append(normalized)
        #len_corpus_y.append(out.sum())
    assert len(corpus_x)==len(corpus_y)==len(hier_x)
    return np.array(corpus_x), np.array(corpus_y).astype('float32'), np.array(hier_x).astype('float32')

def read_corpus_ingredients(num_ingredients=5000):
    with open(wiki_path+'input_to_outputs.pkl', 'r') as f_in:
        input_to_outputs = pickle.load(f_in)
    corpus_x, corpus_y, hier_x = [], [], []
    #y_indptr = [0]
    #y_indices = []
    #y_data = []
    input_keys = sorted(input_to_outputs.keys())
    ings = get_ings(num_ingredients)
    input_tokens = input_to_tokens(input_keys, ings)
    ing_idx_to_hier_map = hier_to_cat.gen_ing_idx_to_hier_map(ings)
    assert len(input_keys) == len(input_tokens) == num_ingredients
    for i in range(num_ingredients):
        inp = input_keys[i]
        tokens = input_tokens[i]
        hier = ing_idx_to_hier_map.get(i, np.zeros(3751))
        assert len(hier) == 3751
        out = input_to_outputs[inp]
        if out.sum() <= 0:
            continue
        #y_data.extend(out)
        #y_indices.extend(range(len(out)))
        #y_indptr.append(len(out))
        if len(tokens) > 5:
            corpus_x.append(tokens)
        else:
            corpus_x.append([])
        hier_x.append(hier)
        normalized = out*1. / out.sum()
        assert np.isclose(normalized.sum(), 1, atol=1e-5)
        corpus_y.append(normalized)
        #len_corpus_y.append(out.sum())
    assert len(corpus_x)==len(corpus_y)==len(hier_x)
    #corpus_y = scipy.sparse.csr_matrix((y_data, y_indices, np.cumsum(y_indptr)))
    return np.array(corpus_x), np.array(corpus_y).astype('float32'), np.array(hier_x).astype('float32')

def read_corpus(path):
    with open(path) as fin:
        lines = fin.readlines()
    lines = [ x.strip().split() for x in lines ]
    lines = [ x for x in lines if x ]
    corpus_x = [ x[1:] for x in lines ]
    corpus_y = [ int(x[0]) for x in lines ]
    return corpus_x, corpus_y


def create_one_batch(ids, x, y, hier):
    batch_x = np.column_stack( [ x[i] for i in ids ] )
    batch_y = np.array( [ y[i] for i in ids ] )
    if hier is None:
        batch_hier = np.column_stack( [[] for i in ids] ).astype('float32')
    else:
        batch_hier = np.column_stack( [ hier[i] for i in ids ] )
    #batch_y = y[ids]
    assert batch_x.shape[1] == batch_y.shape[0]
    return batch_x, batch_y, batch_hier

# shuffle training examples and create mini-batches
def create_batches(perm, x, y, hier, batch_size):

    # sort sequences based on their length
    # permutation is necessary if we want different batches every epoch
    first_nonzero_idx = sum([1 for i in x if len(i)==0])
    lst = sorted(perm, key=lambda i: len(x[i]))[first_nonzero_idx:]
    batches_x = [ ]
    batches_y = [ ]
    batches_hier = [ ]
    size = batch_size
    ids = [ lst[0] ]
    for i in lst[1:]:
        if len(ids) < size and len(x[i]) == len(x[ids[0]]):
            ids.append(i)
        else:
            #print ids
            #print x, len(x)
            #print y, len(y)
            bx, by, bhier = create_one_batch(ids, x, y, hier)
            batches_x.append(bx)
            batches_y.append(by)
            batches_hier.append(bhier)
            ids = [ i ]
    bx, by, bhier = create_one_batch(ids, x, y, hier)
    batches_x.append(bx)
    batches_y.append(by)
    batches_hier.append(bhier)

    # shuffle batches
    batch_perm = range(len(batches_x))
    random.shuffle(batch_perm)
    batches_x = [ batches_x[i] for i in batch_perm ]
    batches_y = [ batches_y[i] for i in batch_perm ]
    batches_hier = [ batches_hier[i] for i in batch_perm ]
    assert len(batches_x) == len(batches_y) == len(batches_hier)
    return batches_x, batches_y, batches_hier

def save_predictions(predict_model, train, dev, test, hier, products):
    trainx, trainy = train
    devx, devy = dev
    testx, testy = test
    hier_train, hier_dev, hier_test = hier
    for x_data, data_name, hier_x in [(
        trainx, 'train', hier_train), (devx, 'dev', hier_dev), (testx, 'test', hier_test)]:
        results = []
        for x_idx, x_for_predict in enumerate(x_data):
            if len(x_for_predict) > 0:
                if hier_x is not None:
                    if products is None:
                        p_y_given_x = predict_model(np.vstack(x_for_predict), 
                            np.vstack(hier_x[x_idx]))[0]
                    else:
                        p_y_given_x = predict_model(np.vstack(x_for_predict), 
                            np.vstack(hier_x[x_idx]), products)[0]
            else:
                results.append(np.zeros(len(results[0])))
        np.save('{}_pred.npy'.format(data_name), np.array(results))


def evaluate(x_data, y_data, hier_x, products, predict_model):
    """Compute the MAP of the data."""
    ing_cat_pair_map = {}
    for x_idx, x in enumerate(x_data):
        for y_idx, out in enumerate(y_data[x_idx]):
            if out > 0:
                ing_cat_pair_map[(x_idx, y_idx)] = True

    valid_ing_indices, results = [], []
    for x_idx, x_for_predict in enumerate(x_data):
        if len(x_for_predict) > 0:
            if hier_x is not None:
                if products is None:
                    p_y_given_x = predict_model(np.vstack(x_for_predict), 
                        np.vstack(hier_x[x_idx]))[0]
                else:
                    p_y_given_x = predict_model(np.vstack(x_for_predict), 
                        np.vstack(hier_x[x_idx]), products)[0]
            else:
                if products is None:
                    p_y_given_x = predict_model(np.vstack(x_for_predict), 
                        np.column_stack([[]]))[0]
                else:
                    p_y_given_x = predict_model(np.vstack(x_for_predict), 
                        np.column_stack([[]]), products)[0]
            valid_ing_indices.append(x_idx)
            results.append(p_y_given_x)
    valid_ing_indices = np.array(valid_ing_indices)
    avg_true_results = scoring.gen_avg_true_results(valid_ing_indices)

    results = np.array(results)
    print "Random:"
    scoring.evaluate_map(valid_ing_indices, results, ing_cat_pair_map, random=True)
    print "Avg True Results:"
    scoring.evaluate_map(valid_ing_indices, avg_true_results, ing_cat_pair_map, random=False)
    print "Model:"
    scoring.evaluate_map(valid_ing_indices, results, ing_cat_pair_map, random=False)

class Model:
    def __init__(self, args, embedding_layer, nclasses, products_len):
        self.args = args
        self.embedding_layer = embedding_layer
        self.nclasses = nclasses
        self.products_len = products_len

    def ready(self):
        args = self.args
        embedding_layer = self.embedding_layer
        self.n_hidden = args.hidden_dim
        self.n_in = embedding_layer.n_d
        dropout = self.dropout = theano.shared(
                np.float64(args.dropout_rate).astype(theano.config.floatX)
            )

        # x is length * batch_size
        # y is batch_size * num_cats
        self.x = T.imatrix('x')
        #self.y = T.ivector('y')
        self.y = T.fmatrix('y')
        self.y_len = T.ivector()
        x = self.x
        y = self.y
        y_len = self.y_len
        n_hidden = self.n_hidden
        n_in = self.n_in

        # hier is batch_size * hier_dim
        self.hier = T.fmatrix('hier')
        hier = self.hier
        if args.use_hier:
            size = 3751 # HIER CODE
        else:
            size = 0
        size_prod = 0

        # fetch word embeddings
        # (len * batch_size) * n_in
        slices  = embedding_layer.forward(x.ravel())
        self.slices = slices

        # 3-d tensor, len * batch_size * n_in
        slices = slices.reshape( (x.shape[0], x.shape[1], n_in) )

        # stacking the feature extraction layers
        pooling = args.pooling
        depth = args.depth
        layers = self.layers = [ ]
        prev_output = slices
        prev_output = apply_dropout(prev_output, dropout, v2=True)

        if args.products:
            self.products = T.imatrix('products')
            products = self.products
            slices_prod = embedding_layer.forward(products.ravel())
            slices_prod = slices_prod.reshape( (products.shape[0], products.shape[1], n_in) )
            prev_output_prod = apply_dropout(slices_prod, dropout, v2=True)
            products_len = theano.shared(self.products_len.astype(theano.config.floatX))
            products_len_mask = create_product_mask(self.products_len, n_hidden)
            products_len_mask = theano.shared(products_len_mask.astype(theano.config.floatX))

        softmax_inputs = [ ]
        softmax_inputs_prod = [ ]
        activation = get_activation_by_name(args.act)
        for i in range(depth):
            if args.layer.lower() == "lstm":
                print "Layer: LSTM"
                layer = LSTM(
                            n_in = n_hidden if i > 0 else n_in,
                            n_out = n_hidden
                        )
            elif args.layer.lower() == "strcnn":
                print "Layer: StrCNN"
                layer = StrCNN(
                            n_in = n_hidden if i > 0 else n_in,
                            n_out = n_hidden,
                            activation = activation,
                            decay = args.decay,
                            order = args.order
                        )
            elif args.layer.lower() == "rcnn":
                print "Layer: RCNN"
                layer = RCNN(
                            n_in = n_hidden if i > 0 else n_in,
                            n_out = n_hidden,
                            activation = activation,
                            order = args.order,
                            mode = args.mode
                        )
            else:
                raise Exception("unknown layer type: {}".format(args.layer))

            layers.append(layer)
            prev_output = layer.forward_all(prev_output)
            if pooling:
                softmax_inputs.append(T.sum(prev_output, axis=0)) # summing over columns
            else:
                softmax_inputs.append(prev_output[-1])
            prev_output = apply_dropout(prev_output, dropout)
            size += n_hidden

            if args.products:
                prev_output_prod = layer.forward_all(prev_output_prod)
                if pooling:
                    inter_result = prev_output_prod * products_len_mask
                    inter_result = T.sum(inter_result, axis=0)
                    inter_result = inter_result / products_len[:,None]
                    softmax_inputs_prod.append(inter_result) # summing over columns
                else:
                    inter_result = prev_output_prod[-1]
                    #inter_result = prev_output_prod[products_len.astype('int32'),np.arange(131),:]
                    softmax_inputs_prod.append(inter_result)
                prev_output_prod = apply_dropout(prev_output_prod, dropout)
                size_prod += n_hidden

        softmax_inputs.append(hier.T)

        # final feature representation is the concatenation of all extraction layers
        if pooling:
            softmax_input = T.concatenate(softmax_inputs, axis=1) / x.shape[0]
        else:
            softmax_input = T.concatenate(softmax_inputs, axis=1)
        softmax_input = apply_dropout(softmax_input, dropout, v2=True)

        if args.products:
            if pooling:
                softmax_inputs_prod = T.concatenate(softmax_inputs_prod, axis=1)# / products.shape[0]
            else:
                softmax_inputs_prod = T.concatenate(softmax_inputs_prod, axis=1)
            softmax_inputs_prod = apply_dropout(softmax_inputs_prod, dropout, v2=True)

        #if not args.products:
            # feed the feature repr. to the softmax output layer
        if args.products:
            softmax_n_in = 131 #size+size_prod
        else:
            softmax_n_in = size
            layers.append( Layer(
                    n_in = softmax_n_in,
                    n_out = self.nclasses,
                    activation = softmax,
                    has_bias = False,
            ) )

        for l,i in zip(layers, range(len(layers))):
            say("layer {}: n_in={}\tn_out={}\n".format(
                i, l.n_in, l.n_out
            ))

        self.softmax_input = softmax_input

        # unnormalized score of y given x
        if args.products:
            softmax_input = T.dot(softmax_input, softmax_inputs_prod.T)
            self.p_y_given_x = softmax(softmax_input)
        else:
            self.p_y_given_x = layers[-1].forward(softmax_input)
        self.pred = T.argmax(self.p_y_given_x, axis=1)

        self.nll_loss = T.mean( T.nnet.categorical_crossentropy(
                                    self.p_y_given_x,
                                    y
                            ))

        # adding regularizations
        self.l2_sqr = None
        self.params = [ ]
        for layer in layers:
            self.params += layer.params
        for p in self.params:
            if self.l2_sqr is None:
                self.l2_sqr = args.l2_reg * T.sum(p**2)
            else:
                self.l2_sqr += args.l2_reg * T.sum(p**2)

        nparams = sum(len(x.get_value(borrow=True).ravel()) \
                        for x in self.params)
        say("total # parameters: {}\n".format(nparams))

    def save_model(self, path, args):
         # append file suffix
        if not path.endswith(".pkl.gz"):
            if path.endswith(".pkl"):
                path += ".gz"
            else:
                path += ".pkl.gz"

        with gzip.open(path, "wb") as fout:
            pickle.dump(
                ([ x.get_value() for x in self.params ], args),
                fout,
                protocol = pickle.HIGHEST_PROTOCOL
            )

    def load_model(self, path):
        if not os.path.exists(path):
            if path.endswith(".pkl"):
                path += ".gz"
            else:
                path += ".pkl.gz"

        with gzip.open(path, "rb") as fin:
            param_values, args = pickle.load(fin)

        assert self.args.layer.lower() == args.layer.lower()
        assert self.args.depth == args.depth
        assert self.args.hidden_dim == args.hidden_dim
        for x,v in zip(self.params, param_values):
            x.set_value(v)

    def eval_accuracy(self, preds, golds):
        fine = sum([ sum(p == y) for p,y in zip(preds, golds) ]) + 0.0
        fine_tot = sum( [ len(y) for y in golds ] )
        return fine/fine_tot


    def train(self, train, dev, test, hier, products):
        args = self.args
        trainx, trainy = train
        if hier is None or not args.use_hier:
            hier = (None, None, None)
        train_hier_x, dev_hier_x, test_hier_x = hier
        batch_size = args.batch

        #if products is None:
        #    products = [[] for i in range(131)]
        if products is not None:
            products = np.column_stack(products)
        blank_product_hier = np.column_stack( [[] for i in range(131)] )

        if dev:
            dev_batches_x, dev_batches_y, dev_batches_hier = create_batches(
                    range(len(dev[0])),
                    dev[0],
                    dev[1],
                    dev_hier_x,
                    batch_size
            )

        if test:
            test_batches_x, test_batches_y, test_batches_hier = create_batches(
                    range(len(test[0])),
                    test[0],
                    test[1],
                    test_hier_x,
                    batch_size
            )

        cost = self.nll_loss + self.l2_sqr

        updates, lr, gnorm = create_optimization_updates(
                cost = cost,
                params = self.params,
                lr = args.learning_rate,
                method = args.learning
            )[:3]
        if products is not None:
            inputs = [self.x, self.y, self.hier, self.products]
            predict_inputs = [self.x, self.hier, self.products]
        else:
            inputs = [self.x, self.y, self.hier]
            predict_inputs = [self.x, self.hier]
        train_model = theano.function(
             inputs = inputs,
             outputs = [ cost, gnorm ],
             updates = updates,
             allow_input_downcast = True
        )
        predict_model = theano.function(
             inputs = predict_inputs,
             outputs = self.p_y_given_x,
             allow_input_downcast = True
        )
        """
        get_representation = theano.function(
             inputs = [self.x, self.hier, self.products],
             outputs = self.softmax_input,
             allow_input_downcast = True
        )
        """
        eval_acc = theano.function(
             inputs = predict_inputs,
             outputs = self.pred,
             allow_input_downcast = True
        )
        unchanged = 0
        best_dev = 0.0
        dropout_prob = np.float64(args.dropout_rate).astype(theano.config.floatX)

        start_time = time.time()
        eval_period = args.eval_period

        perm = range(len(trainx))

        say(str([ "%.2f" % np.linalg.norm(x.get_value(borrow=True)) for x in self.params ])+"\n")
        for epoch in xrange(args.max_epochs):
            unchanged += 1
            #if dev and unchanged > 30: return
            train_loss = 0.0

            random.shuffle(perm)
            batches_x, batches_y, batches_hier = create_batches(
                perm, trainx, trainy, train_hier_x, batch_size)
            N = len(batches_x)
            for i in xrange(N):

                if i % 100 == 0:
                    sys.stdout.write("\r%d" % i)
                    sys.stdout.flush()

                x = batches_x[i]
                y = batches_y[i]
                hier_x = batches_hier[i]
                y_len = np.array([j.sum() for j in batches_y[i]])
                #y = y.toarray()

                assert x.dtype in ['float32', 'int32']
                assert y.dtype in ['float32', 'int32']
                assert hier_x.dtype in ['float32', 'int32']
                #print x.shape
                #print y.shape
                #print hier_x.shape
                if products is not None:
                    #print products.shape
                    assert products.dtype in ['float32', 'int32']
                    va, grad_norm = train_model(x, y, hier_x, products)
                else:
                    va, grad_norm = train_model(x, y, hier_x)
                train_loss += va
                
                #if products is not None:
                #    print x.shape, hier_x.shape, np.array(products[0:1]).T.shape, hier_x[:,0:1].shape
                
                #print i, N
                #if products is not None:
                #    prod_rep = get_representation(products, blank_product_hier)
                #    print prod_rep

                # debug
                if math.isnan(va):
                    print ""
                    print i-1, i
                    print x
                    print y
                    return

                if (i == N-1) or (eval_period > 0 and (i+1) % eval_period == 0):
                    self.dropout.set_value(0.0)

                    say( "\n" )
                    say( "Epoch %.1f\tloss=%.4f\t|g|=%s  [%.2fm]\n" % (
                            epoch + (i+1)/(N+0.0),
                            train_loss / (i+1),
                            float(grad_norm),
                            (time.time()-start_time) / 60.0
                    ))
                    say(str([ "%.2f" % np.linalg.norm(x.get_value(borrow=True)) for x in self.params ])+"\n")

                    """
                    if dev:
                        preds = [ eval_acc(x) for x in dev_batches_x ]
                        nowf_dev = self.eval_accuracy(preds, dev_batches_y)
                        if nowf_dev > best_dev:
                            unchanged = 0
                            best_dev = nowf_dev
                            if args.model:
                                self.save_model(args.model, args)

                        say("\tdev accuracy=%.4f\tbest=%.4f\n" % (
                                nowf_dev,
                                best_dev
                        ))
                        if args.test and nowf_dev == best_dev:
                            preds = [ eval_acc(x) for x in test_batches_x ]
                            nowf_test = self.eval_accuracy(preds, test_batches_y)
                            say("\ttest accuracy=%.4f\n" % (
                                    nowf_test,
                            ))

                        if best_dev > nowf_dev + 0.05:
                            return
                    """

                    self.dropout.set_value(dropout_prob)

                    start_time = time.time()

            #print "Length of trainx: ", len(trainx)
            #for x_idx, x_for_predict in enumerate(trainx[3233:3236]):
            #    if len(x_for_predict) > 0:
            #        p_y_given_x = predict_model(np.vstack(x_for_predict))
            #        print x_idx, p_y_given_x
            if epoch % 10 == 0 or epoch == args.max_epochs-1:
                evaluate_start_time = time.time()
                print "======= Training evaluation ========"
                evaluate(trainx, trainy, train_hier_x, products, predict_model)
                if dev:
                    print "======= Validation evaluation ========"
                    evaluate(dev[0], dev[1], dev_hier_x, products, predict_model)
                if test:
                    print "======= Adulteration evaluation ========"
                    evaluate(test[0], test[1], test_hier_x, products, predict_model)
                print "Evaluate time: {:.1f}m".format((time.time()-evaluate_start_time)/60)
                start_time = time.time()

        save_predictions(predict_model, train, dev, test, hier)


def main(args):
    print args

    model = None

    assert args.embedding, "Pre-trained word embeddings required."

    if '.pkl' in args.embedding:
        with open(args.embedding, 'rb') as f:
            embedding = pickle.load(f)
            if '<unk>' not in embedding:
                embedding['<unk>'] = np.zeros(len(embedding['</s>']))
    else:
        embedding = load_embedding_iterator(args.embedding)

    embedding_layer = EmbeddingLayer(
                n_d = args.hidden_dim,
                vocab = [ "<unk>" ],
                embs = embedding       
            )

    products, products_len = None, None
    if args.products:
        products_text, products_len = read_corpus_products()
        products = [ embedding_layer.map_to_ids(x) for x in products_text ]

    train_hier_x = dev_hier_x = test_hier_x = None
    if args.train:
        train_x_text, train_y, train_hier_x = read_corpus_ingredients()
        #train_hier_x = reduce_dim(train_hier_x, args.hidden_dim)
        num_data = len(train_x_text)
        print "Num data points:", num_data
        if args.dev:
            train_indices, dev_indices = train_test_split(
                range(num_data), test_size=1/3., random_state=42)
            dev_x_text = train_x_text[dev_indices]
            train_x_text = train_x_text[train_indices]
            dev_y = train_y[dev_indices]
            train_y = train_y[train_indices]
            if len(train_hier_x) > 0:
                dev_hier_x = train_hier_x[dev_indices]
                train_hier_x = train_hier_x[train_indices]
        train_x = [ embedding_layer.map_to_ids(x) for x in train_x_text ]

    if args.dev:
        #dev_x, dev_y = read_corpus(args.dev)
        dev_x = [ embedding_layer.map_to_ids(x) for x in dev_x_text ]

    if args.test:
        test_x_text, test_y, test_hier_x = read_corpus_adulterants()
        test_x = [ embedding_layer.map_to_ids(x) for x in test_x_text ]

    if args.train:
        model = Model(
                    args = args,
                    embedding_layer = embedding_layer,
                    nclasses = len(train_y[0]), #max(train_y.data)+1
                    products_len = products_len,
            )
        model.ready()
        #print train_x[0].dtype, train_hier_x[0].dtype, dev_hier_x[0].dtype, test_hier_x[0].dtype
        model.train(
                (train_x, train_y),
                (dev_x, dev_y) if args.dev else None,
                (test_x, test_y) if args.test else None,
                (train_hier_x, dev_hier_x, test_hier_x) if args.use_hier else None,
                products
            )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--train",
            type = str,
            default = "",
            help = "path to training data"
        )
    argparser.add_argument("--dev",
            type = str,
            default = "",
            help = "path to development data"
        )
    argparser.add_argument("--test",
            type = str,
            default = "",
            help = "path to test data"
        )
    argparser.add_argument("--hidden_dim", "-d",
            type = int,
            default = 200,
            help = "hidden dimensions"
        )
    argparser.add_argument("--decay",
            type = float,
            default = 0.3
        )
    argparser.add_argument("--learning",
            type = str,
            default = "adagrad",
            help = "learning method (sgd, adagrad, ...)"
        )
    argparser.add_argument("--learning_rate",
            type = float,
            default = "0.01",
            help = "learning rate"
        )
    argparser.add_argument("--max_epochs",
            type = int,
            default = 100,
            help = "maximum # of epochs"
        )
    argparser.add_argument("--eval_period",
            type = int,
            default = -1,
            help = "evaluate on dev every period"
        )
    argparser.add_argument("--dropout_rate",
            type = float,
            default = 0.0,
            help = "dropout probability"
        )
    argparser.add_argument("--l2_reg",
            type = float,
            default = 0.00001
        )
    argparser.add_argument("--embedding",
            type = str,
            default = ""
        )
    argparser.add_argument("--batch",
            type = int,
            default = 15,
            help = "mini-batch size"
        )
    argparser.add_argument("--depth",
            type = int,
            default = 3,
            help = "number of feature extraction layers (min:1)"
        )
    argparser.add_argument("--order",
            type = int,
            default = 3,
            help = "when the order is k, we use up tp k-grams (k=1,2,3)"
        )
    argparser.add_argument("--act",
            type = str,
            default = "relu",
            help = "activation function (none, relu, tanh)"
        )
    argparser.add_argument("--layer",
            type = str,
            default = "rcnn"
        )
    argparser.add_argument("--mode",
            type = int,
            default = 1
        )
    argparser.add_argument("--seed",
            type = int,
            default = -1,
            help = "random seed of the model"
        )
    argparser.add_argument("--model",
            type = str,
            default = "",
            help = "save model to this file"
        )
    argparser.add_argument("--pooling",
            type = int,
            default = 1,
            help = "whether to use mean pooling or take the last vector"
        )
    argparser.add_argument("--use_hier",
            action='store_true',
            help = "use hierarchy"
        )
    argparser.add_argument("--products",
            action='store_true',
            help = "use product categories"
        )
    args = argparser.parse_args()
    main(args)

