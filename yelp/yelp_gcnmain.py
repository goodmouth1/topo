#!/usr/bin/env python
"""
Yelp용 GCN 훈련 메인 스크립트.
기존 gcnmain.py와 동일하지만 YelpDataLoader를 사용한다.

사용법:
  python yelp_gcnmain.py -d ./yelp_geotext_format -bucket 300 \
    -hid 300 300 300 -mindf 10 -reg 1e-6 -dropout 0.5 \
    -cel 10 -conv -highway -builddata
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import os
import sys

# ★ GPU 대신 CPU 사용 (355K 유저 → 8GB VRAM 초과)
os.environ['THEANO_FLAGS'] = 'device=cpu,floatX=float32'
import glob
import argparse
import pickle
import copy
import logging
import json
import numpy as np
from haversine import haversine
import gzip
import codecs
from collections import OrderedDict, defaultdict, Counter
import re
import networkx as nx
import scipy as sp
import random

# 부모 디렉토리를 path에 추가 (gcnmodel, data 등 import)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data import dump_obj, load_obj
from yelp_data import YelpDataLoader

# Monkey patch for Theano 1.0+ compatibility
try:
    import theano.tensor.signal.pool
    import theano.tensor.signal
    if not hasattr(theano.tensor.signal, 'downsample'):
        print("Patching theano.tensor.signal.downsample = theano.tensor.signal.pool")
        theano.tensor.signal.downsample = theano.tensor.signal.pool
except ImportError as e:
    print(f"Warning: Failed to patch Theano: {e}")

from gcnmodel import GraphConv

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logging.info('Yelp GCN training script initialized.')
np.random.seed(77)
model_args = None


def geo_eval(y_true, y_pred, U_eval, classLatMedian, classLonMedian, userLocation):
    assert len(y_pred) == len(U_eval), "#preds: %d, #users: %d" % (len(y_pred), len(U_eval))
    distances = []
    latlon_pred = []
    latlon_true = []
    for i in range(0, len(y_pred)):
        user = U_eval[i]
        location = userLocation[user].split(',')
        lat, lon = float(location[0]), float(location[1])
        latlon_true.append([lat, lon])
        prediction = str(y_pred[i])
        lat_pred, lon_pred = classLatMedian[prediction], classLonMedian[prediction]
        latlon_pred.append([lat_pred, lon_pred])
        distance = haversine((lat, lon), (lat_pred, lon_pred))
        distances.append(distance)

    acc_at_161 = 100 * len([d for d in distances if d < 161]) / float(len(distances))
    logging.info("Mean: " + str(int(np.mean(distances))) + " Median: " + str(int(np.median(distances))) + " Acc@161: " + str(int(acc_at_161)))

    return np.mean(distances), np.median(distances), acc_at_161, distances, latlon_true, latlon_pred


def preprocess_data(data_home, **kwargs):
    bucket_size = kwargs.get('bucket', 300)
    encoding = kwargs.get('encoding', 'utf-8')
    celebrity_threshold = kwargs.get('celebrity', 10)
    mindf = kwargs.get('mindf', 10)
    dtype = kwargs.get('dtype', 'float32')
    one_hot_label = kwargs.get('onehot', False)
    vocab_file = os.path.join(data_home, 'vocab.pkl')

    # Smart Data Loading
    dump_file = os.path.join(data_home, 'dump.pkl')
    found_files = glob.glob(os.path.join(data_home, '*.pkl.gz'))
    if not found_files:
        all_pkl = glob.glob(os.path.join(data_home, '*.pkl'))
        found_files = [f for f in all_pkl if 'vocab.pkl' not in os.path.basename(f) and 'model-' not in os.path.basename(f)]
    if found_files:
        dump_file = found_files[0]

    if os.path.exists(dump_file) and not model_args.builddata:
        logging.info(f'Loading cached data from {dump_file}...')
        if dump_file.endswith('.gz'):
            with gzip.open(dump_file, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
        else:
            data = load_obj(dump_file)
        logging.info('Data loaded from cache.')
        return data

    # ★ 핵심 변경: DataLoader -> YelpDataLoader
    dl = YelpDataLoader(data_home=data_home, bucket_size=bucket_size, encoding=encoding,
                        celebrity_threshold=celebrity_threshold, one_hot_labels=one_hot_label,
                        mindf=mindf, token_pattern=r'(?u)(?<![@])#?\b\w\w+\b')

    logging.info('Loading Yelp data...')
    dl.load_data()
    logging.info('Assigning classes...')
    dl.assignClasses()
    logging.info('Computing TF-IDF...')
    dl.tfidf()
    vocab = dl.vectorizer.vocabulary_
    logging.info('Saving vocab in {}'.format(vocab_file))
    dump_obj(vocab, vocab_file)

    U_test = dl.df_test.index.tolist()
    U_dev = dl.df_dev.index.tolist()
    U_train = dl.df_train.index.tolist()

    logging.info('Building friendship graph...')
    dl.get_graph()

    logging.info('Creating adjacency matrix...')
    adj = nx.adjacency_matrix(dl.graph, nodelist=range(len(U_train + U_dev + U_test)), weight='w')
    adj.setdiag(0)
    selfloop_value = 1
    adj.setdiag(selfloop_value)
    n, m = adj.shape
    diags = adj.sum(axis=1).flatten()
    with sp.errstate(divide='ignore'):
        diags_sqrt = 1.0 / np.sqrt(diags)
    diags_sqrt[np.isinf(diags_sqrt)] = 0
    D_pow_neghalf = sp.sparse.spdiags(diags_sqrt, [0], m, n, format='csr')
    A = D_pow_neghalf * adj * D_pow_neghalf
    A = A.astype(dtype)
    logging.info('Adjacency matrix created.')

    X_train = dl.X_train
    X_dev = dl.X_dev
    X_test = dl.X_test
    Y_test = dl.test_classes
    Y_train = dl.train_classes
    Y_dev = dl.dev_classes
    classLatMedian = {str(c): dl.cluster_median[c][0] for c in dl.cluster_median}
    classLonMedian = {str(c): dl.cluster_median[c][1] for c in dl.cluster_median}

    P_test = [str(a[0]) + ',' + str(a[1]) for a in dl.df_test[['lat', 'lon']].values.tolist()]
    P_train = [str(a[0]) + ',' + str(a[1]) for a in dl.df_train[['lat', 'lon']].values.tolist()]
    P_dev = [str(a[0]) + ',' + str(a[1]) for a in dl.df_dev[['lat', 'lon']].values.tolist()]
    userLocation = {}
    for i, u in enumerate(U_train):
        userLocation[u] = P_train[i]
    for i, u in enumerate(U_test):
        userLocation[u] = P_test[i]
    for i, u in enumerate(U_dev):
        userLocation[u] = P_dev[i]

    data = (A, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation)
    if not model_args.builddata:
        logging.info('Dumping data in {} ...'.format(str(dump_file)))
        dump_obj(data, dump_file)
        logging.info('Data dump finished!')

    return data


def main(data, args, **kwargs):
    batch_size = kwargs.get('batch', 500)
    hidden_size = kwargs.get('hidden', [100])
    dropout = kwargs.get('dropout', 0.0)
    regul = kwargs.get('regularization', 1e-6)
    dtype = 'float32'
    dtypeint = 'int32'
    check_percentiles = kwargs.get('percent', False)
    A, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation = data

    logging.info('Stacking features and creating indices...')
    X = sp.sparse.vstack([X_train, X_dev, X_test])
    if len(Y_train.shape) == 1:
        Y = np.hstack((Y_train, Y_dev, Y_test))
    else:
        Y = np.vstack((Y_train, Y_dev, Y_test))
    Y = Y.astype(dtypeint)
    X = X.astype(dtype)
    A = A.astype(dtype)

    input_size = X.shape[1]
    output_size = np.max(Y) + 1
    verbose = not args.silent
    fractions = args.lblfraction
    all_train_indices = np.asarray(range(0, X_train.shape[0])).astype(dtypeint)

    logging.info('Running GCN with graph convolution...')
    clf = GraphConv(input_size=input_size, output_size=output_size, hid_size_list=hidden_size,
                    regul_coef=regul, drop_out=dropout, batchnorm=args.batchnorm, highway=model_args.highway)
    clf.build_model(A, use_text=args.notxt, use_labels=args.lp, seed=model_args.seed)

    for percentile in fractions:
        logging.info('***********percentile %f ******************' % percentile)
        model_file = os.path.join(args.dir, 'model-{}-{}.pkl'.format(A.shape[0], percentile))

        selection_size = min(int(percentile * X.shape[0]), all_train_indices.shape[0])
        train_indices = np.random.choice(all_train_indices, size=selection_size, replace=False).astype(dtypeint)
        num_training_samples = train_indices.shape[0]
        logging.info('{} training samples'.format(num_training_samples))

        dev_indices = np.asarray(range(X_train.shape[0], X_train.shape[0] + X_dev.shape[0])).astype(dtypeint)
        test_indices = np.asarray(range(X_train.shape[0] + X_dev.shape[0], X_train.shape[0] + X_dev.shape[0] + X_test.shape[0])).astype(dtypeint)

        if args.load:
            clf.load(load_obj, model_file)
        else:
            if clf.fitted:
                clf.reset()
            clf.fit(X, A, Y, train_indices=train_indices, val_indices=dev_indices,
                    n_epochs=10000, batch_size=batch_size, max_down=args.maxdown, verbose=verbose, seed=model_args.seed)

            logging.info('dev results:')
            y_pred, _ = clf.predict(X, A, dev_indices)
            mean, median, acc, distances, latlon_true, latlon_pred = geo_eval(Y_dev, y_pred, U_dev, classLatMedian, classLonMedian, userLocation)

            # ★ GCN 예측 결과 저장 (Gravity Model에서 사용)
            pred_file = 'gcn_{}_percent_pred_{}.pkl'.format(percentile, output_size)
            with open(pred_file, 'wb') as fout:
                pickle.dump((distances, latlon_true, latlon_pred), fout)
            logging.info(f'GCN predictions saved to {pred_file}')

            logging.info('test results:')
            y_pred, _ = clf.predict(X, A, test_indices)
            mean_t, median_t, acc_t, dist_t, llt, llp = geo_eval(Y_test, y_pred, U_test, classLatMedian, classLonMedian, userLocation)

            # test 예측도 저장
            test_pred_file = 'gcn_{}_percent_pred_{}_test.pkl'.format(percentile, output_size)
            with open(test_pred_file, 'wb') as fout:
                pickle.dump((dist_t, llt, llp), fout)
            logging.info(f'Test predictions saved to {test_pred_file}')


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Yelp GCN Training')
    parser.add_argument('-i', '--dataset', metavar='str', type=str, default='yelp')
    parser.add_argument('-bucket', '--bucket', metavar='int', type=int, default=300)
    parser.add_argument('-batch', '--batch', metavar='int', type=int, default=500)
    parser.add_argument('-hid', nargs='+', type=int, default=[100])
    parser.add_argument('-mindf', '--mindf', metavar='int', type=int, default=10)
    parser.add_argument('-d', '--dir', metavar='str', type=str, default='./yelp_geotext_format')
    parser.add_argument('-enc', '--encoding', metavar='str', type=str, default='utf-8')
    parser.add_argument('-reg', '--regularization', metavar='float', type=float, default=1e-6)
    parser.add_argument('-cel', '--celebrity', metavar='int', type=int, default=10)
    parser.add_argument('-conv', '--convolution', action='store_true')
    parser.add_argument('-tune', '--tune', action='store_true')
    parser.add_argument('-tf', '--tensorflow', action='store_true')
    parser.add_argument('-batchnorm', action='store_true')
    parser.add_argument('-dropout', type=float, default=0)
    parser.add_argument('-percent', action='store_true')
    parser.add_argument('-vis', metavar='str', type=str, default=None)
    parser.add_argument('-builddata', action='store_true')
    parser.add_argument('-lp', action='store_true')
    parser.add_argument('-notxt', action='store_false')
    parser.add_argument('-maxdown', type=int, default=10)
    parser.add_argument('-silent', action='store_true')
    parser.add_argument('-highway', action='store_true')
    parser.add_argument('-seed', metavar='int', type=int, default=77)
    parser.add_argument('-save', action='store_true')
    parser.add_argument('-load', action='store_true')
    parser.add_argument('-feature_report', action='store_true')
    parser.add_argument('-lblfraction', nargs='+', type=float, default=[1.0])
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    model_args = args

    data = preprocess_data(data_home=args.dir, encoding=args.encoding,
                           celebrity=args.celebrity, bucket=args.bucket, mindf=args.mindf)
    main(data, args, batch=args.batch, hidden=args.hid,
         regularization=args.regularization, dropout=args.dropout, percent=args.percent)
