#!/usr/bin/env python
"""
Yelp용 DataLoader: friendship 그래프를 @mention 대신 사용.
기존 data.py의 DataLoader를 상속하여 get_graph()만 오버라이드한다.
"""

import os
import sys
import csv
import networkx as nx
import pandas as pd
import logging

# 부모 디렉토리의 data.py를 import하기 위해 path 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data import DataLoader

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)


class YelpDataLoader(DataLoader):
    """Yelp용 DataLoader. friendship 그래프를 @mention 대신 사용."""

    def load_data(self):
        """
        기존 load_data()를 오버라이드.
        Yelp 리뷰가 트윗보다 훨씬 길어서 C 파서 버퍼 오버플로 발생.
        engine='python'으로 변경하여 해결.
        """
        logging.info('loading Yelp dataset from %s' % self.data_home)
        train_file = os.path.join(self.data_home, 'user_info.train.gz')
        dev_file = os.path.join(self.data_home, 'user_info.dev.gz')
        test_file = os.path.join(self.data_home, 'user_info.test.gz')

        df_train = pd.read_csv(train_file, delimiter='\t', encoding=self.encoding,
                                names=['user', 'lat', 'lon', 'text'],
                                quoting=csv.QUOTE_NONE, engine='python',
                                on_bad_lines='skip')
        df_dev = pd.read_csv(dev_file, delimiter='\t', encoding=self.encoding,
                              names=['user', 'lat', 'lon', 'text'],
                              quoting=csv.QUOTE_NONE, engine='python',
                              on_bad_lines='skip')
        df_test = pd.read_csv(test_file, delimiter='\t', encoding=self.encoding,
                               names=['user', 'lat', 'lon', 'text'],
                               quoting=csv.QUOTE_NONE, engine='python',
                               on_bad_lines='skip')

        df_train.dropna(inplace=True)
        df_dev.dropna(inplace=True)
        df_test.dropna(inplace=True)

        df_train['user'] = df_train['user'].apply(lambda x: str(x).lower())
        df_train.drop_duplicates(['user'], inplace=True, keep='last')
        df_train.set_index(['user'], drop=True, append=False, inplace=True)
        df_train.sort_index(inplace=True)

        df_dev['user'] = df_dev['user'].apply(lambda x: str(x).lower())
        df_dev.drop_duplicates(['user'], inplace=True, keep='last')
        df_dev.set_index(['user'], drop=True, append=False, inplace=True)
        df_dev.sort_index(inplace=True)

        df_test['user'] = df_test['user'].apply(lambda x: str(x).lower())
        df_test.drop_duplicates(['user'], inplace=True, keep='last')
        df_test.set_index(['user'], drop=True, append=False, inplace=True)
        df_test.sort_index(inplace=True)

        self.df_train = df_train
        self.df_dev = df_dev
        self.df_test = df_test
        logging.info('Yelp data loaded: %d train, %d dev, %d test users',
                     len(df_train), len(df_dev), len(df_test))

    def get_graph(self):
        """friendship_edges.tsv에서 그래프 로드 (기존 @mention 파싱 대체)"""
        g = nx.Graph()
        nodes_list = (self.df_train.index.tolist() +
                      self.df_dev.index.tolist() +
                      self.df_test.index.tolist())
        node_id = {node: id for id, node in enumerate(nodes_list)}

        # 모든 노드 추가
        g.add_nodes_from(node_id.values())

        # self-loop 추가
        for node in node_id:
            g.add_edge(node_id[node], node_id[node])

        # friendship edges 로드
        edges_file = os.path.join(self.data_home, 'friendship_edges.tsv')
        edge_count = 0
        missing_count = 0
        with open(edges_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    u, v = parts
                    if u in node_id and v in node_id:
                        g.add_edge(node_id[u], node_id[v])
                        edge_count += 1
                    else:
                        missing_count += 1

        logging.info('Yelp friendship graph: %d nodes, %d edges (%d from file, %d skipped)',
                     g.number_of_nodes(), g.number_of_edges(), edge_count, missing_count)
        self.graph = g
