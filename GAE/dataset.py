
import json
import csv
import os
import re
import numpy as np

from pprint import pprint
from itertools import islice

import torch
import torch.nn.functional as F

import networkx as nx
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import erdos_renyi_graph, to_networkx, from_networkx
import matplotlib.pyplot as plt


class GraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        raw_file_names = []
        for root, dirs, names in os.walk(self.root + '/raw/'):
            patten = re.compile(r'process')
            for filename in names:
                match = patten.search(filename)
                if match:
                    raw_file_names.append(filename)
        return raw_file_names

    @property
    def processed_file_names(self):
        return ["processed.pt"]

    @property
    def event_num(self):
        with open(self.root + '/raw/id_url+type.csv', 'r') as idfile:
            reader = csv.reader(idfile)
            column = [row[0] for row in reader]
            event_num = len(column) - 1
            idfile.close()
        return event_num

    def get_embedding(self):
        vocab = {}
        with open(self.root + '/raw/embedding_id_url+type.csv', 'r', encoding='utf-8') as embedding_file:
            reader = csv.reader(embedding_file)
            for row in islice(reader, 1, None):
                event = int(row[0])
                embeddings = row[5].strip('[]').split()
                event_embedding = np.asarray(embeddings, dtype=np.float)
                vocab[event] = event_embedding
            embedding_file.close()
        return vocab

    def download(self):
        """
        do not need to download anything
        :return:
        """
        pass

    def process(self):
        data_list = []

        vocab = self.get_embedding()

        F12_01_num = 0
        F25_01_num = 0
        F25_02_num = 0
        F25_03_num = 0

        file_paths = self.raw_paths
        for fp in file_paths:
            with open(fp, "r") as f:
                try:
                    for line in f:
                        one_graph = json.loads(line)

                        if one_graph.get('error_trace_type') == 'F12-01':
                            if F12_01_num > 500:
                                continue
                            F12_01_num += 1

                        if one_graph.get('error_trace_type') == 'F25-01':
                            if F25_01_num > 500:
                                continue
                            F25_01_num += 1

                        if one_graph.get('error_trace_type') == 'F25-02':
                            if F25_02_num > 500:
                                continue
                            F25_02_num += 1

                        if one_graph.get('error_trace_type') == 'F25-03':
                            if F25_03_num > 500:
                                continue
                            F25_03_num += 1

                        # 节点信息
                        nodes = one_graph.get('node_info')
                        if len(nodes) <= 2:
                            continue
                        nodes_embedding = []
                        for node in nodes:
                            nodes_embedding.append(vocab[node[-1]])
                        nodes_embedding = torch.tensor(nodes_embedding, dtype=torch.float)
                        nodes = torch.tensor(nodes).long()
                        nodes_token = nodes[:, -1]

                        # 边连接信息
                        edge_pairs = one_graph.get('edge_index')
                        edge_list_1 = []
                        edge_list_2 = []
                        for pair in edge_pairs:
                            edge_list_1.append(pair[0])
                            edge_list_2.append(pair[1])
                        edge_index = torch.tensor([edge_list_1, edge_list_2], dtype=torch.long)

                        # 边属性
                        edge_attr_list = []
                        for ea in one_graph.get('edge_attr'):
                            edge_attr_list.append([ea])
                        edge_attr = torch.tensor(edge_attr_list).float() + 1

                        # 打的标记
                        y_data = one_graph.get('trace_bool')
                        y = torch.LongTensor([0 if y_data is True else 1])

                        # trace id
                        trace_id = one_graph.get('trace_id')

                        # error_trace_type
                        error_type = one_graph.get('error_trace_type')

                        data = Data(x=nodes_embedding,
                                    y=y,
                                    edge_index=edge_index,
                                    edge_attr=edge_attr,
                                    trace_id=trace_id,
                                    nodes_token=nodes_token,
                                    error_trace_type=error_type
                                    )
                        # if trace_id == "7afe7b197c404764859109e7361a0c76.47.16285756424630231" or \
                        #     trace_id == "7afe7b197c404764859109e7361a0c76.47.16285756501180283" or \
                        #     trace_id == "7afe7b197c404764859109e7361a0c76.47.16285756188750113":
                        #
                        #     g = to_networkx(data)
                        #     nx.draw(g)
                        #     plt.savefig(f"{trace_id}.png")
                        #     print(f"save image {trace_id}.png")
                        #
                        #     plt.clf()

                        pprint(data)
                        data_list.append(data)
                except Exception as e:
                    print(e)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    print("start...")
    dataset = GraphDataset(root="./data/")
    dataset.process()


