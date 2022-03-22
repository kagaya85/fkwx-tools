import time
import utils
import os
import random
import torch
import torch.nn as nn
import numpy as np
from itertools import groupby

from model import GAE, GAE_GCN
from tqdm import tqdm
from torch_geometric.datasets import Reddit, TUDataset
from torch_geometric.data import NeighborSampler, DataLoader
from torch_geometric.nn import DeepGraphInfomax, GatedGraphConv, GlobalAttention
from torch_geometric.data import DataLoader
from sklearn import metrics
from torch.utils.data import Subset
# from dataset import GraphDataset
from aug_dataset import TraceDataset
from sklearn.metrics import roc_auc_score, PrecisionRecallDisplay, precision_recall_curve, precision_score, recall_score
from sklearn.mixture import GaussianMixture
from math import sqrt
import matplotlib.pyplot as plt


def save_model(model: 'nn.Module', path):
    print(f"save model to {path}")
    torch.save(model.state_dict(), path)


def load_model_GAE_GCN(path) -> GAE_GCN:
    print(f"load GAE_GCN model form {path}")
    model = GAE_GCN(hidden_dim=hidden_dim, num_layers=num_layers).to(device)  # type: nn.Module
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def load_model_GAE(path) -> GAE:
    print(f"load GAE model form {path}")
    model = GAE(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


if __name__ == '__main__':
    epochs = 20
    batch_size = 32

    hidden_dim = 300
    num_layers = 3
    lr = 0.001
    lr_milestones = [15, 35]

    normal_classes = [0]
    abnormal_classes = [1]
    ratio = [6, 4]
    alpha = 0.7  # attr_loss

    load_model_flag = False
    save_model_flag = False

    outpath = f"save_files_{str(time.time()).split('.')[0]}/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    load_model_path = "alpha_3_7_save_1629040795.model"
    save_model_path = outpath + "alpha_999_001_save_.model"
    save_data_filename = outpath + "alpha_5_5_test." + str(time.time()).split(".")[0]

    # 设置 quick_test 为True来快速测试
    quick_test = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # config dataset
    dataset = TraceDataset(root='./data/', aug='none')

    normal_idx = utils.get_target_label_idx(dataset.data.y.clone().data.cpu().numpy(), normal_classes)
    abnormal_idx = utils.get_target_label_idx(dataset.data.y.clone().data.cpu().numpy(), abnormal_classes)

    random.shuffle(abnormal_idx)
    random.shuffle(normal_idx)

    train_normal = normal_idx[:int(len(normal_idx) * ratio[0] / sum(ratio))]
    test_normal = normal_idx[int(len(normal_idx) * ratio[0] / sum(ratio)):]
    train_abnormal = abnormal_idx[:int(len(abnormal_idx) * 0)]
    test_abnormal = abnormal_idx[int(len(abnormal_idx) * 0):]

    train_dataset = Subset(dataset, train_normal + train_abnormal)
    test_dataset = Subset(dataset, test_abnormal + test_normal)

    if quick_test:
        sampler = [i for i in range(20*batch_size)]
        train_dataset = Subset(dataset, sampler)
        test_dataset = Subset(dataset, sampler)


    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # model = GAE(hidden_dim=hidden_dim, num_layers=num_layers, alpha=alpha).to(device)
    model = GAE_GCN(hidden_dim=hidden_dim, num_layers=num_layers, alpha=alpha).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)

    print("start training......")
    start_time = time.time()

    # 是否读取模型，若是则不训练
    if load_model_flag:
        print(type(model))
        if isinstance(model, GAE_GCN):
            model = load_model_GAE_GCN(load_model_path)
        elif isinstance(model, GAE):
            model = load_model_GAE(load_model_path)
        else:
            raise Exception(f"Load Model Failed, No Such model {type(model)}")

    else:
        for epoch in range(epochs):
            start_time = time.time()
            model.train()

            total_loss = 0.0
            total = 0

            # train
            for batch in tqdm(train_loader, desc=f'Epoch {epoch:02d}'):
                batch = batch.to(device)
                embed_nodes = batch.x

                optimizer.zero_grad()
                nodes_hidden, recon_nodes = model(batch)
                loss, _ = model.loss_gae(nodes_hidden, recon_nodes, embed_nodes, batch.edge_index, batch.batch,
                                         batch.edge_attr)

                # loss, _ = model.caculate_ad_score(nodes, recon_nodes, embed_nodes, batch.edge_index, batch.batch, batch.edge_attr)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total += 1

            scheduler.step()
            total_loss = total_loss / total
            end_time = time.time()
            print('Epoch: %3d/%3d, Train Loss: %.5f, Time cost: %.5f, Learning Rate: %.5f' % (
                epoch + 1, epochs, total_loss, end_time - start_time, float(scheduler.get_last_lr()[0])))

        # 保存模型
        if save_model_flag:
            save_model(model, save_model_path)

    # 停止dropout
    model.eval()

    # calculate scores for threshold
    score_list = []
    node_score_list = []
    trace_id_list = []
    for batch in tqdm(train_loader):
        labels = batch.y
        trace_ids = batch.trace_id
        batch = batch.to(device)
        embed_nodes = batch.x
        error_types = batch.trace_class

        nodes_hidden, recon_nodes = model(batch)

        scores, nodes_scores, nodes_index, trace_id, attr_scores, structure_scores, nodes_attr_scores, nodes_structure_scores = model.caculate_ad_score(nodes_hidden, recon_nodes,
                                                                                    embed_nodes,
                                                                                    batch.edge_index, batch.batch,
                                                                                    batch.edge_attr,
                                                                                    batch.trace_id)

        score_list += list(scores)
        node_score_list += list(nodes_scores)
        trace_id_list += list(trace_id)

    # print(f"trace_id_list {trace_id_list}")
    # print(f"score_list {score_list}")
    # print(f"node_score_list {node_score_list}")

    # for ns in node_score_list:
    #     print(f"ns {ns}  type ns: {type(ns)}")
    #     print(f"ns item {ns.item()} type ns item {type(ns.item())}")


    # 混合高斯分布模型 用于阈值选择
    gm = GaussianMixture(n_components=2, random_state=0)

    gm_X = np.array(node_score_list).reshape(-1, 1)
    print("GaussianMixture fitting......")
    gm = gm.fit(gm_X)  # type: GaussianMixture
    print("GM Means...")
    print(gm.means_)

    print("GM covariances... 方差")
    print(gm.covariances_)

    three_sigmas = []
    for idx in range(len(gm.means_)):
        mean = gm.means_[idx]
        cov = gm.covariances_[idx]
        # 方差-> 标准差
        standard = sqrt(cov)
        three_sigma = mean + 3*standard
        three_sigmas.append(three_sigma)

    gm_node_threshold = max(three_sigmas)
    gm_node_threshold = gm_node_threshold[0]
    print(f"gm_node_threshold {gm_node_threshold}")


    graph_abnormal_threshold = max(score_list)
    node_abnormal_threshold = max(node_score_list)
    max_index = node_score_list.index(node_abnormal_threshold)
    max_abnormal_traceid = trace_id_list[max_index]

    print(f"graph_abnormal_threshold {graph_abnormal_threshold}  \n"
          f"node_abnormal_threshold {node_abnormal_threshold} \n"
          f"max_abnormal_traceid {max_abnormal_traceid}")

    # test
    print("start testing......")
    idx_label_score = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            labels = batch.y
            trace_ids = batch.trace_id
            batch = batch.to(device)
            embed_nodes = batch.x
            error_types = batch.trace_class

            nodes_hidden, recon_nodes = model(batch)
            scores, nodes_scores, nodes_index, trace_id, attr_scores, structure_scores, nodes_attr_scores, nodes_structure_scores = model.caculate_ad_score(nodes_hidden, recon_nodes,
                                                                                        embed_nodes,
                                                                                        batch.edge_index, batch.batch,
                                                                                        batch.edge_attr,
                                                                                        batch.trace_id)

            # Save triples of (idx, label, score) in a list
            idx_label_score += list(zip(labels.cpu().data.numpy().tolist(),
                                        scores.cpu().data.numpy().tolist(),
                                        nodes_scores.cpu().data.numpy().tolist(),
                                        nodes_index.cpu().data.numpy().tolist(),
                                        trace_ids,
                                        error_types,
                                        attr_scores.cpu().data.numpy().tolist(),
                                        structure_scores.cpu().data.numpy().tolist(),
                                        nodes_attr_scores.cpu().data.numpy().tolist(),
                                        nodes_structure_scores.cpu().data.numpy().tolist()))

    test_time = time.time() - start_time
    print('Testing time: %.3f' % test_time)

    # Compute AUC
    labels, scores, nodes_scores, nodes_index, trace_ids, error_types, attr_scores, structure_scores, nodes_attr_scores, nodes_structure_scores = zip(*idx_label_score)
    error_types = np.array(error_types)
    labels = np.array(labels)
    scores = np.array(scores)
    trace_ids = np.array(trace_ids)
    nodes_scores = np.array(nodes_scores)
    nodes_index = np.array(nodes_index)

    test_auc = roc_auc_score(labels, scores)
    test_auc_node = roc_auc_score(labels, nodes_scores)
    print('Test set Graph AUC: {:.2f}%'.format(100. * test_auc))
    print('Test set Node AUC: {:.2f}%'.format(100. * test_auc_node))

    test_attr_auc = roc_auc_score(labels, attr_scores)
    print('Test set Graph Attr AUC: {:.2f}%'.format(100. * test_attr_auc))
    test_structure_auc = roc_auc_score(labels, structure_scores)
    print('Test set Graph Structure AUC: {:.2f}%'.format(100. * test_structure_auc))

    test_node_attr_auc = roc_auc_score(labels, nodes_attr_scores)
    print('Test set Node Attr AUC: {:.2f}%'.format(100. * test_node_attr_auc))
    test_node_structure_auc = roc_auc_score(labels, nodes_structure_scores)
    print('Test set Node Structure AUC: {:.2f}%'.format(100. * test_node_structure_auc))

    node_pred_labels = nodes_scores.copy()
    node_pred_labels[node_pred_labels > gm_node_threshold] = 1
    node_pred_labels[node_pred_labels <= gm_node_threshold] = 0

    gm_precision = precision_score(labels, node_pred_labels)
    gm_recall = recall_score(labels, node_pred_labels)

    print('Test set Node Precision: {:.2f}%'.format(100. * gm_precision))
    print('Test set Node Recall: {:.2f}%'.format(100. * gm_recall))

    # precision recall 曲线
    # precision_score_list = []
    # recall_score_list = []
    #
    # thresholds = np.arange(np.min(nodes_scores), np.max(nodes_scores), step=0.1)
    # for threshold in thresholds:
    #     node_pred_labels = nodes_scores.copy()
    #     node_pred_labels[node_pred_labels > threshold] = 1
    #     node_pred_labels[node_pred_labels <= threshold] = 0
    #
    #     one_precision = precision_score(labels, node_pred_labels)
    #     one_recall = recall_score(labels, node_pred_labels)
    #
    #     precision_score_list.append(one_precision)
    #     recall_score_list.append(one_recall)

    # precision 与 recall 最后一个去掉
    # https://stackoverflow.com/questions/31639016/in-scikits-precision-recall-curve-why-does-thresholds-have-a-different-dimensi
    precision_score_list, recall_score_list, thresholds = precision_recall_curve(labels, nodes_scores)
    precision_score_list = np.delete(precision_score_list, -1)
    recall_score_list = np.delete(recall_score_list, -1)

    np.savetxt(save_data_filename + "precision.csv", precision_score_list, delimiter=",")
    np.savetxt(save_data_filename + "recall.csv", recall_score_list, delimiter=",")
    np.savetxt(save_data_filename + "thresholds.csv", thresholds, delimiter=",")

    plt.plot(precision_score_list, recall_score_list)
    figure_pr_path = save_data_filename + "precision_recall" + ".png"
    print(f"save precision recall image at {figure_pr_path}\n")
    plt.savefig(figure_pr_path)
    plt.clf()

    plt.plot(thresholds, precision_score_list)
    figure_p_path = save_data_filename + "precision" + ".png"
    plt.savefig(figure_p_path)
    plt.clf()

    plt.plot(thresholds, recall_score_list)
    figure_r_path = save_data_filename + "recall" + ".png"
    plt.savefig(figure_r_path)
    plt.clf()


    # 使用阈值计算各类异常的准确率
    # 1. 准备数据
    accuracy_map = {}
    for i, v in enumerate(error_types):
        # v 为 error_type
        if accuracy_map.get(v) is None:
            accuracy_map[v] = {
                "all_cnt": 0,
                "right_cnt": 0,
                "node_scores": [],
                "graph_scores": [],
                "trace_ids": [],
            }

        this_node_score = nodes_scores[i]
        this_graph_score = scores[i]
        this_trace_id = trace_ids[i]

        accuracy_map[v]["node_scores"].append(this_node_score)
        accuracy_map[v]["graph_scores"].append(this_graph_score)
        accuracy_map[v]["trace_ids"].append(this_trace_id)


        # normal label is 0
        # abnormal label is 1
        this_judge = 0
        if this_node_score > gm_node_threshold:
            this_judge = 1

        if this_judge == labels[i]:
            accuracy_map[v]["right_cnt"] += 1
        accuracy_map[v]["all_cnt"] += 1

    # 2. 准确率计算
    for key in accuracy_map.keys():
        right_cnt = accuracy_map[key]['right_cnt']
        all_cnt = accuracy_map[key]['all_cnt']

        type_node_score_list = accuracy_map[key]["node_scores"]
        type_graph_score_list = accuracy_map[key]["graph_scores"]
        type_trace_ids = accuracy_map[key]["trace_ids"]

        print(">>>----------------------------------------------------")
        print(f"trace type={key} ; accuracy = {100. * (right_cnt/all_cnt)} "
              f"all_cnt {all_cnt}  right_cnt {right_cnt} ")

        print("node scores list:")
        for k, g in groupby(sorted(type_node_score_list), key=lambda x: round(x, 2)):
            idx = -2
            for i, score in enumerate(type_node_score_list):
                if round(score, 2) == k:
                    idx = i
                    break
            if idx == -2:
                tid = "no such tid!!!"
            else:
                tid = type_trace_ids[idx]

            print('{}: {}  tid: {}'.format(k, len(list(g)), tid))

        print("<<<----------------------------------------------------")

    print('Finished testing.')
