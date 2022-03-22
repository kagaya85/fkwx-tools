import torch
import time
import utils
import random
import os
import logging
import scipy
import numpy as np
from model import GGNN, GGNN_NODE

from torch_geometric.data import DataLoader
from torch.utils.data import Subset
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve
from dataset import GraphDataset
from sklearn.mixture import GaussianMixture
from math import sqrt
from itertools import groupby
from scipy.stats import median_absolute_deviation


def save_model(model: 'nn.Module', path):
    logger.info(f"save model to {path}")
    torch.save(model.state_dict(), path)

def load_model_GGNN_NODE(path) -> GGNN_NODE:
    print(f"load GAE model form {path}")
    model = GGNN(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


if __name__ == '__main__':
    epochs = 50
    batch_size = 32
    hidden_dim = 300

    num_layers = 3
    lr = 0.0001
    normal_classes = [0]
    abnormal_classes = [1]
    ratio = [6, 4]
    lr_milestones = [60, 80]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # svdd param
    R = 0.0
    c = None
    R = torch.tensor(R, device=device)
    nu = 0.1
    objective = 'one-class'
    # R updata after warm up epochs
    warm_up_n_epochs = 10

    # config dataset
    dataset = GraphDataset(root='../gae/data/')

    normal_idx = utils.get_target_label_idx(dataset.data.y.clone().data.cpu().numpy(), normal_classes)
    abnormal_idx = utils.get_target_label_idx(dataset.data.y.clone().data.cpu().numpy(), abnormal_classes)

    random.shuffle(abnormal_idx)
    random.shuffle(normal_idx)

    train_normal = normal_idx[:int(len(normal_idx) * 0.6)]
    test_normal = normal_idx[int(len(normal_idx) * 0.7):]
    val_normal = normal_idx[int(len(normal_idx) * 0.6):int(len(normal_idx) * 0.7)]
    train_abnormal = abnormal_idx[:int(len(abnormal_idx) * 0)]
    test_abnormal = abnormal_idx[int(len(abnormal_idx) * 0):]
    val_abnormal = abnormal_idx[int(len(abnormal_idx) * 0):int(len(abnormal_idx) * 0)]

    train_dataset = Subset(dataset, train_normal + train_abnormal)
    test_dataset = Subset(dataset, test_abnormal + test_normal)
    val_dataset = Subset(dataset, val_normal + val_abnormal)

    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Set up logging
    xp_path = f"./output/" + time.strftime("%Y-%m-%d_%Hh-%Mm-%Ss", time.localtime())
    if not os.path.exists(xp_path):
        os.makedirs(xp_path)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    load_model_flag = False
    save_model_flag = True
    save_model_path = xp_path + "/save_" + str(time.time()).split(".")[0] + ".model"
    load_model_path = "./output/2021-08-20_16h-52m-04s/save_1629449524.model"

    # train
    model = GGNN_NODE(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)

    c = utils.init_center_c(train_loader, model, device)

    logger.info(
        "************************************** Start Train ****************************************************")
    logger.info(
        "batch_size=%s, epochs=%s, ratio=%s, device=%s" % (
            batch_size, epochs, ratio,  device))
    logger.info(
        "objective=%s, lr=%s, num_layers=%s, c=%s" % (
            objective, lr, num_layers, c))


    if load_model_flag:
        print(type(model))
        if isinstance(model, GGNN_NODE):
            model = load_model_GGNN_NODE(load_model_path)
        else:
            raise Exception(f"Load Model Failed, No Such model {type(model)}")

    else:

        for epoch in range(epochs):
            start_time = time.time()

            model.train()

            total_loss = 0.0
            total = 0

            # train
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()

                output = model(batch)
                dist = torch.sum((output - c) ** 2, dim=1)
                if objective == 'soft-boundary':
                    scores = dist - R ** 2
                    loss = R ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)

                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                total += 1

                if (objective == 'soft-boundary') and (epoch >= warm_up_n_epochs):
                    R.data = torch.tensor(utils.get_radius(dist, nu), device=device)

            scheduler.step()
            total_loss = total_loss / total
            end_time = time.time()
            logger.info('Epoch: %3d/%3d, Train Loss: %.10f, Time cost: %.5f, Learning Rate: %.5f' % (
                epoch + 1, epochs, total_loss, end_time - start_time, float(scheduler.get_last_lr()[0])))


    # eval
    if save_model_flag:
        save_model(model, save_model_path)


    # calculate threshold
    start_time = time.time()
    threshold_score = []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            output = model(batch)
            dist = torch.sum((output - c) ** 2, dim=1)
            if objective == 'soft-boundary':
                scores = dist - R ** 2
            else:
                scores = dist

            # Save triples of (idx, label, score) in a list
            threshold_score += list(zip(scores.cpu().data.numpy().tolist(),
                                        batch.trace_id))

    test_time = time.time() - start_time
    logger.info('Calculate threshold time: %.3f' % test_time)

    train_scores, _ = zip(*threshold_score)
    train_scores = np.array(train_scores)

    # MAD
    mad = median_absolute_deviation(train_scores)
    median = np.median(train_scores)
    mad_threshold = median + 3 * mad
    logger.info(f"mad threshold {mad_threshold}")


    # 置信区间
    mean = train_scores.mean()
    n = len(train_scores)
    std = train_scores.std()

    threshold = mean + 1.96 * (std / np.sqrt(n))
    logger.info(f"95% threshold {threshold}")

    # 混合高斯分布模型 用于阈值选择
    gm = GaussianMixture(n_components=2, random_state=0)

    gm_X = np.array(train_scores).reshape(-1, 1)
    logger.info("GaussianMixture fitting......")
    gm = gm.fit(gm_X)  # type: GaussianMixture
    logger.info("GM Means...")
    logger.info(gm.means_)

    logger.info("GM covariances... 方差")
    logger.info(gm.covariances_)

    three_sigmas = []
    for idx in range(len(gm.means_)):
        mean = gm.means_[idx]
        cov = gm.covariances_[idx]
        # 方差-> 标准差
        standard = sqrt(cov)
        three_sigma = mean + 3*standard
        three_sigmas.append(three_sigma)

    gm_node_threshold = max(three_sigmas)
    logger.info(f"gm_node_threshold {gm_node_threshold}")


    # test
    start_time = time.time()
    idx_label_score = []
    with torch.no_grad():
        for batch in test_loader:
            labels = batch.y
            batch = batch.to(device)
            output = model(batch)
            dist = torch.sum((output - c) ** 2, dim=1)
            if objective == 'soft-boundary':
                scores = dist - R ** 2
            else:
                scores = dist

            nodes_scores = []
            max_nodes_scores = []
            for ng in range(batch.batch.max() + 1):
                graph_nodes_scores = scores[batch.batch==ng]
                graph_nodes_scores = graph_nodes_scores.cpu().data.numpy()
                graph_nodes_scores_str = np.array2string(graph_nodes_scores, formatter={'float_kind': lambda x: "%.10f" % x})
                nodes_scores.append(graph_nodes_scores_str)
                max_nodes_scores.append(max(graph_nodes_scores.tolist()))

            # Save triples of (idx, label, score) in a list
            idx_label_score += list(zip(labels.cpu().data.numpy().tolist(),
                                        max_nodes_scores,
                                        batch.trace_id,
                                        batch.error_trace_type,
                                        nodes_scores))

    test_time = time.time() - start_time
    logger.info('Testing time: %.3f' % test_time)

    test_scores = idx_label_score

    # Compute AUC
    labels, scores, trace_ids, error_types, _ = zip(*idx_label_score)
    labels = np.array(labels)
    # for label in np.nditer(labels, op_flags=['readwrite']):
    #     if label == 0:
    #         label[...] = 1
    #     elif label == 1:
    #         label[...] = 0
    scores = np.array(scores)
    trace_ids = np.array(trace_ids)

    test_auc = roc_auc_score(labels, scores)
    logger.info('Test set AUC: {:.2f}%'.format(100. * test_auc))

    # 95% confidence
    node_pred_labels = scores.copy()
    node_pred_labels[node_pred_labels > threshold] = 1
    node_pred_labels[node_pred_labels < threshold] = 0

    gm_precision = precision_score(labels, node_pred_labels)
    gm_recall = recall_score(labels, node_pred_labels)
    f1 = f1_score(labels, node_pred_labels)

    logger.info('Test confidence Precision: {:.2f}%'.format(100. * gm_precision))
    logger.info('Test confidence Recall: {:.2f}%'.format(100. * gm_recall))
    logger.info('Test confidence F1: {:.2f}%'.format(100. * f1))

    # gmm
    node_pred_labels = scores.copy()
    node_pred_labels[node_pred_labels > gm_node_threshold] = 1
    node_pred_labels[node_pred_labels < gm_node_threshold] = 0

    gm_precision = precision_score(labels, node_pred_labels)
    gm_recall = recall_score(labels, node_pred_labels)
    f1 = f1_score(labels, node_pred_labels)

    logger.info('Test gm Precision: {:.2f}%'.format(100. * gm_precision))
    logger.info('Test gm Recall: {:.2f}%'.format(100. * gm_recall))
    logger.info('Test gm F1: {:.2f}%'.format(100. * f1))

    # mad
    node_pred_labels = scores.copy()
    node_pred_labels[node_pred_labels > mad_threshold] = 1
    node_pred_labels[node_pred_labels < mad_threshold] = 0

    gm_precision = precision_score(labels, node_pred_labels)
    gm_recall = recall_score(labels, node_pred_labels)
    f1 = f1_score(labels, node_pred_labels)

    logger.info('Test mad Precision: {:.2f}%'.format(100. * gm_precision))
    logger.info('Test mad Recall: {:.2f}%'.format(100. * gm_recall))
    logger.info('Test mad F1: {:.2f}%'.format(100. * f1))

    prec, recall, thresholds = precision_recall_curve(labels, scores)

    p_r_curve = list(zip(prec, recall, thresholds))

    utils.write_csv_file(xp_path, 'p_r_curve.csv', ("prec", "recall", "thresholds"),p_r_curve)

    # pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()

    # 使用阈值计算各类异常的准确率
    # 1. 准备数据
    accuracy_map = {}
    for i, v in enumerate(error_types):
        # v 为 error_type
        if accuracy_map.get(v) is None:
            accuracy_map[v] = {
                "all_cnt": 0,
                "right_cnt": 0,
                "graph_scores": [],
                "trace_ids": [],
            }

        this_graph_score = scores[i]
        this_trace_id = trace_ids[i]

        accuracy_map[v]["graph_scores"].append(this_graph_score)
        accuracy_map[v]["trace_ids"].append(this_trace_id)


        # normal label is 0
        # abnormal label is 1
        this_judge = 0
        if this_graph_score > mad_threshold:
            this_judge = 1

        if this_judge == labels[i]:
            accuracy_map[v]["right_cnt"] += 1
        accuracy_map[v]["all_cnt"] += 1

    # 2. 准确率计算
    for key in accuracy_map.keys():
        right_cnt = accuracy_map[key]['right_cnt']
        all_cnt = accuracy_map[key]['all_cnt']

        type_graph_score_list = accuracy_map[key]["graph_scores"]
        type_trace_ids = accuracy_map[key]["trace_ids"]

        logger.info(">>>----------------------------------------------------")
        logger.info(f"trace type={key} ; accuracy = {100. * (right_cnt/all_cnt)} "
              f"all_cnt {all_cnt}  right_cnt {right_cnt} ")

        logger.info("node scores list:")
        for k, g in groupby(sorted(type_graph_score_list), key=lambda x: round(x, 6)):
            idx = -2
            for i, score in enumerate(type_graph_score_list):
                if round(score, 6) == k:
                    idx = i
                    break
            if idx == -2:
                tid = "no such tid!!!"
            else:
                tid = type_trace_ids[idx]

            logger.info('{}: {}  tid: {}'.format(k, len(list(g)), tid))

        logger.info("<<<----------------------------------------------------")

    utils.write_csv_file(xp_path, 'result.csv', ("label", "score", "trace_id", "error_type", "nodes_score"),
                   idx_label_score)

    logger.info('Finished testing.')
