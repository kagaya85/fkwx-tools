import torch
import time
import utils
import random
import os
import logging
import scipy
import numpy as np
import matplotlib.pyplot as plt
from model import GGNN

from torch_geometric.data import DataLoader
from torch.utils.data import Subset
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve
from dataset import GraphDataset
from sklearn.mixture import GaussianMixture
from math import sqrt
from itertools import groupby
from scipy.stats import median_abs_deviation
from sklearn.manifold import TSNE
from numpy import percentile, quantile

def save_model(model: 'nn.Module', path):
    logger.info(f"save model to {path}")
    torch.save(model.state_dict(), path)

def load_model_GGNN(path) -> GGNN:
    print(f"load GAE model form {path}")
    model = GGNN(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

if __name__ == '__main__':
    epochs = 100
    batch_size = 32
    hidden_dim = 300

    num_layers = 3
    lr = 0.0001
    normal_classes = [0]
    abnormal_classes = [1]
    ratio = [6, 4]
    lr_milestones = [60]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # svdd param
    R = 0.0
    c = None
    R = torch.tensor(R, device=device)
    nu = 0.05
    objective = 'soft-boundary'
    # R updata after warm up epochs
    warm_up_n_epochs = 10

    # config dataset
    dataset = GraphDataset(root='../gae/data/')

    normal_idx = utils.get_target_label_idx(dataset.data.y.clone().data.cpu().numpy(), normal_classes)
    abnormal_idx = utils.get_target_label_idx(dataset.data.y.clone().data.cpu().numpy(), abnormal_classes)

    random.shuffle(abnormal_idx)
    random.shuffle(normal_idx)

    # train_normal = normal_idx[:65490]
    # test_normal = normal_idx[70090:]
    # val_normal = normal_idx[65490:70090]

    train_normal = normal_idx[:int(len(normal_idx) * 0.6)]
    test_normal = normal_idx[int(len(normal_idx) * 0.7):]
    val_normal = normal_idx[int(len(normal_idx) * 0.6):int(len(normal_idx) * 0.7)]
    train_abnormal = abnormal_idx[:int(len(abnormal_idx) * 0.1)]
    test_abnormal = abnormal_idx[int(len(abnormal_idx) * 0.1):]
    val_abnormal = abnormal_idx[int(len(abnormal_idx) * 0):int(len(abnormal_idx) * 0)]

    train_dataset = Subset(dataset, train_normal + train_abnormal)
    test_dataset = Subset(dataset, test_abnormal + test_normal)
    val_dataset = Subset(dataset, val_normal + val_abnormal)

    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Set up logging
    xp_path = f"./output/newR" + time.strftime("%Y-%m-%d_%Hh-%Mm-%Ss", time.localtime())
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
    model = GGNN(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)

    c = utils.init_center_c(train_loader, model, device)

    logger.info(
        "************************************** Start Train ****************************************************")
    logger.info(
        "batch_size=%s, epochs=%s, ratio=%s, device=%s" % (
            batch_size, epochs, ratio,  device))
    logger.info(
        "objective=%s, lr=%s, num_layers=%s, nu=%s, c=%s" % (
            objective, lr, num_layers, nu, c))
    logger.info(
        "train_normal=%s, test_normal=%s, val_normal=%s, train_abnormal=%s, test_abnormal=%s, val_abnormal=%s" % (
            len(train_normal), len(test_normal), len(val_normal), len(train_abnormal), len(test_abnormal), len(val_abnormal)))
    logger.info(
        "new r version")

    train_start_time = time.time()
    if load_model_flag:
        print(type(model))
        if isinstance(model, GGNN):
            model = load_model_GGNN(load_model_path)
        else:
            raise Exception(f"Load Model Failed, No Such model {type(model)}")

    else:

        for epoch in range(epochs):
            start_time = time.time()

            model.train()

            total_loss = 0.0
            total = 0

            # train
            dist_set = []
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()

                output, _ = model(batch)
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

                dist_set.append(dist)

            if (objective == 'soft-boundary') and (epoch >= warm_up_n_epochs):
                R.data = torch.tensor(utils.get_radius(torch.cat(tuple(dist_set), 0), nu), device=device)

            scheduler.step()
            total_loss = total_loss / total
            end_time = time.time()
            logger.info('Epoch: %3d/%3d, Train Loss: %.10f, Time cost: %.5f, Learning Rate: %.5f' % (
                epoch + 1, epochs, total_loss, end_time - start_time, float(scheduler.get_last_lr()[0])))

    train_time = time.time() - train_start_time
    logger.info('Training time: %.3f' % train_time)
    logger.info("last r is %s" % R.data)

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
            output, _ = model(batch)
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
    mad = median_abs_deviation(train_scores)
    median = np.median(train_scores)
    mad_threshold = median + 3 * mad
    mad_threshold_2n = median + 2 * mad
    mad_threshold_n = median + mad
    mad_threshold_4n = median + 4 * mad
    mad_threshold_5n = median + 5 * mad
    mad_threshold_6n = median + 6 * mad
    mad_threshold_7n = median + 7 * mad
    mad_threshold_8n = median + 8 * mad
    logger.info(f"3 n mad threshold {mad_threshold}")
    logger.info(f'median is {median}')
    logger.info(f"mad is {mad}")
    logger.info(f"2 n mad threshold {mad_threshold_2n}")
    logger.info(f"1 n mad threshold {mad_threshold_n}")
    logger.info(f"4 n mad threshold {mad_threshold_4n}")
    logger.info(f"5 n mad threshold {mad_threshold_5n}")
    logger.info(f"6 n mad threshold {mad_threshold_6n}")
    logger.info(f"7 n mad threshold {mad_threshold_7n}")
    logger.info(f"8 n mad threshold {mad_threshold_8n}")



    # 4 分位
    lower_q = np.quantile(train_scores, 0.25, interpolation='lower')  # 下四分位数
    higher_q = np.quantile(train_scores, 0.75, interpolation='higher')  # 上四分位数
    int_r = higher_q - lower_q  # 四分位距

    q_15 = higher_q + 1.5*int_r
    q_30 = lower_q + 3*int_r


    # 置信区间
    mean = train_scores.mean()
    n = len(train_scores)
    std = train_scores.std()

    confidence_threshold = mean + 2.576 * (std / np.sqrt(n))
    logger.info(f"99% threshold {confidence_threshold}")

    # 置信区间
    mean = train_scores.mean()
    n = len(train_scores)
    std = train_scores.std()

    confidence_threshold_95 = mean + 1.96 * (std / np.sqrt(n))
    logger.info(f"95% threshold {confidence_threshold_95}")

    # 3sigma
    mean = train_scores.mean()
    n = len(train_scores)
    std = train_scores.std()

    sigma_3 = mean + 3 * std
    logger.info(f"3 sigma {sigma_3}")

    # 2sigma
    mean = train_scores.mean()
    n = len(train_scores)
    std = train_scores.std()

    sigma_2 = mean + 2 * std
    logger.info(f"2 sigma {sigma_2}")

    # 1sigma
    mean = train_scores.mean()
    n = len(train_scores)
    std = train_scores.std()

    sigma_1 = mean + 1 * std
    logger.info(f"2 sigma {sigma_1}")

    # 6sigma
    mean = train_scores.mean()
    n = len(train_scores)
    std = train_scores.std()

    sigma_6 = mean + 6 * std
    logger.info(f"6 sigma {sigma_6}")

    # # 混合高斯分布模型 用于阈值选择
    # gm = GaussianMixture(n_components=2, random_state=0)
    #
    # gm_X = np.array(train_scores).reshape(-1, 1)
    # logger.info("GaussianMixture fitting......")
    # gm = gm.fit(gm_X)  # type: GaussianMixture
    # logger.info("GM Means...")
    # logger.info(gm.means_)
    #
    # logger.info("GM covariances... 方差")
    # logger.info(gm.covariances_)
    #
    # three_sigmas = []
    # for idx in range(len(gm.means_)):
    #     mean = gm.means_[idx]
    #     cov = gm.covariances_[idx]
    #     # 方差-> 标准差
    #     standard = sqrt(cov)
    #     three_sigma = mean + 3*standard
    #     three_sigmas.append(three_sigma)
    #
    # gm_node_threshold = max(three_sigmas)
    # logger.info(f"gm_node_threshold {gm_node_threshold}")


    # test
    start_time = time.time()
    idx_label_score = []
    with torch.no_grad():
        for batch in test_loader:
            labels = batch.y
            batch = batch.to(device)
            output, attention_scores = model(batch)
            dist = torch.sum((output - c) ** 2, dim=1)
            if objective == 'soft-boundary':
                scores = dist - R ** 2
            else:
                scores = dist

            nodes_scores = []
            for ng in range(batch.batch.max() + 1):
                nodes_attention_scores = attention_scores[batch.batch==ng]
                nodes_attention_scores = nodes_attention_scores.cpu().data.numpy()
                nodes_attention_scores = np.array2string(nodes_attention_scores, formatter={'float_kind': lambda x: "%.10f" % x})
                nodes_scores.append(nodes_attention_scores)

            # Save triples of (idx, label, score) in a list
            idx_label_score += list(zip(labels.cpu().data.numpy().tolist(),
                                        scores.cpu().data.numpy().tolist(),
                                        batch.trace_id,
                                        batch.error_trace_type,
                                        nodes_scores,
                                        output.cpu().data.numpy().tolist()))

    test_time = time.time() - start_time
    logger.info('Testing time: %.3f' % test_time)

    test_scores = idx_label_score

    # Compute AUC
    labels, scores, trace_ids, error_types, _, graph_embedding = zip(*idx_label_score)
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

    # 99% confidence
    node_pred_labels = scores.copy()
    node_pred_labels[node_pred_labels > confidence_threshold] = 1
    node_pred_labels[node_pred_labels < confidence_threshold] = 0

    gm_precision = precision_score(labels, node_pred_labels)
    gm_recall = recall_score(labels, node_pred_labels)
    f1 = f1_score(labels, node_pred_labels)

    logger.info('Test confidence Precision: {:.2f}%'.format(100. * gm_precision))
    logger.info('Test confidence Recall: {:.2f}%'.format(100. * gm_recall))
    logger.info('Test confidence F1: {:.2f}%'.format(100. * f1))

    # 95% confidence
    node_pred_labels = scores.copy()
    node_pred_labels[node_pred_labels > confidence_threshold_95] = 1
    node_pred_labels[node_pred_labels < confidence_threshold_95] = 0

    gm_precision = precision_score(labels, node_pred_labels)
    gm_recall = recall_score(labels, node_pred_labels)
    f1 = f1_score(labels, node_pred_labels)

    logger.info('Test 95 confidence Precision: {:.2f}%'.format(100. * gm_precision))
    logger.info('Test 95 confidence Recall: {:.2f}%'.format(100. * gm_recall))
    logger.info('Test 95 confidence F1: {:.2f}%'.format(100. * f1))

    # gmm
    # node_pred_labels = scores.copy()
    # node_pred_labels[node_pred_labels > gm_node_threshold] = 1
    # node_pred_labels[node_pred_labels < gm_node_threshold] = 0
    #
    # gm_precision = precision_score(labels, node_pred_labels)
    # gm_recall = recall_score(labels, node_pred_labels)
    # f1 = f1_score(labels, node_pred_labels)
    #
    # logger.info('Test gm Precision: {:.2f}%'.format(100. * gm_precision))
    # logger.info('Test gm Recall: {:.2f}%'.format(100. * gm_recall))
    # logger.info('Test gm F1: {:.2f}%'.format(100. * f1))

    # 0
    node_pred_labels = scores.copy()
    node_pred_labels[node_pred_labels > 0] = 1
    node_pred_labels[node_pred_labels < 0] = 0

    gm_precision = precision_score(labels, node_pred_labels)
    gm_recall = recall_score(labels, node_pred_labels)
    f1 = f1_score(labels, node_pred_labels)

    logger.info('Test 0 Precision: {:.2f}%'.format(100. * gm_precision))
    logger.info('Test 0 Recall: {:.2f}%'.format(100. * gm_recall))
    logger.info('Test 0 F1: {:.2f}%'.format(100. * f1))

    # mad
    node_pred_labels = scores.copy()
    node_pred_labels[node_pred_labels > mad_threshold] = 1
    node_pred_labels[node_pred_labels < mad_threshold] = 0

    gm_precision = precision_score(labels, node_pred_labels)
    gm_recall = recall_score(labels, node_pred_labels)
    f1 = f1_score(labels, node_pred_labels)

    logger.info('Test 3 mad Precision: {:.2f}%'.format(100. * gm_precision))
    logger.info('Test 3 mad Recall: {:.2f}%'.format(100. * gm_recall))
    logger.info('Test 3 mad F1: {:.2f}%'.format(100. * f1))

    # mad 2n
    node_pred_labels = scores.copy()
    node_pred_labels[node_pred_labels > mad_threshold_2n] = 1
    node_pred_labels[node_pred_labels < mad_threshold_2n] = 0

    gm_precision = precision_score(labels, node_pred_labels)
    gm_recall = recall_score(labels, node_pred_labels)
    f1 = f1_score(labels, node_pred_labels)

    logger.info('Test 2 mad Precision: {:.2f}%'.format(100. * gm_precision))
    logger.info('Test 2 mad Recall: {:.2f}%'.format(100. * gm_recall))
    logger.info('Test 2 mad F1: {:.2f}%'.format(100. * f1))

    # mad n
    node_pred_labels = scores.copy()
    node_pred_labels[node_pred_labels > mad_threshold_n] = 1
    node_pred_labels[node_pred_labels < mad_threshold_n] = 0

    gm_precision = precision_score(labels, node_pred_labels)
    gm_recall = recall_score(labels, node_pred_labels)
    f1 = f1_score(labels, node_pred_labels)

    logger.info('Test 1 mad Precision: {:.2f}%'.format(100. * gm_precision))
    logger.info('Test 1 mad Recall: {:.2f}%'.format(100. * gm_recall))
    logger.info('Test 1 mad F1: {:.2f}%'.format(100. * f1))

    # mad 4n
    node_pred_labels = scores.copy()
    node_pred_labels[node_pred_labels > mad_threshold_4n] = 1
    node_pred_labels[node_pred_labels < mad_threshold_4n] = 0

    gm_precision = precision_score(labels, node_pred_labels)
    gm_recall = recall_score(labels, node_pred_labels)
    f1 = f1_score(labels, node_pred_labels)

    logger.info('Test 4 mad Precision: {:.2f}%'.format(100. * gm_precision))
    logger.info('Test 4 mad Recall: {:.2f}%'.format(100. * gm_recall))
    logger.info('Test 4 mad F1: {:.2f}%'.format(100. * f1))


    # mad 5n
    node_pred_labels = scores.copy()
    node_pred_labels[node_pred_labels > mad_threshold_5n] = 1
    node_pred_labels[node_pred_labels < mad_threshold_5n] = 0

    gm_precision = precision_score(labels, node_pred_labels)
    gm_recall = recall_score(labels, node_pred_labels)
    f1 = f1_score(labels, node_pred_labels)

    logger.info('Test 5 mad Precision: {:.2f}%'.format(100. * gm_precision))
    logger.info('Test 5 mad Recall: {:.2f}%'.format(100. * gm_recall))
    logger.info('Test 5 mad F1: {:.2f}%'.format(100. * f1))


    # mad 6n
    node_pred_labels = scores.copy()
    node_pred_labels[node_pred_labels > mad_threshold_6n] = 1
    node_pred_labels[node_pred_labels < mad_threshold_6n] = 0

    gm_precision = precision_score(labels, node_pred_labels)
    gm_recall = recall_score(labels, node_pred_labels)
    f1 = f1_score(labels, node_pred_labels)

    logger.info('Test 6 mad Precision: {:.2f}%'.format(100. * gm_precision))
    logger.info('Test 6 mad Recall: {:.2f}%'.format(100. * gm_recall))
    logger.info('Test 6 mad F1: {:.2f}%'.format(100. * f1))

    # mad 7n
    node_pred_labels = scores.copy()
    node_pred_labels[node_pred_labels > mad_threshold_7n] = 1
    node_pred_labels[node_pred_labels < mad_threshold_7n] = 0

    gm_precision = precision_score(labels, node_pred_labels)
    gm_recall = recall_score(labels, node_pred_labels)
    f1 = f1_score(labels, node_pred_labels)

    logger.info('Test 7 mad Precision: {:.2f}%'.format(100. * gm_precision))
    logger.info('Test 7 mad Recall: {:.2f}%'.format(100. * gm_recall))
    logger.info('Test 7 mad F1: {:.2f}%'.format(100. * f1))

    # mad 8n
    node_pred_labels = scores.copy()
    node_pred_labels[node_pred_labels > mad_threshold_8n] = 1
    node_pred_labels[node_pred_labels < mad_threshold_8n] = 0

    gm_precision = precision_score(labels, node_pred_labels)
    gm_recall = recall_score(labels, node_pred_labels)
    f1 = f1_score(labels, node_pred_labels)

    logger.info('Test 8 mad Precision: {:.2f}%'.format(100. * gm_precision))
    logger.info('Test 8 mad Recall: {:.2f}%'.format(100. * gm_recall))
    logger.info('Test 8 mad F1: {:.2f}%'.format(100. * f1))


    # sigam 3
    node_pred_labels = scores.copy()
    node_pred_labels[node_pred_labels > sigma_3] = 1
    node_pred_labels[node_pred_labels < sigma_3] = 0

    gm_precision = precision_score(labels, node_pred_labels)
    gm_recall = recall_score(labels, node_pred_labels)
    f1 = f1_score(labels, node_pred_labels)

    logger.info('Test sigma_3 Precision: {:.2f}%'.format(100. * gm_precision))
    logger.info('Test sigma_3 Recall: {:.2f}%'.format(100. * gm_recall))
    logger.info('Test sigma_3 F1: {:.2f}%'.format(100. * f1))

    # sigam 2
    node_pred_labels = scores.copy()
    node_pred_labels[node_pred_labels > sigma_2] = 1
    node_pred_labels[node_pred_labels < sigma_2] = 0

    gm_precision = precision_score(labels, node_pred_labels)
    gm_recall = recall_score(labels, node_pred_labels)
    f1 = f1_score(labels, node_pred_labels)

    logger.info('Test sigma_2 Precision: {:.2f}%'.format(100. * gm_precision))
    logger.info('Test sigma_2 Recall: {:.2f}%'.format(100. * gm_recall))
    logger.info('Test sigma_2 F1: {:.2f}%'.format(100. * f1))

    # sigam 6
    node_pred_labels = scores.copy()
    node_pred_labels[node_pred_labels > sigma_6] = 1
    node_pred_labels[node_pred_labels < sigma_6] = 0

    gm_precision = precision_score(labels, node_pred_labels)
    gm_recall = recall_score(labels, node_pred_labels)
    f1 = f1_score(labels, node_pred_labels)

    logger.info('Test sigma_6 Precision: {:.2f}%'.format(100. * gm_precision))
    logger.info('Test sigma_6 Recall: {:.2f}%'.format(100. * gm_recall))
    logger.info('Test sigma_6 F1: {:.2f}%'.format(100. * f1))

    # sigam 1
    node_pred_labels = scores.copy()
    node_pred_labels[node_pred_labels > sigma_1] = 1
    node_pred_labels[node_pred_labels < sigma_1] = 0

    gm_precision = precision_score(labels, node_pred_labels)
    gm_recall = recall_score(labels, node_pred_labels)
    f1 = f1_score(labels, node_pred_labels)

    logger.info('Test sigma_1 Precision: {:.2f}%'.format(100. * gm_precision))
    logger.info('Test sigma_1 Recall: {:.2f}%'.format(100. * gm_recall))
    logger.info('Test sigma_1 F1: {:.2f}%'.format(100. * f1))

    # 1.5四分位数
    node_pred_labels = scores.copy()
    node_pred_labels[node_pred_labels > q_15] = 1
    node_pred_labels[node_pred_labels < q_15] = 0

    gm_precision = precision_score(labels, node_pred_labels)
    gm_recall = recall_score(labels, node_pred_labels)
    f1 = f1_score(labels, node_pred_labels)

    logger.info('Test q_15 Precision: {:.2f}%'.format(100. * gm_precision))
    logger.info('Test q_15 Recall: {:.2f}%'.format(100. * gm_recall))
    logger.info('Test q_15 F1: {:.2f}%'.format(100. * f1))

    # 3四分位数
    node_pred_labels = scores.copy()
    node_pred_labels[node_pred_labels > q_30] = 1
    node_pred_labels[node_pred_labels < q_30] = 0

    gm_precision = precision_score(labels, node_pred_labels)
    gm_recall = recall_score(labels, node_pred_labels)
    f1 = f1_score(labels, node_pred_labels)

    logger.info('Test q_30 Precision: {:.2f}%'.format(100. * gm_precision))
    logger.info('Test q_30 Recall: {:.2f}%'.format(100. * gm_recall))
    logger.info('Test q_30 F1: {:.2f}%'.format(100. * f1))


    # prec, recall, thresholds = precision_recall_curve(labels, scores)
    # p_r_curve = list(zip(prec, recall, thresholds))
    # utils.write_csv_file(xp_path, 'p_r_curve.csv', ("prec", "recall", "thresholds"),p_r_curve)

    # pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()



    # logger.info("1-sigma")
    # # 使用阈值计算各类异常的准确率
    # # 1. 准备数据
    # accuracy_map = {}
    # for i, v in enumerate(error_types):
    #     # v 为 error_type
    #     if accuracy_map.get(v) is None:
    #         accuracy_map[v] = {
    #             "all_cnt": 0,
    #             "right_cnt": 0,
    #             "graph_scores": [],
    #             "trace_ids": [],
    #         }
    #
    #     this_graph_score = scores[i]
    #     this_trace_id = trace_ids[i]
    #
    #     accuracy_map[v]["graph_scores"].append(this_graph_score)
    #     accuracy_map[v]["trace_ids"].append(this_trace_id)
    #
    #
    #     # normal label is 0
    #     # abnormal label is 1
    #     this_judge = 0
    #     if this_graph_score > sigma_1:
    #         this_judge = 1
    #
    #     if this_judge == labels[i]:
    #         accuracy_map[v]["right_cnt"] += 1
    #     accuracy_map[v]["all_cnt"] += 1
    #
    # # 2. 准确率计算
    # for key in accuracy_map.keys():
    #     right_cnt = accuracy_map[key]['right_cnt']
    #     all_cnt = accuracy_map[key]['all_cnt']
    #
    #     type_graph_score_list = accuracy_map[key]["graph_scores"]
    #     type_trace_ids = accuracy_map[key]["trace_ids"]
    #
    #     logger.info(">>>----------------------------------------------------")
    #     logger.info(f"trace type={key} ; accuracy = {100. * (right_cnt/all_cnt)} "
    #           f"all_cnt {all_cnt}  right_cnt {right_cnt} ")
    #
    #     logger.info("<<<----------------------------------------------------")
    #
    #
    # logger.info("2-sigma")
    # # 使用阈值计算各类异常的准确率
    # accuracy_map = {}
    # for i, v in enumerate(error_types):
    #     # v 为 error_type
    #     if accuracy_map.get(v) is None:
    #         accuracy_map[v] = {
    #             "all_cnt": 0,
    #             "right_cnt": 0,
    #             "graph_scores": [],
    #             "trace_ids": [],
    #         }
    #
    #     this_graph_score = scores[i]
    #     this_trace_id = trace_ids[i]
    #
    #     accuracy_map[v]["graph_scores"].append(this_graph_score)
    #     accuracy_map[v]["trace_ids"].append(this_trace_id)
    #
    #
    #     # normal label is 0
    #     # abnormal label is 1
    #     this_judge = 0
    #     if this_graph_score > sigma_2:
    #         this_judge = 1
    #
    #     if this_judge == labels[i]:
    #         accuracy_map[v]["right_cnt"] += 1
    #     accuracy_map[v]["all_cnt"] += 1
    #
    # # 2. 准确率计算
    # for key in accuracy_map.keys():
    #     right_cnt = accuracy_map[key]['right_cnt']
    #     all_cnt = accuracy_map[key]['all_cnt']
    #
    #     type_graph_score_list = accuracy_map[key]["graph_scores"]
    #     type_trace_ids = accuracy_map[key]["trace_ids"]
    #
    #     logger.info(">>>----------------------------------------------------")
    #     logger.info(f"trace type={key} ; accuracy = {100. * (right_cnt/all_cnt)} "
    #           f"all_cnt {all_cnt}  right_cnt {right_cnt} ")
    #
    #     logger.info("<<<----------------------------------------------------")
    #
    # utils.write_csv_file(xp_path, 'result.csv', ("label", "score", "trace_id", "error_type", "nodes_score", "graph_embedding"),
    #                idx_label_score)


    logger.info("0")
    # 使用阈值计算各类异常的准确率
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
        if this_graph_score > 0:
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
        logger.info(f"trace type={key} ; accuracy = {100. * (right_cnt / all_cnt)} "
                    f"all_cnt {all_cnt}  right_cnt {right_cnt} ")

        logger.info("<<<----------------------------------------------------")

    utils.write_csv_file(xp_path, 'result.csv',
                         ("label", "score", "trace_id", "error_type", "nodes_score", "graph_embedding"),
                         idx_label_score)

    # tsne = TSNE()
    # x = tsne.fit_transform(graph_embedding)
    # plt.scatter(x[:, 0], x[:, 1], c=labels, marker='o', s=40, cmap=plt.cm.Spectral)
    # plt.savefig(xp_path+'/t-sne-test.jpg')
    #
    #
    # low_dim = list(zip(x, labels))
    # utils.write_csv_file(xp_path, 'low_dim.csv', ("graph", "labels"), low_dim)

    logger.info('Finished testing.')








