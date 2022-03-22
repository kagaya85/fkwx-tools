import torch
import time
import utils
import numpy as np
from model import GGNN, GGNN_O
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch.utils.data import Subset
from sklearn.metrics import roc_auc_score


if __name__ == '__main__':
    epochs = 100
    batch_size = 64
    hidden_dim = 50
    num_layers = 5
    lr = 0.0001
    normal_classes = [1]
    abnormal_classes = [0]
    ratio = [7, 3]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # svdd param
    R = 0.0
    c = None
    R = torch.tensor(R, device=device)
    nu = 0.1
    objective = 'one-class'
    # R updata after warm up epochs
    warm_up_n_epochs = 10

    # test dataset
    dataset = TUDataset(root='./tmp/AIDS', name='AIDS', use_node_attr = True)

    train_idx_normal = utils.get_target_label_idx(dataset.data.y.clone().data.cpu().numpy(), normal_classes)
    test_idx_abnormal = utils.get_target_label_idx(dataset.data.y.clone().data.cpu().numpy(), abnormal_classes)

    train_normal = train_idx_normal[:int(len(train_idx_normal)*ratio[0]/sum(ratio))]
    test_normal = train_idx_normal[int(len(train_idx_normal)*ratio[0]/sum(ratio)):]
    train_abnormal = test_idx_abnormal[:int(len(test_idx_abnormal)*0)]
    test_abnormal = test_idx_abnormal[int(len(test_idx_abnormal)*0):]

    train_dataset = Subset(dataset, train_normal+train_abnormal)
    test_dataset = Subset(dataset, test_abnormal+test_normal)

    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = GGNN_O(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = torch.nn.CrossEntropyLoss()

    c = utils.init_center_c(train_loader, model, device)

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


        total_loss = total_loss / total
        end_time = time.time()
        print('Epoch: %3d/%3d, Train Loss: %.5f, Time: %.5f' % (epoch+1, epochs, total_loss, end_time - start_time))

        # eval

    # test
    idx_label_score = []
    model.eval()
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

            # Save triples of (idx, label, score) in a list
            idx_label_score += list(zip(labels.cpu().data.numpy().tolist(),
                                        scores.cpu().data.numpy().tolist()))

    test_time = time.time() - start_time
    print('Testing time: %.3f' % test_time)

    test_scores = idx_label_score

    # Compute AUC
    labels, scores = zip(*idx_label_score)
    labels = np.array(labels)
    for label in np.nditer(labels, op_flags=['readwrite']):
        if label == 0:
            label[...] = 1
        elif label == 1:
            label[...] = 0
    scores = np.array(scores)

    test_auc = roc_auc_score(labels, scores)
    print('Test set AUC: {:.2f}%'.format(100. * test_auc))

    print('Finished testing.')





