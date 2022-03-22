import torch
import torch_geometric
import torch.nn as nn
from torch_geometric.nn import GatedGraphConv, GlobalAttention, GCNConv
from torch.nn import Embedding


class GAE(nn.Module):
    def __init__(self, num_layers=5, hidden_dim=150, alpha=0.5):
        super(GAE, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.alpha = alpha

        self.encoder = GatedGraphConv(out_channels=hidden_dim, num_layers=3)
        self.decoder = GatedGraphConv(out_channels=hidden_dim, num_layers=3)

    def forward(self, data):
        x = self.encoder(data.x, data.edge_index, data.edge_attr)
        recon_x = self.decoder(x, data.edge_index, data.edge_attr)
        return x, recon_x

    def loss_gae(self, x, recon_x, embed_x, edge_index, batch, edge_attr):
        ratio = [self.alpha, 1 - self.alpha]

        num_graph = batch.max() + 1
        num_node, _ = x.size()

        adj = torch.zeros((num_node, num_node)).to(x.device)
        adj[edge_index[0], edge_index[1]] = torch.squeeze(edge_attr)
        adj[list(range(num_node)), list(range(num_node))] = -1

        sigmoid = nn.Sigmoid()
        structure_loss_func = nn.MSELoss()
        attr_loss_func = nn.MSELoss()

        structure_loss = 0
        attr_loss = 0
        scores = []
        for ng in range(num_graph):
            xg = x[batch==ng]
            recon_xg = recon_x[batch==ng]
            embed_xg = embed_x[batch==ng]

            adj_pred = torch.einsum('ik,jk->ij', xg, xg)
            adj_pred = sigmoid(adj_pred)
            adj_true = adj[batch==ng, :][:, batch==ng].to(x.device)

            graph_structure_loss = structure_loss_func(adj_pred[adj_true>-1], adj_true[adj_true>-1])
            graph_attr_loss = attr_loss_func(recon_xg, embed_xg)

            scores.append(ratio[0] * graph_attr_loss + ratio[1] * graph_structure_loss)

            structure_loss += graph_structure_loss
            attr_loss += graph_attr_loss

        structure_loss /= num_graph
        attr_loss /= num_graph
        loss = ratio[0] * attr_loss + ratio[1] * structure_loss
        return loss, torch.Tensor(scores)

    def caculate_ad_score(self, x, recon_x, embed_x, edge_index, batch, edge_attr, trace_ids):
        ratio = [self.alpha, 1 - self.alpha]

        num_graph = batch.max() + 1
        num_node, _ = x.size()

        adj = torch.zeros((num_node, num_node)).to(x.device)
        adj[edge_index[0], edge_index[1]] = torch.squeeze(edge_attr)
        adj[list(range(num_node)), list(range(num_node))] = -1

        sigmoid = nn.Sigmoid()
        structure_loss_func = nn.MSELoss()
        attr_loss_func = nn.MSELoss()
        node_attr_loss_func = nn.MSELoss()
        node_edge_loss_func = nn.MSELoss()

        structure_loss = 0
        attr_loss = 0
        scores = []
        attr_scores = []
        structure_scores = []
        nodes_scores = []
        nodes_index = []
        trace_id_result = []
        nodes_attr_scores = []
        nodes_structure_scores = []

        max_node_loss_ng_index = -1
        now_max_node_loss = -1

        for ng in range(num_graph):
            xg = x[batch==ng]
            recon_xg = recon_x[batch==ng]
            embed_xg = embed_x[batch==ng]

            adj_pred = torch.einsum('ik,jk->ij', xg, xg)
            adj_pred = sigmoid(adj_pred)
            adj_true = adj[batch==ng, :][:, batch==ng].to(x.device)

            graph_structure_loss = structure_loss_func(adj_pred[adj_true>-1], adj_true[adj_true>-1])
            graph_attr_loss = attr_loss_func(recon_xg, embed_xg)

            node_loss = []
            nodes_attr_loss = []
            nodes_structure_scores = []
            for i in range(xg.size()[0]):
                node_edge_exit_pred = adj_pred[i, :]
                node_edge_entry_pred = adj_pred[:, i]
                node_edge_exit_true = adj_true[i, :]
                node_edge_entry_true = adj_true[:, i]

                node_recon = recon_xg[i, :]
                node_embed = embed_xg[i, :]

                node_attr_loss = node_attr_loss_func(node_recon, node_embed)
                node_edge_exit_loss = node_edge_loss_func(node_edge_exit_pred[node_edge_exit_true>-1],
                                                          node_edge_exit_true[node_edge_exit_true>-1])
                node_edge_entry_loss = node_edge_loss_func(node_edge_entry_pred[node_edge_entry_true>-1],
                                                           node_edge_entry_true[node_edge_entry_true>-1])

                nodes_attr_loss.append(node_attr_loss)
                nodes_structure_scores.append(node_edge_entry_loss + node_edge_exit_loss)
                node_loss.append(ratio[0] * node_attr_loss + ratio[1] * (node_edge_exit_loss + node_edge_entry_loss))

            max_node_loss = max(node_loss)
            max_node_index = node_loss.index(max_node_loss)

            if max_node_loss > now_max_node_loss:
                now_max_node_loss = max_node_loss
                max_node_loss_ng_index = ng

            trace_id_result.append(trace_ids[max_node_loss_ng_index])

            nodes_scores.append(max_node_loss)
            nodes_index.append(max_node_index)
            nodes_attr_scores.append(max(nodes_attr_loss))
            nodes_structure_scores.append(max(nodes_structure_scores))

            scores.append(ratio[0] * graph_attr_loss + ratio[1] * graph_structure_loss)
            attr_scores.append(graph_attr_loss)
            structure_scores.append(graph_structure_loss)

            structure_loss += graph_structure_loss
            attr_loss += graph_attr_loss

        # print(f"max_node_loss_ng_index {max_node_loss_ng_index}  trace_id {trace_ids[max_node_loss_ng_index]}")

        return torch.Tensor(scores), torch.Tensor(nodes_scores), torch.Tensor(nodes_index), \
               trace_id_result, torch.Tensor(attr_scores), \
               torch.Tensor(structure_scores), torch.Tensor(nodes_attr_scores), torch.Tensor(nodes_structure_scores)




class GAE_GCN(nn.Module):
    def __init__(self, num_layers=5, hidden_dim=150, alpha=0.5):
        super(GAE_GCN, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.alpha = alpha

        self.conv1 = GCNConv(hidden_dim, 200)
        self.conv2 = GCNConv(200, 200)
        self.conv3 = GCNConv(200, 150)

        self.decoder = GCNConv(150, hidden_dim)

    def forward(self, data):
        edge_attr = torch.squeeze(data.edge_attr)
        x = self.conv1(data.x, data.edge_index, edge_attr).tanh()
        x = self.conv2(x, data.edge_index, edge_attr).tanh()
        x = self.conv3(x, data.edge_index, edge_attr).tanh()

        recon_x = self.decoder(x, data.edge_index, edge_attr).tanh()
        return x, recon_x

    def loss_gae(self, x, recon_x, embed_x, edge_index, batch, edge_attr):
        ratio = [self.alpha, 1 - self.alpha]

        num_graph = batch.max() + 1
        num_node, _ = x.size()

        adj = torch.zeros((num_node, num_node)).to(x.device)
        adj[edge_index[0], edge_index[1]] = torch.squeeze(edge_attr)
        adj[list(range(num_node)), list(range(num_node))] = -1

        sigmoid = nn.Sigmoid()
        structure_loss_func = nn.MSELoss()
        attr_loss_func = nn.MSELoss()

        structure_loss = 0
        attr_loss = 0
        scores = []
        for ng in range(num_graph):
            xg = x[batch==ng]
            recon_xg = recon_x[batch==ng]
            embed_xg = embed_x[batch==ng]

            adj_pred = torch.einsum('ik,jk->ij', xg, xg)
            adj_pred = sigmoid(adj_pred)
            adj_true = adj[batch==ng, :][:, batch==ng].to(x.device)

            graph_structure_loss = structure_loss_func(adj_pred[adj_true>-1], adj_true[adj_true>-1])
            graph_attr_loss = attr_loss_func(recon_xg, embed_xg)

            scores.append(ratio[0] * graph_attr_loss + ratio[1] * graph_structure_loss)

            structure_loss += graph_structure_loss
            attr_loss += graph_attr_loss

        structure_loss /= num_graph
        attr_loss /= num_graph
        loss = ratio[0] * attr_loss + ratio[1] * structure_loss
        return loss, torch.Tensor(scores)

    def caculate_ad_score(self, x, recon_x, embed_x, edge_index, batch, edge_attr, trace_ids):
        ratio = [self.alpha, 1 - self.alpha]

        num_graph = batch.max() + 1
        num_node, _ = x.size()

        adj = torch.zeros((num_node, num_node)).to(x.device)
        adj[edge_index[0], edge_index[1]] = torch.squeeze(edge_attr)
        adj[list(range(num_node)), list(range(num_node))] = -1

        sigmoid = nn.Sigmoid()
        structure_loss_func = nn.MSELoss()
        attr_loss_func = nn.MSELoss()
        node_attr_loss_func = nn.MSELoss()
        node_edge_loss_func = nn.MSELoss()

        structure_loss = 0
        attr_loss = 0
        scores = []
        attr_scores = []
        structure_scores = []
        nodes_scores = []
        nodes_index = []
        trace_id_result = []
        nodes_attr_scores = []
        nodes_structure_scores = []

        max_node_loss_ng_index = -1
        now_max_node_loss = -1

        for ng in range(num_graph):
            xg = x[batch==ng]
            recon_xg = recon_x[batch==ng]
            embed_xg = embed_x[batch==ng]

            adj_pred = torch.einsum('ik,jk->ij', xg, xg)
            adj_pred = sigmoid(adj_pred)
            adj_true = adj[batch==ng, :][:, batch==ng].to(x.device)

            graph_structure_loss = structure_loss_func(adj_pred[adj_true>-1], adj_true[adj_true>-1])
            graph_attr_loss = attr_loss_func(recon_xg, embed_xg)

            node_loss = []
            nodes_attr_loss = []
            nodes_structure_scores = []
            for i in range(xg.size()[0]):
                node_edge_exit_pred = adj_pred[i, :]
                node_edge_entry_pred = adj_pred[:, i]
                node_edge_exit_true = adj_true[i, :]
                node_edge_entry_true = adj_true[:, i]

                node_recon = recon_xg[i, :]
                node_embed = embed_xg[i, :]

                node_attr_loss = node_attr_loss_func(node_recon, node_embed)
                node_edge_exit_loss = node_edge_loss_func(node_edge_exit_pred[node_edge_exit_true>-1],
                                                          node_edge_exit_true[node_edge_exit_true>-1])
                node_edge_entry_loss = node_edge_loss_func(node_edge_entry_pred[node_edge_entry_true>-1],
                                                           node_edge_entry_true[node_edge_entry_true>-1])

                nodes_attr_loss.append(node_attr_loss)
                nodes_structure_scores.append(node_edge_entry_loss + node_edge_exit_loss)
                node_loss.append(ratio[0] * node_attr_loss + ratio[1] * (node_edge_exit_loss + node_edge_entry_loss))

            max_node_loss = max(node_loss)
            max_node_index = node_loss.index(max_node_loss)

            if max_node_loss > now_max_node_loss:
                now_max_node_loss = max_node_loss
                max_node_loss_ng_index = ng

            trace_id_result.append(trace_ids[max_node_loss_ng_index])

            nodes_scores.append(max_node_loss)
            nodes_index.append(max_node_index)
            nodes_attr_scores.append(max(nodes_attr_loss))
            nodes_structure_scores.append(max(nodes_structure_scores))

            scores.append(ratio[0] * graph_attr_loss + ratio[1] * graph_structure_loss)
            attr_scores.append(graph_attr_loss)
            structure_scores.append(graph_structure_loss)

            structure_loss += graph_structure_loss
            attr_loss += graph_attr_loss

        # print(f"max_node_loss_ng_index {max_node_loss_ng_index}  trace_id {trace_ids[max_node_loss_ng_index]}")

        return torch.Tensor(scores), torch.Tensor(nodes_scores), torch.Tensor(nodes_index), \
               trace_id_result, torch.Tensor(attr_scores), \
               torch.Tensor(structure_scores), torch.Tensor(nodes_attr_scores), torch.Tensor(nodes_structure_scores)
