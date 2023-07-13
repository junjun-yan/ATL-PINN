import torch
import torch.nn as nn
import numpy as np
from utilities import wn_linear

torch.manual_seed(1234)
np.random.seed(1234)


class Expert(nn.Module):
    def __init__(self, expert_layers, device):
        super().__init__()

        self.network = nn.Sequential().to(device)
        for i in range(0, len(expert_layers) - 1):
            self.network.add_module('dense_%d' % i, wn_linear(expert_layers[i], expert_layers[i + 1]))
            self.network.add_module('tanh_%d' % i, nn.Tanh())

    def forward(self, X):
        # data standrad
        feature = self.network(X)
        return feature


class Tower(nn.Module):
    def __init__(self, tower_layers, device):
        super().__init__()

        self.network = nn.Sequential().to(device)
        for i in range(0, len(tower_layers) - 1):
            self.network.add_module('dense_%d' % i, wn_linear(tower_layers[i], tower_layers[i + 1]))
            if i < len(tower_layers) - 2:
                self.network.add_module('tanh %d' % i, nn.Tanh())

    def forward(self, X):
        # data standrad
        feature = self.network(X)
        return feature


class Gate(nn.Module):
    def __init__(self, gate_layers, output_shape, device):
        super().__init__()

        self.pooling = nn.AvgPool1d(gate_layers)
        self.network = nn.Sequential().to(device)
        self.network.add_module('dense_1', wn_linear(output_shape, gate_layers))
        self.network.add_module('tanh', nn.Tanh())
        self.network.add_module('dense_2', wn_linear(gate_layers, output_shape))
        self.network.add_module('softmax', nn.Softmax(dim=1))

    def forward(self, features):
        # data standrad
        input = self.pooling(features)
        attention = self.network(input)
        return attention


class NeuralNetwork_PINN(nn.Module):
    def __init__(self, X, layers, device, method='nosorp'):
        '''
        :param X:       total input data for standardization
        :param layers:  [input dim] + number of layers * [number of cells] + [output dim]
        :param device:  device
        :param method:  if mehtod == sorption, then add a ReLU after output
        '''
        super().__init__()

        # initial mean and std
        self.X_mean = X.mean(0, keepdims=True).to(device)
        self.X_std = X.std(0, keepdims=True).to(device)

        # build network
        self.fnn = nn.Sequential().to(device)
        for i in range(0, len(layers) - 1):
            self.fnn.add_module('dense_%d' % i, wn_linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.fnn.add_module('tanh_%d' % i, nn.Tanh())

        if method == 'sorption':
            self.fnn.add_module('ReLU', nn.ReLU())

    def forward(self, X):
        # standardizing
        H = (X - self.X_mean) / self.X_std
        out = self.fnn(H)
        return out


class NeuralNetwork_Hard(nn.Module):
    def __init__(self, X, shared_layers, special_layers, task_number, device, method='no sorp'):
        '''

        :param X:
        :param shared_layers:
        :param special_layers:
        :param task_number:
        :param device:
        :param method:
        '''
        super().__init__()
        self.task_number = task_number

        # initial mean and std
        self.X_mean = X.mean(0).to(device)
        self.X_std = X.std(0).to(device)

        # shared network
        self.shared_network = nn.Sequential().to(device)
        for i in range(0, len(shared_layers) - 1):
            self.shared_network.add_module('dense_%d' % i, wn_linear(shared_layers[i], shared_layers[i + 1]))
            self.shared_network.add_module('tanh_%d' % i, nn.Tanh())

        # special network for different tasks
        self.special_networks = nn.ModuleList().to(device)
        for task in range(task_number):
            special_network = nn.Sequential()
            for i in range(len(special_layers) - 1):
                special_network.add_module('dense_%d' % i, wn_linear(special_layers[i], special_layers[i + 1]))
                if i < len(special_layers) - 2:
                    special_network.add_module('tanh %d' % i, nn.Tanh())
            if method == 'sorption':
                special_network.add_module('relu', nn.ReLU())
            self.special_networks.append(special_network)

    def forward(self, X):
        # data standrad
        H = (X - self.X_mean) / self.X_std
        feature = self.shared_network(H)
        out = torch.stack([self.special_networks[task](feature) for task in range(self.task_number)])
        return out

    def get_shared(self):
        return self.shared_network.parameters()


class NeuralNetwork_Soft_Add(nn.Module):
    def __init__(self, X, shared_layers, expert_layers, tower_layers, task_number, device):
        super().__init__()
        self.task_number = task_number

        # initial mean and std
        self.X_mean = X.mean(0).to(device)
        self.X_std = X.std(0).to(device)

        # shared expert
        self.shared = Expert(shared_layers, device)

        # special experts
        self.experts = nn.ModuleList()
        [self.experts.append(Expert(expert_layers, device)) for _ in range(task_number)]

        # towers
        self.towers = nn.ModuleList()
        [self.towers.append(Tower(tower_layers, device)) for _ in range(task_number)]

    def forward(self, X):
        # data standrad
        H = (X - self.X_mean) / self.X_std

        shared_feature = self.shared(H)
        expert_feature = [self.experts[task](H) for task in range(self.task_number)]
        feature = [shared_feature + expert_feature[task] for task in range(self.task_number)]
        out = torch.stack([self.towers[task](feature[task]) for task in range(self.task_number)])

        return out

    def get_shared(self):
        return self.shared.parameters()


class NeuralNetwork_Soft_Attention(nn.Module):
    def __init__(self, X, expert_layers, tower_layers, task_number, device):
        super().__init__()
        self.task_number = task_number

        # initial mean and std
        self.X_mean = X.mean(0).to(device)
        self.X_std = X.std(0).to(device)

        # special experts
        self.experts = nn.ModuleList()
        [self.experts.append(Expert(expert_layers, device)) for _ in range(task_number + 1)]

        # attention layer
        self.attention = Gate(tower_layers[0], task_number, device)

        # towers
        self.towers = nn.ModuleList()
        [self.towers.append(Tower(tower_layers, device)) for _ in range(task_number)]

    def forward(self, X):
        # data standrad
        H = (X - self.X_mean) / self.X_std

        features = torch.stack([self.experts[task](H) for task in range(self.task_number + 1)])
        attention = torch.unsqueeze(self.attention(features), -1)
        features = torch.permute(features, (1, 2, 0))
        features = torch.squeeze(torch.matmul(features, attention))
        out = torch.stack([self.towers[task](features) for task in range(self.task_number)])

        return out

    def get_shared(self):
        return self.experts.parameters()


class NeuralNetwork_Soft_Concat(nn.Module):
    def __init__(self, X, shared_layers, expert_layers, tower_layers, task_number, device):
        super().__init__()
        self.task_number = task_number

        # initial mean and std
        self.X_mean = X.mean(0).to(device)
        self.X_std = X.std(0).to(device)

        # shared expert
        self.shared = Expert(shared_layers, device)

        # special experts
        self.experts = nn.ModuleList()
        [self.experts.append(Expert(expert_layers, device)) for _ in range(task_number)]

        # towers
        self.towers = nn.ModuleList()
        [self.towers.append(Tower(tower_layers, device)) for _ in range(task_number)]

    def forward(self, X):
        # data standrad
        H = (X - self.X_mean) / self.X_std

        shared_feature = self.shared(H)
        expert_feature = [self.experts[task](H) for task in range(self.task_number)]
        feature = [torch.concat((shared_feature, expert_feature[task]), 1) for task in range(self.task_number)]
        out = torch.stack([self.towers[task](feature[task]) for task in range(self.task_number)])

        return out

    def get_shared(self):
        return self.shared.parameters()


class NeuralNetwork_Soft_MMOE(nn.Module):
    def __init__(self, X, expert_layers, tower_layers, task_number, device):
        super().__init__()
        self.task_number = task_number

        # initial mean and std
        self.X_mean = X.mean(0).to(device)
        self.X_std = X.std(0).to(device)

        # special experts
        self.experts = nn.ModuleList()
        [self.experts.append(Expert(expert_layers, device)) for _ in range(task_number + 1)]

        # attention layer
        self.attentions = nn.ModuleList()
        [self.attentions.append(Gate(tower_layers[0], task_number + 1, device)) for _ in range(task_number)]

        # towers
        self.towers = nn.ModuleList()
        [self.towers.append(Tower(tower_layers, device)) for _ in range(task_number)]

    def forward(self, X):
        # data standrad
        H = (X - self.X_mean) / self.X_std

        features_input = [self.experts[task](H) for task in range(self.task_number + 1)]
        features_concat = torch.concat(features_input, 1)
        features_stack = torch.stack(features_input, 2)

        out = []
        for task in range(self.task_number):
            attention = self.attentions[task](features_concat).unsqueeze(1).expand(features_stack.shape)
            features_output = torch.sum(attention * features_stack, 2)
            out.append(self.towers[task](features_output))

        return torch.stack(out)

    def get_shared(self):
        return self.experts.parameters()


class NeuralNetwork_Soft_PLE(nn.Module):
    def __init__(self, X, shared_layers, expert_layers, tower_layers, task_number, device):
        super().__init__()
        self.task_number = task_number

        # initial mean and std
        self.X_mean = X.mean(0).to(device)
        self.X_std = X.std(0).to(device)

        # shared expert
        self.shared = Expert(shared_layers, device)

        # special experts
        self.experts = nn.ModuleList()
        [self.experts.append(Expert(expert_layers, device)) for _ in range(task_number)]

        # attention layer
        self.attentions = nn.ModuleList()
        [self.attentions.append(Gate(tower_layers[0], task_number, device)) for _ in range(task_number)]

        # towers
        self.towers = nn.ModuleList()
        [self.towers.append(Tower(tower_layers, device)) for _ in range(task_number)]

    def forward(self, X):
        # data standrad
        H = (X - self.X_mean) / self.X_std

        feature_shared = self.shared(H)

        out = []
        for task in range(self.task_number):
            features_expert = self.experts[task](H)
            # attention
            features_concat = torch.concat([feature_shared, features_expert], 1)
            features_stack = torch.stack([feature_shared, features_expert], 2)
            attention = self.attentions[task](features_concat).unsqueeze(1).expand(features_stack.shape)

            features_output = torch.sum(attention * features_stack, 2)
            out.append(self.towers[task](features_output))

        return torch.stack(out)

    def get_shared(self):
        return self.experts.parameters()
