import numpy as np
from torch.nn import CrossEntropyLoss, BCELoss
from torch import nn
from torch.optim import SGD, RMSprop, Adam
import torch
from cos_dist import get_cos_dist_matrix
from metric import NMI, ARI, ACC


def check_shape(x):
    import torch
    import numpy as np
    import tensorflow as tf

    if torch.is_tensor(x):
        print(x.size())
    elif isinstance(x, np.ndarray):
        print(x.shape)
    elif isinstance(x, type([])):
        print(np.array(x).shape)
    elif tf.is_tensor(x):
        print(tf.shape(x))


class Trainer(object):
    def __init__(
            self,
            model: nn.Module,
            epoch: int,
            device: str
    ):
        super().__init__()
        self.model = model().to(device)
        self.epoch = epoch
        self.upper_thr = 0.99
        self.lower_thr = 0.75
        self.eta = (self.upper_thr - self.lower_thr) / self.epoch
        self.device = device

    def train(self, data):
        # criterion = torch.nn.MSELoss()
        criterion = torch.nn.CrossEntropyLoss()
        # criterion = torch.nn.BCELoss()

        optimizer = RMSprop(self.model.parameters(), lr=1e-3, momentum=0.9)

        while self.upper_thr > self.lower_thr:
            self.model.train()

            for epoch in range(self.epoch):
                for idx, (x, _) in enumerate(data):
                    x = x.view(-1, 1, 28, 28).to(self.device)
                    pred = self.model(x)
                    cos_dist = get_cos_dist_matrix(pred).to(self.device)

                    r_u = torch.where(cos_dist >= self.upper_thr, 1.0, 0.0).to(self.device)
                    r_l = torch.where(cos_dist < self.lower_thr, 1.0, 0.0).to(self.device)
                    r_label = torch.where(
                        cos_dist >= (self.upper_thr + self.lower_thr) / 2, 1.0, 0.0
                    ).to(self.device)
                    loss = torch.sum(
                        (r_u + r_l) * criterion(cos_dist, r_label)
                    ) / torch.sum(r_u + r_l).to(self.device)

                    # r_label = torch.where(cos_dist >= self.upper_thr, 1.0, 0.0)
                    # loss = criterion(cos_dist, r_label)
                    if not (idx % 100):
                        print(f'Epoch: {epoch}, Iteration:{idx}, loss: {loss.item()}')
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            self.model.eval()
            y_pred = []
            y_true = []

            for idx, (x, y) in enumerate(data):
                x = x.view(-1, 1, 28, 28).to(self.device)
                pred = self.model(x)

                y_pred.append(torch.argmax(pred, 1).detach().cpu().numpy())
                y_true.append(y.numpy())

                if idx == 10:
                    break

            pre_y = np.concatenate(y_pred, 0)
            tru_y = np.concatenate(y_true, 0)

            print(f'ACC: {ACC(tru_y, pre_y)}, NMI: {NMI(tru_y, pre_y)}, ARI: {ARI(tru_y, pre_y)}')

            self.upper_thr -= self.eta
            self.lower_thr += self.eta
