import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam, RMSprop

from cos_dist import get_cos_dist_matrix
from metric import ACC, ARI, NMI


class Trainer(object):
    def __init__(self, model: nn.Module, epoch: int, device: str):
        super().__init__()
        self.model = model().to(device)
        self.epoch = epoch
        self.upper_thr = 0.99
        self.lower_thr = 0.75
        self.eta = (self.upper_thr - self.lower_thr) / self.epoch
        self.device = device

    def train(self, data):
        criterion = torch.nn.MSELoss(reduction="sum")
        # criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        optimizer = RMSprop(self.model.parameters(), lr=1e-3, momentum=0.9)

        while self.upper_thr > self.lower_thr:
            self.model.train()

            for epoch in range(self.epoch):
                for idx, (x, _) in enumerate(data):
                    x = x.view(-1, 1, 28, 28).to(self.device)
                    pred = self.model(x)
                    dist = get_cos_dist_matrix(pred).to(self.device)
                    r_u = torch.where(dist >= self.upper_thr, 1.0, 0.0)
                    r_l = torch.where(dist < self.lower_thr, 1.0, 0.0)
                    r_label = torch.where(
                        dist >= (self.upper_thr + self.lower_thr) / 2, 1.0, 0.0
                    )
                    loss = torch.sum(
                        (r_u + r_l) * criterion(dist, r_label)
                    ) / torch.sum(r_u + r_l)
                    if not (idx % 100):
                        print(f"Epoch: {epoch}, Iteration:{idx}, loss: {loss.item()}")
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

            print(
                f"ACC: {ACC(tru_y, pre_y)}, NMI: {NMI(tru_y, pre_y)}, ARI: {ARI(tru_y, pre_y)}"
            )

            self.upper_thr -= self.eta
            self.lower_thr += self.eta
