from torch import nn
# model 1 -> already trained ALL-ConvNets -> generate the high-level features of images

# model 2 -> randomly initialized ALL-ConvNets -> capture the low-level features of images, since the randomly initialized filters act as edge detectors


def conv_block(in_channels: int, out_channels: int, kernel_size: int, bn_momentum: float, bn_track_running_stats):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
        nn.BatchNorm2d(64, momentum=bn_momentum, track_running_stats=bn_track_running_stats),
        nn.ReLU()
    )


def pooling_block(kernel_size, out_channels, bn_momentum, bn_track_running_stats, pooling):
    return nn.Sequential(
        pooling(kernel_size=(2, 2)),
        nn.BatchNorm2d(out_channels, momentum=bn_momentum, track_running_stats=bn_track_running_stats)
    )


def mlp(in_dim, out_dim, bn_momentum, bn_track_running_stats):
    return nn.Sequential(
        nn.Linear(10, 10),
        nn.BatchNorm1d(10, momentum=bn_momentum,
                       track_running_stats=bn_track_running_stats),
        nn.ReLU()
    )


class MNISTNetwork(nn.Module):
    def __init__(self, config: dict):
        super(MNISTNetwork, self).__init__()
        self.config = config
        bn_track_running_stats = self.config["track_running_stats"]
        bn_momentum = 0.01
        self.net1 = nn.Sequential(
            conv_block(1, 64, 3, bn_momentum, bn_track_running_stats),
            conv_block(64, 64, 3, bn_momentum, bn_track_running_stats),
            conv_block(64, 64, 3, bn_momentum, bn_track_running_stats),
            pooling_block((2, 2), 64, bn_momentum, bn_track_running_stats, nn.MaxPool2d)
        )
        self.net2 = nn.Sequential(
            conv_block(64, 128, 3, bn_momentum, bn_track_running_stats),
            conv_block(128, 128, 3, bn_momentum, bn_track_running_stats),
            conv_block(128, 128, 3, bn_momentum, bn_track_running_stats),
            pooling_block((2, 2), 128, bn_momentum, bn_track_running_stats, nn.MaxPool2d)
        )
        self.net3 = nn.Sequential(
            conv_block(128, 10, 1, bn_momentum, bn_track_running_stats),
            pooling_block((2, 2), 128, bn_momentum, bn_track_running_stats, nn.AvgPool2d)
        )
        self.net4 = nn.Sequential(
            mlp(10, 10, bn_momentum, bn_track_running_stats),
            mlp(10, 10, bn_momentum, bn_track_running_stats)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_in):
        x = self.net1(x_in)
        x = self.net2(x)
        x = self.net3(x)
        x = self.net4(x)
        return self.softmax(x)
