import torch
import torch.nn as nn


def myConv1d(kernel_size):
    return nn.Sequential(nn.Conv1d(1, 1, kernel_size), nn.Linear(1000-kernel_size+1, 100), nn.PReLU())


class DMNSubnet(nn.Module):
    def __init__(self, dataInLength, dropout=0.5):
        super().__init__()
        self.Layer1 = nn.Sequential(
            nn.Linear(dataInLength, 2000),
            nn.PReLU(),
            nn.Linear(2000, 1000),
            nn.PReLU(),
            nn.Dropout(dropout)
        )
        self.Layer2_1 = myConv1d(8)
        self.Layer2_2 = myConv1d(16)
        self.Layer2_3 = myConv1d(32)
        self.Layer2_4 = myConv1d(64)
        self.Layer2_5 = myConv1d(128)
        self.Layer3 = nn.Sequential(nn.Conv1d(5, 1, 1), nn.PReLU(), nn.Dropout(dropout))
        self.Layer4 = nn.Sequential(nn.Linear(100, 50), nn.PReLU())
        self.Layer5 = nn.Sequential(nn.Linear(50, 10), nn.PReLU())
    def forward(self, data):
        data1 = self.Layer1(data)
        data2_1 = self.Layer2_1(data1)
        data2_2 = self.Layer2_2(data1)
        data2_3 = self.Layer2_3(data1)
        data2_4 = self.Layer2_4(data1)
        data2_5 = self.Layer2_5(data1)
        data2 = torch.cat([data2_1, data2_2, data2_3, data2_4, data2_5], axis=1)
        data3 = self.Layer3(data2)
        data4 = self.Layer4(data3)
        data5 = self.Layer5(data4)
        return data5


class DMN(nn.Module):
    def __init__(self, dataInLength, dropout=0.5):
        super().__init__()
        self.subnet1 = self.subnet2 = DMNSubnet(dataInLength, dropout)
    def forward(self, dataSource, dataTarget):
        return 1-torch.mean(nn.CosineSimilarity(dim=2)(self.subnet1(dataSource), self.subnet2(dataTarget)))


if __name__ == '__main__':
    '''V1-test for DMNSubnet'''
    dataSource = torch.randn(10, 1, 500)
    DMNSubnet1 = DMNSubnet(500)
    print(DMNSubnet1(dataSource).shape)
    '''V2-test for DMN'''
    DMN1 = DMN(500)
    dataTarget = torch.randn(10, 1, 500)
    print(DMN1(dataSource, dataTarget))

