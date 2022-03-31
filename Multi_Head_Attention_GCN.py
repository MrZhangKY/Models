import torch
import torch.nn as nn


class selfAttentionLayer(nn.Module):
    '''(batch, channelNum, dataInLength) -> (batch, channelNum, channelNum)'''
    def __init__(self, dataInLength, dataMiddleLength):
        '''
        dataInLength: 输入样本的长度
        dataMiddleLength： 处理过程中数据的中间长度，多大都可以，但是可能会影响模型性能
        '''
        super().__init__()
        self.layer1 = nn.Linear(dataInLength, dataMiddleLength)
        self.layer2 = nn.Linear(dataInLength, dataMiddleLength)
        
    def forward(self, dataIn):
        dataInExchanged1 = self.layer1(dataIn)
        dataInExchanged2 = torch.transpose(self.layer2(dataIn), 1, 2)
        dataOut = nn.Softmax(dim=2)(torch.matmul(dataInExchanged1, dataInExchanged2))  #转置矩阵乘积，softmax归一化，得到权重矩阵
        return dataOut


class multiheadAttentionLayer(nn.Module):
    '''(batch, channelNum, dataInLength) -> (batch, channelNum, channelNum)'''
    def __init__(self, dataInLength, dataMiddleLength, k=5, channelNum=6):
        '''
        dataInLength: 输入样本的长度
        dataMiddleLength： 处理过程中数据的中间长度，多大都可以，但是可能会影响模型性能
        k: 注意力机制的头数
        channelNum: 融合的通道数
        '''
        super().__init__()
        self.k = k
        self.attentionLayerList = nn.ModuleList()   #自动注册到主网络中
        for i in range(k):
            self.attentionLayerList.append(selfAttentionLayer(dataInLength, dataMiddleLength))
        self.linearExchangeLayer = nn.Linear(k*channelNum, channelNum)
        
    def forward(self, dataIn):
        attentionLayerOutList = []
        for i in range(self.k):
            attentionLayerOutList.append(self.attentionLayerList[i](dataIn))
        attentionLayerOut = nn.Softmax(dim=2)(self.linearExchangeLayer(torch.cat(attentionLayerOutList, dim=2)))
        return attentionLayerOut


class GraphConvolution(nn.Module):
    '''(batch, channelNum, dataInLength)&(batch, channelNum, channelNum)-> (batch, channelNum, dataOutLength)'''
    def __init__(self, dataInLength, dataOutLength, dropout, bias=True):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.Tensor(dataInLength, dataOutLength))
        nn.init.xavier_uniform_(self.weight)
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(dataOutLength))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
            
    def forward(self, dataIn, attentionMatrix):
        support = torch.matmul(self.dropout(dataIn), self.weight)
        output = torch.matmul(attentionMatrix, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class MHAGCN(nn.Module):
    def __init__(self, dataInLength, dataHiddenLength, dataOutLength, dataChannelNum, dropout=0.5):
        super().__init__()
        self.multiheadAttention_Layer = multiheadAttentionLayer(dataInLength, dataHiddenLength)
        self.first_layer = GraphConvolution(dataInLength, dataHiddenLength, dropout)
        self.batchNormLayer1 = nn.BatchNorm1d(dataChannelNum)
        self.second_layer = GraphConvolution(dataHiddenLength, dataOutLength, 0)
        self.batchNormLayer2 = nn.BatchNorm1d(dataChannelNum)
        self.relu = nn.PReLU()
    def forward(self, dataIn):
        attentionMatrix = self.multiheadAttention_Layer(dataIn)
        dataMiddle = self.batchNormLayer1(self.relu(self.first_layer(dataIn, attentionMatrix)))
        dataOut = self.batchNormLayer2(self.second_layer(dataMiddle, attentionMatrix))
        return dataOut
  

if __name__ == '__main__':

    '''1-test for selfAttentionLayer'''
    # dataIn = torch.randn(1, 6, 2048)
    # selfAttentionLayer1 = selfAttentionLayer(2048, 10)
    # print(selfAttentionLayer1(dataIn))
    
    '''2-test for multiheadAttentionLayer'''
    # dataIn = torch.randn(10, 6, 2048)
    # multiheadAttentionLayer1 = multiheadAttentionLayer(2048, 10)
    # print(multiheadAttentionLayer1(dataIn))
    # print(multiheadAttentionLayer1(dataIn).shape)
    # # for parameter in multiheadAttentionLayer1.parameters():
    # #     print(parameter.shape)
    # print(multiheadAttentionLayer1)
    
    '''3-test for GraphConvolution'''
    dataIn = torch.randn(10, 6, 2048)
    multiheadGNN = MHAGCN(2048, 1000, 500, 6)
    print(multiheadGNN(dataIn))
    print(multiheadGNN(dataIn).shape)
    print(multiheadGNN)
    for parameter in multiheadGNN.parameters():
        print(parameter.shape)
    