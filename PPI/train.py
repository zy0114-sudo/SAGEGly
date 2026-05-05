import matplotlib.pyplot as plt          #########
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import download_url
import os
from torch_geometric.io import read_planetoid_data
from torch_geometric.datasets import Planetoid
import numpy as np
from torch_geometric.data import Data
from torch.nn import Linear
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv,SAGEConv,GATConv
import os.path as osp
from torch_geometric.nn import global_mean_pool
import scipy.sparse as sp
import torch_geometric.nn as pyg_nn
from torch_geometric.data import DataLoader
import warnings
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import matthews_corrcoef
warnings.filterwarnings("ignore", category=Warning)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from tqdm import tqdm
import torch.nn as nn
from config import config_runtime
from collections import Counter
#import wandb
#wandb.init(project="add_train_1")

#train
file='/PPI/train_npz'
#file1='/data4_large1/home_data/ccsun/scc_neuralnetwork/03_Antibody_affinity/database/try4-all-gab/pdb/test-npz-inter=0'

class MyOwnDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        

    # 返回数据集源文件名，告诉原始的数据集存放在哪个文件夹下面，如果数据集已经存放进去了，那么就会直接从raw文件夹中读取。
    @property
    def raw_file_names(self):
        # pass # 不能使用pass，会报join() argument must be str or bytes, not 'NoneType'错误
        return []

    # 首先寻找processed_paths[0]路径下的文件名也就是之前process方法保存的文件名
    @property
    def processed_file_names(self):
        return ['datas0.pt','datas1.pt']

    # 用于从网上下载数据集，下载原始数据到指定的文件夹下，自己的数据集可以跳过
    def download(self):
        pass
    def process(self):
        #
        data_list=[]
        for i in range(4852):  #5100就不行
            path_list=os.listdir(file)
            path_list.sort()
            npz=path_list[i]
            print(npz)
            pdb_name=npz[0:10]
            #pdb2=path_list[i]
            aa=os.path.join(file,npz)
    
            fileload=np.load(aa)  #读取npz文件
            key=list(fileload.keys()) #获取npz里面的键，由一个个字典组成
            #print(key)

            node=fileload['H'] #获取节点以及节点特征，数据类型为numpy.ndarray
            #node = torch.tensor(x) #数据类型为torch.Tensor
            #print(np.shape(node))

            adj=fileload['A1'] #data.edge_index节点和边的邻接矩阵
            #adj2=fileload['A2'] #data.edge_attr: 边属性
    

            #邻接矩阵转换成COO稀疏矩阵及转换  
            edge_index_temp = sp.coo_matrix(adj)  
            #print('edge_index_temp为：')
            #print(np.shape(edge_index_temp))
            tar=fileload['T']
    
    
            indices = np.vstack((edge_index_temp.row, edge_index_temp.col))
            edge_index = torch.LongTensor(indices)
            #print(np.shape(edge_index))
            #节点及节点特征数据转换
            x = node
            #print(x)
            #x = x.squeeze(0)
            x = torch.FloatTensor(x)
            #print(np.shape(x))
            #print(tar)
            a=tar
            #print(a)
            y=torch.LongTensor(a)
            #print(y)
            #edge_attr=adj2

            #构建数据集:为一张图，节点数量，节点特征，Coo稀疏矩阵的边(邻接矩阵)，边的特征矩阵,一个图一个标签
            data=Data(x=x, edge_index=edge_index,y=y)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            torch.save(data,osp.join(self.processed_dir,'datas{}.pt'.format(i)))
    
    def len(self):
        return 4852 #21746

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'datas{}.pt'.format(idx)))
        return data
    

    
    
dataset_train=MyOwnDataset('/PPI/train_pt')



#test
file1='/PPI/test_npz'

class MyOwnDataset2(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        

    # 返回数据集源文件名，告诉原始的数据集存放在哪个文件夹下面，如果数据集已经存放进去了，那么就会直接从raw文件夹中读取。
    @property
    def raw_file_names(self):
        # pass # 不能使用pass，会报join() argument must be str or bytes, not 'NoneType'错误
        return []

    # 首先寻找processed_paths[0]路径下的文件名也就是之前process方法保存的文件名
    @property
    def processed_file_names(self):
        return ['datas0.pt','datas1.pt']

    # 用于从网上下载数据集，下载原始数据到指定的文件夹下，自己的数据集可以跳过
    def download(self):
        pass
    def process(self):
        #
        data_list=[]
        for i in range(540):
            path_list=os.listdir(file1)
            path_list.sort()
            npz=path_list[i]
            #print(npz)
            #pdb2=path_list[i]
            aa=os.path.join(file1,npz)
    
            fileload=np.load(aa)  #读取npz文件
            key=list(fileload.keys()) #获取npz里面的键，由一个个字典组成
            #print(key)

            node=fileload['H'] #获取节点以及节点特征，数据类型为numpy.ndarray
            #node = torch.tensor(x) #数据类型为torch.Tensor
            #print(np.shape(node))

            adj=fileload['A1'] #data.edge_index节点和边的邻接矩阵
            #adj2=fileload['A2'] #data.edge_attr: 边属性
    

            #邻接矩阵转换成COO稀疏矩阵及转换  
            edge_index_temp = sp.coo_matrix(adj)  
            #print('edge_index_temp为：')
            #print(np.shape(edge_index_temp))
            tar=fileload['T']
    
    
            indices = np.vstack((edge_index_temp.row, edge_index_temp.col))
            edge_index = torch.LongTensor(indices)
            #print(np.shape(edge_index))
            #节点及节点特征数据转换
            x = node
            #print(x)
            #x = x.squeeze(0)
            x = torch.FloatTensor(x)
            #print(np.shape(x))
            #print(tar)
            a=tar
            y=torch.LongTensor(a)
            #print(y)
    
            #edge_attr=adj2

            #构建数据集:为一张图，节点数量，节点特征，Coo稀疏矩阵的边(邻接矩阵)，边的特征矩阵,一个图一个标签
            data=Data(x=x, edge_index=edge_index,y=y)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            torch.save(data,osp.join(self.processed_dir,'datas{}.pt'.format(i)))
    
    def len(self):
        return 540

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'datas{}.pt'.format(idx)))
        return data
    
dataset_test=MyOwnDataset2('/PPI/test_pt')


class ImprovedTripleGraphModel(nn.Module):
    def __init__(self, num_node_features, num_output_features=2, dropout_rate=0.3):
        super(ImprovedTripleGraphModel, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)

        # 定义三层GATConv用于B链图
        self.conv_1 = SAGEConv(num_node_features, 512)
        self.conv_2 = SAGEConv(512, 1024)
        self.conv_3 = SAGEConv(1024, 2)

        # 最终层（不应用sigmoid）
        self.fc1 = nn.Linear(256, 512)  # 3个64维的特征向量融合后的维度
        self.fc2 = nn.Linear(512, num_output_features)  # 输出层
        
        
    def forward(self, x,edge_index, batch):
        # 处理B链图
        
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv_1(x,edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv_2(x,edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv_3(x,edge_index)
        x = F.relu(x)
        return F.softmax(x, dim=1)


    
    



train_loader = DataLoader(dataset_train, batch_size=1, shuffle=True)  
#for i in train_ab_loader:
    #print(i.edge_index)
test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False)       




weight_new=torch.tensor([0.1, 0.6])
model = ImprovedTripleGraphModel(num_node_features=1310, num_output_features=2)  # Adjust dimensions as needed  
device = torch.device(config_runtime['device'])
model = model.to(config_runtime['device'])
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # 优化器，参数优化计算
criterion = torch.nn.CrossEntropyLoss(weight=weight_new)
criterion = criterion.to(device)
#wandb.watch(model, log="all")

def train():
    model.train() # 表示模型开始训练
    loss_all = 0
    # 一轮epoch优化的内容
    for data in train_loader: # 每次提取训练数据集一批20张data图片数据赋值给data
        # data是batch_size图片的大小
        #print(data.x)

        data=data.to(device)

        output = model(data.x, data.edge_index, data.batch) # 前向传播，把一批训练数据集导入模型并返回输出结果，输出结果的维度是[20,2]
        label = data.y # 20张图片数据的标签集合，维度是[20]
        label1 = data.y
        #label2 = data.y
        ##print(type(label1))
        #label = torch.stack((label1, label2), dim=0)  ## BCE的损失函数需要label和ouput的维度是一致的，所以在这里先拼接再转置，可看下面cell里的例子
        #label = label.t()
        #label=label.float()
        #print(type(label))
        #print(output)
        loss = criterion(output,label) # 损失函数计算，
        loss.backward() #反向传播
        loss_all += loss.item() # 将最后的损失值汇总
        optimizer.step() # 更新模型参数
        optimizer.zero_grad() # 梯度清零
    train_loss = (loss_all / len(dataset_train)) # 算出损失值或者错误率
 
    return train_loss
                #print(out)
            #print(np.shape(out))
            
            
# 测试
def evaluate(loader): # 构造训练函数计算acc
    model.eval()
    preds=[]
    tru=[]
    correct = 0
    for data in loader:
        # Iterate in batches over the training/test dataset.

        data=data.to(device)

        out = model(data.x, data.edge_index, data.batch)
        #print(out)
        pred = out.argmax(dim=1)  # 返回最大值对应的索引值
        #print('预测的是:',pred)
        aa=data.y
        for i,ii in zip(pred,aa):
            i=int(i)   #将预测值转换成整数
            ii=int(ii)  #将真实值转换成整数
            preds.append(ii)  #保存真实标签
            if i==ii: #如果预测值等于真实值
                tru.append(ii)
       # print('实际的是：',aa)
        correct=len(tru)
        all1=len(preds)
    return correct / all1


def evaluatee(loader): # 构造测试函数计算acc
    model.eval()
    preds=[]
    tru=[]
    correct = 0
    for data in loader:
        # Iterate in batches over the training/test dataset.

        data=data.to(device)

        out = model(data.x, data.edge_index, data.batch)
        #print(out)
        pred = out.argmax(dim=1)  # 返回最大值对应的索引值
        #print('预测的是:',pred)
        aa=data.y
        for i,ii in zip(pred,aa):
            i=int(i)  #将预测值转成整数
            ii=int(ii)  #将真实值转成整数
            preds.append(ii) #保存真实标签
            if i==ii:
                tru.append(ii)
        #print('实际的是：',aa)
        correct=len(tru)
        all1=len(preds)
        
    return correct / all1  # Derive ratio of correct predictions.

def evaluate1(loader): # 构造函数计算 train auc
    model.eval()
    prob_all=[]
    label_all=[]
    correct = 0
    for data in loader:
        # Iterate in batches over the training/test dataset.

        data=data.to(device)

        out = model(data.x, data.edge_index, data.batch)
        #print(out)
        aa=data.y.detach().cpu()
        #print(aa)
        prob_all.extend(out[:,1].detach().cpu().numpy()) #分数必须是具有较大标签的类的分数，通俗点理解:模型打分的第二列
        label_all.extend(aa)
        
    return roc_auc_score(label_all,prob_all)# Derive ratio of correct pre

def evaluate11(loader): # 构造test函数计算auc
    model.eval()
    prob_all=[]
    label_all=[]
    correct = 0
    for data in loader:
        # Iterate in batches over the training/test dataset.

        data=data.to(device)

        out = model(data.x, data.edge_index, data.batch)
        #print(out)
        aa=data.y.detach().cpu()
        #print(aa)
        prob_all.extend(out[:,1].detach().cpu().numpy()) #分数必须是具有较大标签的类的分数，通俗点理解:模型打分的第二列
        label_all.extend(aa)
        
    return roc_auc_score(label_all,prob_all)


def evaluate2(loader): # 构造测试函数计算精确率,召回率,f1_score   #
    model.eval()
    a1=[]
    a2=[]
    a3=[]
    a4=[]
    correct = 0
    for data in loader:
        # Iterate in batches over the training/test dataset.

        data=data.to(device)

        out = model(data.x, data.edge_index, data.batch)
        #print(out)
        pred = out.argmax(dim=1).detach().cpu()  # 返回最大值对应的索引值
        #print(pred)
        a1.extend(pred)
    
        aa=data.y.detach().cpu()
        a2.extend(aa)
    #print(a2)
    for i,ii in zip(a1,a2):
        i=int(i)
        ii=int(ii)
        #print(type(i))
        a3.append(i)
        a4.append(ii)
    #print(a3)
    #print(a4)
    return precision_score(a4,a3), recall_score(a4,a3), matthews_corrcoef(a4,a3), f1_score(a4,a3)




#开始训练
train_loss_all = []
#valid_loss_all = []

train_acc_all=[]
train_auc_all=[]

#valid_acc_all=[]
test_acc_all=[]
test_auc_all=[]

test_precision_all=[]
test_recall_all=[]
test_mcc_all=[]
test_f1_All=[]

train_precision_all=[]
train_recall_all=[]
train_mcc_all=[]
train_f1_All=[]

print(config_runtime)
for epoch in range(config_runtime['num_epochs']):

    train_loss = train()	
    train_acc = evaluate(train_loader)
    train_auc = evaluate1(train_loader)
    #a3=evaluate2(train_loader)
    #print('a3',a3)
    #train_recision = a3[0]
   # train_recall = a3[1]
    #print('===================================================================================')
    test_acc = evaluatee(test_loader)
    test_auc = evaluate11(test_loader)
    
    a4=evaluate2(test_loader)
    
    a5=evaluate2(train_loader)
    #print(a4)
    test_precision = a4[0]
    test_recall = a4[1]
    test_mcc = a4[2]
    test_f1=a4[3]
    
    train_precision = a5[0]
    train_recall = a5[1]
    train_mcc = a5[2]
    train_f1=a5[3]
    
    
    train_loss_all.append(train_loss)
    train_acc_all.append(train_acc)
    train_auc_all.append(train_auc)
    test_acc_all.append(test_acc)
    test_auc_all.append(test_auc)
    test_precision_all.append(test_precision)
    test_recall_all.append(test_recall)
    test_mcc_all.append(test_mcc)
    test_f1_All.append(test_f1)
    
    train_precision_all.append(train_precision)
    train_recall_all.append(train_recall)
    train_mcc_all.append(train_mcc)
    train_f1_All.append(train_f1)
    '''
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_auc": train_auc,
        "train_precision": train_precision,
        "train_recall": train_recall,
        "train_mcc": train_mcc,
        "train_f1": train_f1,
        "test_acc": test_acc,
        "test_auc": test_auc,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_mcc": test_mcc,
        "test_f1": test_f1,
    }, step=epoch)
    '''
    output_path='/PPI/model/'
    #if test_acc >= 0.7:
    #    print('测试集的准确率为%s，在第%d中'%(t_acc,epoch))
    torch.save(model.state_dict(), "/PPI/model/%d_model.pt"%epoch)
    #wandb.save("/data5_large/home/yanzh/deeplearn/graph_residue/my_data/protein_new_3222/dataset_shuanglian_buchong/add_train_npz/model/model1/%d_model.pt"%epoch)
    print(f'Epoch: {epoch}, train_loss: {train_loss:.4f},train_Acc: {train_acc:.4f},train_AUC: {train_auc:.4f},train_prescision:{train_precision:.4f},train_recall:{train_recall:.4f},train_mcc:{train_mcc:.4f},train_f1:{train_f1:.4f}')
    print(f'test_Acc: {test_acc:.4f},test_AUC: {test_auc:.4f},test_prescision:{test_precision:.4f} ,test_recall:{test_recall:.4f},test_mcc:{test_mcc:.4f},test_f1:{test_f1:.4f}')
    print('===========================================')