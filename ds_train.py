import argparse
import torch
import torchvision
import deepspeed
from model import FashionModel, img_transform

## 命令行启动参数：deepspeed ds_train.py --epoch 2 --deepspeed --deepspeed_config ds_config.json
######### 训练逻辑，每个子进程都要执行完整代码，彼此共同协商训练 #########
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
parser.add_argument('--epoch', type=int, default=-1, help='epoch')
parser=deepspeed.add_config_arguments(parser)
cmd_args = parser.parse_args() #deepseed命令行参数

dataset = torchvision.datasets.FashionMNIST(root='./data', download=True, transform=img_transform) # 衣服数据集
dataload = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True) # 数据加载器,batch_size应该等于train_batch_size/gpu数量

model = FashionModel().cuda() # 原始模型
model, _, _, _ = deepspeed.initialize(args=cmd_args, model=model, model_parameters=model.parameters()) # deepspeed分布式模型,返回的model是分布式模型
loss_fn = torch.nn.CrossEntropyLoss().cuda() # 损失函数

for epoch in range(cmd_args.epoch): # 训练epoch
    for x, y in dataload: # 遍历数据集
        x, y = x.cuda(), y.cuda() # 数据移到GPU
        output = model(x) # 前向传播，x进到model就不知道在哪张卡上了，由deepspeed分配，多卡之间基于nvidia nccl通信
        loss = loss_fn(output, y) # 计算损失
        model.backward(loss) # 走deepspeed的反向传播
        model.step() # 更新参数
    print('epoch {} done'.format(epoch)) # 打印epoch
    model.save_checkpoint('./checkpoint') # 保存模型