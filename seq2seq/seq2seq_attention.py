# %%
# 代码由 Tae Hwan Jung @graykode 提供
# 参考：https://github.com/hunkim/PyTorchZeroToAll/blob/master/14_2_seq2seq_att.py

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as tud
import deepspeed

# S: 开始解码输入的符号
# E: 开始解码输出的符号
# P: 如果当前批次数据大小小于时间步长，则用该符号填充空序列

class MyData(tud.Dataset):
    def __init__(self, sentences, n_step, n_hidden):
        self.sentences = sentences
        word_list = " ".join(self.sentences).split()
        self.word_list = list(set(word_list))
        self.word_dict = {w: i for i, w in enumerate(self.word_list)}
        self.number_dict = {i: w for i, w in enumerate(self.word_list)}
        self.n_class = len(self.word_dict)  # 词汇表大小
        self.n_hidden = n_hidden  # 单个单元格中的隐藏单元数
        self.n_step = n_step  # 单元格数量（步长）

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

    def make_batch(self):
        input_batch = [np.eye(self.n_class)[[self.word_dict[n] for n in self.sentences[0].split()]]]
        output_batch = [np.eye(self.n_class)[[self.word_dict[n] for n in self.sentences[1].split()]]]
        target_batch = [[self.word_dict[n] for n in self.sentences[2].split()]]

        # 转换为张量
        return torch.FloatTensor(input_batch), torch.FloatTensor(output_batch), torch.LongTensor(target_batch)
    

class Attention(nn.Module):
    def __init__(self, n_class, n_hidden):
        super(Attention, self).__init__()
        self.enc_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)
        self.dec_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)

        # 用于注意力的线性层
        self.n_class = n_class
        self.attn = nn.Linear(n_hidden, n_hidden)
        self.out = nn.Linear(n_hidden * 2, self.n_class)

    def forward(self, enc_inputs, hidden, dec_inputs):
        enc_inputs = enc_inputs.transpose(0, 1)  # enc_inputs: [n_step(=n_step, 时间步长), batch_size, n_class]
        dec_inputs = dec_inputs.transpose(0, 1)  # dec_inputs: [n_step(=n_step, 时间步长), batch_size, n_class]

        # enc_outputs : [n_step, batch_size, num_directions(=1) * n_hidden], 矩阵 F
        # enc_hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        enc_outputs, enc_hidden = self.enc_cell(enc_inputs, hidden)

        trained_attn = []
        hidden = enc_hidden
        n_step = len(dec_inputs)
        model = torch.empty([n_step, 1, self.n_class])

        for i in range(n_step):  # 每个时间步
            # dec_output : [n_step(=1), batch_size(=1), num_directions(=1) * n_hidden]
            # hidden : [num_layers(=1) * num_directions(=1), batch_size(=1), n_hidden]
            dec_output, hidden = self.dec_cell(dec_inputs[i].unsqueeze(0), hidden)
            attn_weights = self.get_att_weight(dec_output, enc_outputs)  # attn_weights : [1, 1, n_step]
            trained_attn.append(attn_weights.squeeze().data.numpy())

            # 矩阵-矩阵乘积 [1,1,n_step] x [1,n_step,n_hidden] = [1,1,n_hidden]
            context = attn_weights.bmm(enc_outputs.transpose(0, 1))
            dec_output = dec_output.squeeze(0)  # dec_output : [batch_size(=1), num_directions(=1) * n_hidden]
            context = context.squeeze(1)  # [1, num_directions(=1) * n_hidden]
            model[i] = self.out(torch.cat((dec_output, context), 1))

        # 使模型形状变为 [n_step, n_class]
        return model.transpose(0, 1).squeeze(0), trained_attn

    def get_att_weight(self, dec_output, enc_outputs):  # 获取 'dec_output' 对 'enc_outputs' 的注意力权重
        n_step = len(enc_outputs)
        attn_scores = torch.zeros(n_step)  # attn_scores : [n_step]

        for i in range(n_step):
            attn_scores[i] = self.get_att_score(dec_output, enc_outputs[i])

        # 将得分归一化为范围在 0 到 1 的权重
        return F.softmax(attn_scores).view(1, 1, -1)

    def get_att_score(self, dec_output, enc_output):  # enc_outputs [batch_size, num_directions(=1) * n_hidden]
        score = self.attn(enc_output)  # score : [batch_size, n_hidden]
        return torch.dot(dec_output.view(-1), score.view(-1))  # 内积得到标量值
    
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    parser.add_argument('--epoch', type=int, default=-1, help='epoch')
    parser=deepspeed.add_config_arguments(parser)
    cmd_args = parser.parse_args() #deepseed命令行参数
    return cmd_args

def train():
    args = parse_arguments()

    # init distributed
    deepspeed.init_distributed()

    n_step = 5  # 时间步长
    n_hidden = 128  # 隐藏单元数

    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']
    mydata = MyData(sentences, n_step, n_hidden)
    input_batch, output_batch, target_batch = mydata.make_batch()

    hidden = torch.zeros(1, 1, n_hidden).cuda()
    model = Attention(mydata.n_class, n_hidden).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 使用DeepSpeed进行单机多卡训练
    model, _, _, _ = deepspeed.initialize(args=args, model=model, model_parameters=model.parameters(),training_data=mydata) # deepspeed分布式模型,返回的model是分布式模型
    print('????????????????????')
    # 训练
    model.train()
    for epoch in range(args.epoch):
        print('!!!!!!!!!!!!!!!!!!!!!')
        input_batch = input_batch.cuda()
        output_batch = output_batch.cuda()
        hidden = hidden.cuda()
        target_batch = target_batch.cuda()
        output, _ = model(input_batch, hidden, output_batch)

        loss = criterion(output, target_batch.squeeze(0))
        if (epoch + 1) % 400 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        model.backward(loss) # 走deepspeed的反向传播
        model.step() # 更新参数
        # loss.backward()
        # optimizer.step()
    
    # 保存模型
    model.save_checkpoint('./checkpoint')

    # 测试
    test_batch = [np.eye(mydata.n_class)[[mydata.word_dict[n] for n in 'SPPPP']]]
    test_batch = torch.FloatTensor(test_batch)
    predict, trained_attn = model(input_batch, hidden, test_batch)
    predict = predict.data.max(1, keepdim=True)[1]
    print(sentences[0], '->', [mydata.number_dict[n.item()] for n in predict.squeeze()])

    # 显示注意力
    # fig = plt.figure(figsize=(5, 5))
    # ax = fig.add_subplot(1, 1, 1)
    # ax.matshow(trained_attn, cmap='viridis')
    # ax.set_xticklabels([''] + sentences[0].split(), fontdict={'fontsize': 14})
    # ax.set_yticklabels([''] + sentences[2].split(), fontdict={'fontsize': 14})
    # plt.show()

if __name__ == '__main__':
    train()

# 运行 deepspeed seq2seq_attention.py --epoch 1 --deepspeed --deepspeed_config ds_config.json