# To run the code, load the MINIST-DVS data first

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from neurodata.load_data import create_dataloader
import snntorch as snn
import numpy as np

n_val = 1000
n_para = 3


def softmax(x):
    exp_x = torch.exp(x)
    sum_exp_x = torch.sum(exp_x, dim=1, keepdim=True)
    softmax_probs = exp_x / sum_exp_x
    return softmax_probs

parser = argparse.ArgumentParser(description='Point Classifier')
parser.add_argument('--home', default=r"D:\pycharm\ILAC")
parser.add_argument('--dataset', default=r"/mnist_dvs_events/mnist_dvs_events.hdf5")
args = parser.parse_args()
print(args)

batch_size = 128
num_ter = 80

dtype = torch.float
device = torch.device("cpu")

input_size = [2, 26, 26]

dataset_path = args.home + args.dataset
digits = [i for i in range(10)]

num_inputs = 2*26*26
num_hidden = 1000
num_outputs = 10

num_steps = 80
beta = 0.95

# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk1_rec = []
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            y = x[step].view(x.shape[1], -1)
            cur1 = self.fc1(y)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk1_rec.append(spk1)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0), torch.stack(spk1_rec, dim=0)


loss = nn.CrossEntropyLoss(reduction='none')

train_loader, combined_dataset = create_dataloader(dataset_path, batch_size=batch_size,
                                                   size=input_size,
                                                   classes=digits,
                                                   sample_length_train=2000000,
                                                   sample_length_test=2000000, dt=25000,
                                                   polarity=False, ds=1,
                                                   shuffle_test=False, num_workers=0, n_val=n_val)
net = Net().to(device)
num_epochs = 1
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))
# pre-train the SNN
for epoch in range(num_epochs):
    train_batch = iter(train_loader)

    for data, targets in train_batch:
        data = data.transpose(0, 1).to(device)  # [80, 128, 2, 26, 26]
        targets = targets.to(device)  # [128, 10, 80]
        targets = torch.sum(targets, dim=-1).argmax(-1).to(device)  # [128,]

        net.train()
        spk_rec, mem_rec, _ = net(data)  # [80, 128, 10]

        loss_tr = torch.zeros((1), dtype=dtype, device=device)
        for step in range(num_steps):
            loss_tr += loss(mem_rec[step], targets).sum() / targets.size(0)

        optimizer.zero_grad()
        loss_tr.backward()
        optimizer.step()


aver_length = 50
av_acc = torch.zeros(n_para, aver_length)
av_delay = torch.zeros(n_para, aver_length)
av_acco = torch.zeros(n_para, aver_length)
av_delayo = torch.zeros(n_para, aver_length)

datasets = {}
for i in range(aver_length):
    val_dataset, test_dataset = random_split(combined_dataset, [n_val, 1000])
    datasets[i] = {'val': val_dataset, 'test': test_dataset}

n_calibration = 100
for av in range(aver_length):
    total = 0
    correct_th = torch.zeros(101)

    val_dataset = datasets[av]['val']
    subset = Subset(val_dataset, range(n_calibration))
    test_dataset = datasets[av]['test']
    val_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_batch = iter(test_loader)
    val_batch = iter(val_loader)

    with torch.no_grad():
        net.eval()

        for data_val, targets_val in val_batch:
            data_val = data_val.transpose(0, 1).to(device)
            targets_val = targets_val.to(device)
            targets_val = torch.sum(targets_val, dim=-1).argmax(-1).to(device)

            val_spk, _, _ = net(data_val)
            num_spk_val = val_spk.sum(dim=0)
            _, prediction = num_spk_val.max(1)
            cumsum_spk_val = torch.cumsum(val_spk, dim=0).transpose(0, 1)

            MP = torch.zeros(targets_val.size(0), num_ter)
            for nn in range(num_ter):
                MP[:, nn], _ = torch.max(
                    softmax(val_spk[:int((num_steps / num_ter) * (nn + 1))].sum(dim=0)), dim=1)

            count_time = 0
            for TH in np.linspace(0, 1, num=101):
                mask_mp = torch.zeros(targets_val.size(0), num_ter)
                mask_mp = (MP >= TH) + 0
                zeros_mp = mask_mp.sum(dim=1)
                idx_mp = torch.where(zeros_mp == 0)[0]
                mask_mp[idx_mp, num_ter - 1] = 1
                time_val = torch.argmax((mask_mp == 1) + 0, dim=1)
                delay_val = (time_val + 1) * int((num_steps / num_ter))
                num_vfinal = cumsum_spk_val[torch.arange(targets_val.size(0)), delay_val - 1]  # [128, 10]
                _, pred_val = num_vfinal.max(1)
                correct_th[count_time] += (pred_val == targets_val).sum().item()
                count_time = count_time + 1

        # find the threshold p_th for each p_targ
        TH = torch.zeros(3)
        acc_val_th = correct_th / n_calibration
        j = 0
        for p_targ in [0.6, 0.7, 0.8]:
            above = (acc_val_th >= p_targ) + 0
            if above.sum() == 0:  # no thresholds lead to p_targ
                TH[j] = 1
            else:
                # find indices of non 0s
                indices = torch.nonzero(above, as_tuple=True)[0]
                # pick the smallest threshold from those in 'above'
                TH[j] = indices[0] * 0.01
            j = j + 1

        correct_test = torch.zeros(n_para)
        delay_collect = torch.zeros(n_para)
        correct2_test = torch.zeros(n_para)
        delay2_collect = torch.zeros(n_para)
        # test
        for data, targets in test_batch:
            data = data.transpose(0, 1).to(device)
            targets = targets.to(device)
            targets = torch.sum(targets, dim=-1).argmax(-1).to(device)
            total += targets.size(0)

            test_spk, _, _ = net(data)
            cumsum_spk = torch.cumsum(test_spk, dim=0).transpose(0, 1)
            MP = torch.zeros(targets.size(0), num_ter)
            for nn in range(num_ter):
                MP[:, nn], _ = torch.max(
                    softmax(test_spk[:int((num_steps / num_ter) * (nn + 1))].sum(dim=0)), dim=1)

            count_time = 0
            for th in TH:
                mask_mp = torch.zeros(targets.size(0), num_ter)
                mask_mp = (MP >= th) + 0
                zeros_mp = mask_mp.sum(dim=1)
                idx_mp = torch.where(zeros_mp == 0)[0]
                mask_mp[idx_mp, num_ter - 1] = 1
                time_test = torch.argmax((mask_mp == 1) + 0, dim=1)  # [128,]
                delay_test = (time_test + 1) * int((num_steps / num_ter))
                delay_collect[count_time] += delay_test.sum()
                num_tfinal = cumsum_spk[torch.arange(targets.size(0)), delay_test - 1]  # [128, 10]
                _, pred_test = num_tfinal.max(1)
                correct_test[count_time] += (pred_test == targets).sum().item()
                count_time = count_time + 1

            # p_th=p_targ
            count_time = 0
            for th in [0.6, 0.7, 0.8]:
                mask_mp = torch.zeros(targets.size(0), num_ter)
                mask_mp = (MP >= th) + 0
                zeros_mp = mask_mp.sum(dim=1)
                idx_mp = torch.where(zeros_mp == 0)[0]
                mask_mp[idx_mp, num_ter - 1] = 1
                time_test = torch.argmax((mask_mp == 1) + 0, dim=1)
                delay_test = (time_test + 1) * int((num_steps / num_ter))
                delay2_collect[count_time] += delay_test.sum()
                num_tfinal = cumsum_spk[torch.arange(targets.size(0)), delay_test - 1]
                _, pred_test = num_tfinal.max(1)
                correct2_test[count_time] += (pred_test == targets).sum().item()
                count_time = count_time + 1

    av_acc[:, av] = correct_test / 1000
    av_delay[:, av] = delay_collect / 1000
    av_acco[:, av] = correct2_test / 1000
    av_delayo[:, av] = delay2_collect / 1000


acc_final = av_acc.sum(dim=1) / aver_length
delay_final = av_delay.sum(dim=1) / aver_length
acco_final = av_acco.sum(dim=1) / aver_length
delayo_final = av_delayo.sum(dim=1) / aver_length


