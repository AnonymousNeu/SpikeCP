# To run the code, load the data first
# Feel free to use any other neuromorphic dataset, and any other pre-trained SNN models

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from neurodata.load_data import create_dataloader
import snntorch as snn
import numpy as np

n_val = 1000

def softmax(x):
    exp_x = torch.exp(x)
    sum_exp_x = torch.sum(exp_x, dim=1, keepdim=True)
    softmax_probs = exp_x / sum_exp_x
    return softmax_probs

parser = argparse.ArgumentParser(description='SpikeCP')
parser.add_argument('--home', default=r"D:\pycharm\ILAC")
parser.add_argument('--dataset', default=r"/mnist_dvs_events/mnist_dvs_events.hdf5")
args = parser.parse_args()
print(args)

dataset_path = args.home + args.dataset
dtype = torch.float
device = torch.device("cpu")

batch_size = 128
num_ter = 4
target_size = 4
input_size = [2, 26, 26]
digits = [i for i in range(10)]

num_inputs = 2*26*26
num_hidden = 1000
num_outputs = 10

num_steps = 80
beta = 0.95

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):

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
for epoch in range(num_epochs):
    train_batch = iter(train_loader)

    for data, targets in train_batch:
        data = data.transpose(0, 1).to(device)
        targets = targets.to(device)
        targets = torch.sum(targets, dim=-1).argmax(-1).to(device)

        net.train()
        spk_rec, mem_rec, _ = net(data)

        loss_tr = torch.zeros((1), dtype=dtype, device=device)
        for step in range(num_steps):
            loss_tr += loss(mem_rec[step], targets).sum() / targets.size(0)

        optimizer.zero_grad()
        loss_tr.backward()
        optimizer.step()


aver_length = 50
# For global scores
av_cov = torch.zeros(aver_length)
av_car = torch.zeros(aver_length)
av_spk = torch.zeros(aver_length)
av_dl = torch.zeros(aver_length)
# For local scores
av_covs = torch.zeros(aver_length)
av_cars = torch.zeros(aver_length)
av_spks = torch.zeros(aver_length)
av_dls = torch.zeros(aver_length)

datasets = {}
for i in range(aver_length):
    val_dataset, test_dataset = random_split(combined_dataset, [n_val, 1000])
    datasets[i] = {'val': val_dataset, 'test': test_dataset}

alpha = 0.2  # alpha is 1 - p_targ
n_calibration = 100
sub_alpha = alpha / num_ter
for av in range(aver_length):
    total = 0
    # For global scores
    car = 0
    cov = 0
    delay = 0
    num_spk_sm = 0
    # For local scores
    cars = 0
    covs = 0
    delays = 0
    num_spks = 0

    val_dataset = datasets[av]['val']
    subset = Subset(val_dataset, range(n_calibration))
    test_dataset = datasets[av]['test']
    val_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_batch = iter(test_loader)
    val_batch = iter(val_loader)
    num_cv = len(val_loader)

    with torch.no_grad():
        net.eval()

        NCs = torch.zeros(num_ter, n_calibration + 1)
        NCs[:, 0] = 99999
        # global scores
        NC = torch.zeros(num_ter, n_calibration + 1)
        NC[:, 0] = 99999
        counts = 0
        for data_val, targets_val in val_batch:
            counts = counts + 1
            data_val = data_val.transpose(0, 1).to(device)
            targets_val = targets_val.to(device)
            targets_val = torch.sum(targets_val, dim=-1).argmax(-1).to(device)

            val_spk, _, _ = net(data_val)
            num_spk_val = val_spk.sum(dim=0)
            _, prediction = num_spk_val.max(1)
            cumsum_spk_val = torch.cumsum(val_spk, dim=0).transpose(0, 1)

            # local scores
            targets_val_int = targets_val.long()
            NC_batch = torch.zeros(num_ter, targets_val.size(
                0))
            for nn in range(num_ter):
                NC_batch[nn] = (num_steps / num_ter) * (nn + 1) - \
                               val_spk[:int((num_steps / num_ter) * (nn + 1))].sum(dim=0)[
                                   torch.arange(targets_val.size(0)), targets_val_int]
                if counts == num_cv:
                    NCs[nn, (counts - 1) * batch_size + 1:] = NC_batch[nn]
                else:
                    NCs[nn, (counts - 1) * targets_val.size(0) + 1:counts * targets_val.size(0) + 1] = NC_batch[
                        nn]

            # global scores
            one_hot_matrix = torch.zeros(targets_val.size(0), num_outputs)
            one_hot_matrix.scatter_(1, targets_val.long().unsqueeze(1), 1)
            NC_batch_sm = torch.zeros(num_ter, targets_val.size(0))
            for nn in range(num_ter):
                NC_batch_sm[nn] = torch.diag(
                    torch.mm(-torch.log(softmax(val_spk[:int((num_steps / num_ter) * (nn + 1))].sum(dim=0))),
                             one_hot_matrix.transpose(0, 1)))
                if counts == num_cv:
                    NC[nn, (counts - 1) * batch_size + 1:] = NC_batch_sm[nn]
                else:
                    NC[nn, (counts - 1) * targets_val.size(0) + 1:counts * targets_val.size(0) + 1] = \
                        NC_batch_sm[nn]

        # threshold for local scores
        index_ts = int(np.ceil((1 - sub_alpha) * (n_calibration + 1))) - 1
        thresholds_all = torch.zeros(num_ter)
        for nn in range(num_ter):
            thresholds_all[nn] = torch.sort(NCs[nn])[0][index_ts]

        # threshold for global scores
        index_ts_sm = int(np.ceil((1 - sub_alpha) * (n_calibration + 1))) - 1
        thresholdsm_all = torch.zeros(num_ter)
        for nn in range(num_ter):
            thresholdsm_all[nn] = torch.sort(NC[nn])[0][index_ts_sm]


        correct_test = 0
        delay_collect = 0
        # test:
        for data, targets in test_batch:
            data = data.transpose(0, 1).to(device)
            targets = targets.to(device)  # [128, 10, 80]
            targets = torch.sum(targets, dim=-1).argmax(-1).to(device)  # [128,]
            total += targets.size(0)

            test_spk, _, hidden_spk = net(data)
            spike_batch = (hidden_spk.sum(dim=2) / num_hidden).transpose(0, 1)
            cumsum_spk = torch.cumsum(spike_batch, dim=1)


            # local score:
            num = torch.zeros(num_ter, targets.size(0), num_outputs)
            for nn in range(num_ter):
                num[nn] = test_spk[:int((num_steps / num_ter) * (nn + 1)), :, :].sum(dim=0)
            NC_test = torch.zeros(num_ter, targets.size(0), num_outputs)
            cars_matrix = torch.zeros(targets.size(0), num_ter)
            for nn in range(num_ter):
                NC_test[nn] = (num_steps / num_ter) * (nn + 1) - num[nn]
                cars_matrix[:, nn] = ((NC_test[nn] <= thresholds_all[nn]) + 0).sum(dim=1)

            mask = (cars_matrix <= target_size) + 0
            zeros = mask.sum(dim=1)
            # find indices of 0s
            idx = torch.where(zeros == 0)[0]
            mask[idx, num_ter - 1] = 1
            min_indices = torch.argmax((mask == 1) + 0, dim=1)  # [128,]

            # informativeness
            cars_batch = cars_matrix[torch.arange(targets.size(0)), min_indices]  # [batch_size]
            cars += cars_batch.sum()

            mappings = ((torch.arange(num_ter) + 1) * (num_steps / num_ter)).int()
            delay_batch = mappings[min_indices]
            delays += delay_batch.sum()

            num_spike = cumsum_spk[torch.arange(targets.size(0)), delay_batch.long() - 1]
            num_spks += num_spike.sum()

            combined_NC = NC_test.transpose(0, 1)
            # pick them out
            mask = torch.zeros_like(combined_NC, dtype=torch.bool)
            for i in range(min_indices.shape[0]):
                mask[i, min_indices[i], :] = True
            selected_tensor = combined_NC[mask].reshape(min_indices.shape[0], -1)  # [128, 10]
            selected_thresholds = thresholds_all[min_indices]
            for i in range(num_outputs):
                if i == 0:
                    # For class 0, convert 1 to 0, and convert 0 to 1 and then to 10
                    pre = ((selected_tensor <= selected_thresholds.unsqueeze(1)) + 0)[:, i:i + 1] * (-1) + 1
                    pree = torch.where(pre == 1, torch.tensor(10), pre)
                else:
                    pre = ((selected_tensor <= selected_thresholds.unsqueeze(1)) + 0)[:, i:i + 1] * i
                    pree = torch.where(pre == 0, torch.tensor(10), pre)
                covs += (pree == torch.unsqueeze(targets, 1)).sum().item()

            # global score:
            NC_test_sm = torch.zeros(num_ter, targets.size(0), num_outputs)
            cars_matrix_sm = torch.zeros(targets.size(0), num_ter)
            for i in range(num_outputs):
                for nn in range(num_ter):
                    possible = i * torch.ones([targets.size(0)]).to(torch.uint8)
                    one_hot_matrix_test = torch.zeros(possible.size(0), num_outputs)  # [128, 10]
                    one_hot_matrix_test.scatter_(1, possible.long().unsqueeze(1), 1)
                    NC_test_sm[nn][:, i] = torch.diag(
                        torch.mm(-torch.log(softmax(num[nn])), one_hot_matrix_test.transpose(0, 1)))  # [128,]
                    cars_matrix_sm[:, nn] = ((NC_test_sm[nn] <= thresholdsm_all[nn]) + 0).sum(dim=1)

            mask_sm = (cars_matrix_sm <= target_size) + 0
            zeros = mask_sm.sum(dim=1)
            # find indices of 0s
            idx = torch.where(zeros == 0)[0]
            mask_sm[idx, num_ter - 1] = 1
            min_indices_sm = torch.argmax((mask_sm == 1) + 0, dim=1)  # [128,]

            # informativeness
            car_batch = cars_matrix_sm[torch.arange(targets.size(0)), min_indices_sm]
            car += car_batch.sum()

            delay_batch_sm = mappings[min_indices_sm]
            delay += delay_batch_sm.sum()

            num_spike = cumsum_spk[torch.arange(targets.size(0)), delay_batch_sm.long() - 1]
            num_spk_sm += num_spike.sum()

            combined_NC_sm = NC_test_sm.transpose(0, 1)
            mask_sm = torch.zeros_like(combined_NC_sm, dtype=torch.bool)
            for i in range(min_indices_sm.shape[0]):
                mask_sm[i, min_indices_sm[i], :] = True
            selected_tensor_sm = combined_NC_sm[mask_sm].reshape(min_indices_sm.shape[0], -1)  # [128, 10]
            selected_threshold_sm = thresholdsm_all[min_indices_sm]
            for i in range(num_outputs):
                if i == 0:
                    # For class 0, convert 1 to 0, and convert 0 to 1 and then to 10
                    pre_sm = ((selected_tensor_sm <= selected_threshold_sm.unsqueeze(1)) + 0)[:, i:i + 1] * (
                        -1) + 1
                    pree_sm = torch.where(pre_sm == 1, torch.tensor(10), pre_sm)
                else:
                    # For class i, convert 1 to i, and convert 0 to 10
                    pre_sm = ((selected_tensor_sm <= selected_threshold_sm.unsqueeze(1)) + 0)[:, i:i + 1] * i
                    pree_sm = torch.where(pre_sm == 0, torch.tensor(10), pre_sm)
                cov += (pree_sm == torch.unsqueeze(targets, 1)).sum().item()

    # For global scores
    av_cov[av] = cov / total
    av_car[av] = car / total
    av_dl[av] = delay / total
    av_spk[av] = num_spk_sm / total
    # For local scores
    av_covs[av] = covs / total
    av_cars[av] = cars / total
    av_spks[av] = num_spks / total
    av_dls[av] = delays / total

# For global scores
cov_final = av_cov.sum() / aver_length
car_final = av_car.sum() / aver_length
spk_final = av_spk.sum() / aver_length
dl_final = av_dl.sum() / aver_length
# For local scores
covs_final = av_covs.sum() / aver_length
cars_final = av_cars.sum() / aver_length
spks_final = av_spks.sum() / aver_length
dls_final = av_dls.sum() / aver_length

