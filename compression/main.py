import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from dgl.data import GINDataset
from dataloader import GINDataLoader
from ginparser import Parser
from gin import GIN
from gcn import GCN
import time
from utils.compresser import Compresser


def train(args, net, trainloader, optimizer, criterion, epoch):
    net.train()

    running_loss = 0
    total_iters = len(trainloader)
    # setup the offset to avoid the overlap with mouse cursor
    begin = time.time()
    for graphs, labels in trainloader:
        # batch graphs will be shipped to device in forward part of model
        labels = labels.to(args.device)
        graphs = graphs.to(args.device)
        feat = graphs.ndata.pop('attr')
        outputs = net(graphs, feat)

        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # report
    end = time.time()

    # the final batch will be aligned
    running_loss = running_loss / total_iters

    return running_loss, end - begin


def eval_net(args, net, dataloader, criterion):
    net.eval()

    total = 0
    total_loss = 0
    total_correct = 0

    for data in dataloader:
        graphs, labels = data
        graphs = graphs.to(args.device)
        labels = labels.to(args.device)
        feat = graphs.ndata.pop('attr')
        total += len(labels)
        outputs = net(graphs, feat)
        _, predicted = torch.max(outputs.data, 1)

        total_correct += (predicted == labels.data).sum().item()
        loss = criterion(outputs, labels)
        # crossentropy(reduce=True) for default
        total_loss += loss.item() * len(labels)

    loss, acc = 1.0*total_loss / total, 1.0*total_correct / total

    net.train()

    return loss, acc


def main(args):

    # set up seeds, args.seed supported
    torch.manual_seed(seed=args.seed)
    np.random.seed(seed=args.seed)

    is_cuda = not args.disable_cuda and torch.cuda.is_available()

    if is_cuda:
        args.device = torch.device("cuda:" + str(args.device))
        torch.cuda.manual_seed_all(seed=args.seed)
    else:
        args.device = torch.device("cpu")

    dataset = GINDataset(args.dataset, not args.learn_eps, args.degree_as_nlabel)

    print(len(dataset))
    print(dataset[1])
    print(dataset[1][0].ndata['attr'])

    trainloader, validloader = GINDataLoader(
        dataset, batch_size=args.batch_size, device=args.device,
        seed=args.seed, shuffle=True,
        split_name='fold10', fold_idx=args.fold_idx).train_valid_loader()
    # or split_name='rand', split_ratio=0.7

    if args.model == 'gin':
        model = GIN(
            args.num_layers, args.num_mlp_layers,
            dataset.dim_nfeats, args.hidden_dim, dataset.gclasses,
            args.final_dropout, args.learn_eps,
            args.graph_pooling_type, args.neighbor_pooling_type).to(args.device)
    elif args.model == 'gcn':
        model = GCN(
            args.num_layers, args.num_mlp_layers,
            dataset.dim_nfeats, args.hidden_dim, dataset.gclasses,
            args.final_dropout, args.learn_eps,
            args.graph_pooling_type, args.neighbor_pooling_type).to(args.device)

    criterion = nn.CrossEntropyLoss()  # defaul reduce is true
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    epoch_time_list = []
    for epoch in range(args.epochs):

        _, epoch_time = train(args, model, trainloader, optimizer, criterion, epoch)
        scheduler.step()

        epoch_time_list.append(epoch_time)

        train_loss, train_acc = eval_net(
            args, model, trainloader, criterion)

        valid_loss, valid_acc = eval_net(
            args, model, validloader, criterion)


        print("[epoch %d][train_loss, train_acc, valid_loss, valid_acc] : %.4f %.4f %.4f %.4f" % (
                    epoch,
                    train_loss,
                    train_acc,
                    valid_loss,
                    valid_acc
                ))

    print("avg_epoch_Time ", np.mean(epoch_time_list[10:]))
    print("max_epoch_Time ", np.max(epoch_time_list[10:]))
    print("min_epoch_Time ", np.min(epoch_time_list[10:]))


if __name__ == '__main__':
    args = Parser(description='GIN').args
    print('show all arguments configuration...')
    print(args)

    main(args)