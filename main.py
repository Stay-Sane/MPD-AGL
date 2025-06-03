# -*- coding: utf-8 -*-
"""
    @Author       : 保持理智
    @Version      : v1.0
    @Date         : 2025-05-20 09:16:18
    @Description  : Main Function
"""
import os
import math
import time
import tqdm
import torch
import torch.utils.data
from datetime import datetime

from base_config import args
from data_loader import loader
from models import ResNet19, VGGSNN
os.environ['CUDA_VISIBLE_DEVICES']='0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main_SNN():
    if args.data_type == "CIFAR10":
        snn = ResNet19.ResNet19(T=args.T)
    elif args.data_type == "CIFAR10-DVS":
        snn = VGGSNN.VGGSNN(T=args.T)
    else:
        raise (ValueError('Unavailable dataset'))
    snn.to(device)
    print('{}'.format(snn))
    f_para.write('\n {}'.format(snn))

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(snn.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4) # weight_decay=1e-4 for CIFAR10 / 5e-4 for CIFAR10-DVS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch, eta_min=0)

    train_loader, test_loader = loader.getDataLoader(args.data_set, args.data_type, args.batch_size,
                                                     args.data_augment, args.num_workers)
    start_time = time.time()
    best_acc, best_epoch = 0, 0
    for epoch in range(args.num_epoch):
        snn.train()
        total, correct, train_loss = 0., 0., 0.
        for images, labels in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)

            outputs_T = snn(images)
            loss = loss_function(outputs_T.mean(dim=1), labels.long())
            train_loss += loss.cpu().detach().item()
            loss.backward()
            optimizer.step()

            total += labels.numel()
            correct += (outputs_T.mean(dim=1).argmax(dim=1) == labels).float().sum().item()
        time_point = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        train_acc = 100. * float(correct / total)
        print('%s | Epoch [%d/%d], Train_loss: %f, Train_acc: %.2f'
              % (time_point, epoch + 1, args.num_epoch, train_loss, train_acc))
        f_log.write('%s | Epoch [%d/%d], Train_loss: %f, Train_acc: %.2f\n'
                    % (time_point, epoch + 1, args.num_epoch, train_loss, train_acc))
        scheduler.step()


        total, correct, test_loss = 0., 0., 0.
        snn.eval()
        with torch.no_grad():
            for images, labels in tqdm.tqdm(test_loader):
                images, labels = images.to(device), labels.to(device)

                outputs_T = snn(images)
                loss = loss_function(outputs_T.mean(dim=1), labels.long())
                test_loss += loss.cpu().detach().item()

                total += labels.numel()
                correct += (outputs_T.mean(dim=1).argmax(dim=1) == labels).float().sum().item()
        time_elapsed = time.time() - start_time
        test_acc = 100. * float(correct / total)
        print('%s | Epoch [%d/%d], Test_loss: %f, Test_acc: %.2f, Time elapsed:%.fh %.0fm %.0fs'
            % (time_point, epoch + 1, args.num_epoch, test_loss, test_acc, time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))
        f_log.write('%s | Epoch [%d/%d], Test_loss: %f, Test_acc: %.2f, Time elapsed:%.fh %.0fm %.0fs\n'
                    % (time_point, epoch + 1, args.num_epoch, test_loss, test_acc, time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))

        if test_acc >= best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            print('Saving.....')
            f_log.write('Saving.....\n')
            state = {
                'net': snn.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_acc': best_acc,
                'best_epoch': best_epoch,
            }
            torch.save(state, result_path + 'best_model.pth')
        print('best acc is %.2f in epoch %d\n' % (best_acc, best_epoch))
        f_log.write('best acc is %.2f in epoch %d\n\n' % (best_acc, best_epoch))

if __name__ == '__main__':
    run_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    result_path = './results/' + args.data_type + '/' + run_time + '/checkpoints/'
    if not os.path.isdir(result_path):
        os.makedirs(result_path)

    f_para = open(os.path.join(result_path, 'hyper_parameter.log'), 'w', buffering=1)
    f_para.write(' Arguments: ')
    for arg in vars(args):
        f_para.write('\n\t {:25} : {}'.format(arg, getattr(args, arg)))
    f_log = open(os.path.join(result_path, args.data_type + '.log'), 'a', buffering=1)
    main_SNN()