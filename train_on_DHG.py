#tensorflow2.0.0
import torch
from util.DHG_parse_data import *
from util.Mydataset import *
import torch.optim as optim
import numpy as np
from datetime import datetime
import time
import argparse
import os
from model.network import *
import matplotlib.pyplot as plt
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()  #使用 argparse 的第一步是创建一个 ArgumentParser 对象。
#给一个 ArgumentParser 添加程序参数信息是通过调用 add_argument() 方法完成的
#Namespace(batch_size=32, cuda=True, data_cfg=0, dp_rate=0.2, epochs=300, learning_rate=0.001, patiences=50, test_subject_id=3, workers=8)
parser.add_argument("-b", "--batch_size", type=int, default=32)  # 16
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
parser.add_argument('--cuda', default=True, help='enables cuda')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')  # 1000
parser.add_argument('--patiences', default=50, type=int,
                    help='number of epochs to tolerate no improvement of val_loss')  # 1000
parser.add_argument('--test_subject_id', type=int, default=2,
                    help='id of test subject, for cross-validation')
parser.add_argument('--data_cfg', type=int, default=1,
                    help='0 for 14 class, 1 for 28')
parser.add_argument('--dp_rate', type=float, default=0.2,
                    help='dropout rate')  # 1000




def init_data_loader(test_subject_id, data_cfg):

    ##########调用DHG_parse_data.py 中的函数 get_train_test_data（）
    train_data, test_data = get_train_test_data(test_subject_id, data_cfg)#类型为列表，每个列表元素对应一个样本的 数据，标签 的字典
    print("数据载入完成!!!")

    #########调用 Mydataset.py 中的类Hand_Dataset（）
    ##加噪声，并归一化每一个video_data的帧数为time_len
    #1、随机选取time_len个index（包含头尾）
    #2、
    print('开始归一化每一个video_data的帧数为time_len!!!')
    train_dataset = Hand_Dataset(train_data, use_data_aug = True, time_len = 8)
    test_dataset = Hand_Dataset(test_data, use_data_aug = False, time_len = 8)

    print("train data num: ",len(train_dataset))
    print("test data num: ",len(test_dataset))
    print("batch size:", args.batch_size)
    print("workers:", args.workers)

    ######torch.utils.data.DataLoader（） 函数用来把训练数据分成多个(batch_size)小组，此函数每次抛出一组数据，直至把所有的数据都抛出。就是做一个数据的初始化。
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)  # batch_size=32  workers=8
    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    print("train_loader num: ", len(train_loader))
    print("val_loader num: ", len(val_loader))

    return train_loader, val_loader


def init_model(data_cfg):
    if data_cfg == 0:
        class_num = 14
    elif data_cfg == 1:
        class_num = 28

    ############network.py 中的 类 DG_STA（）
    model = DG_STA(class_num, args.dp_rate)

    #nn.DataParallel函数可以调用多个GPU，帮助加速训练。
    model = torch.nn.DataParallel(model).to(device)#######################################################cuda()

    return model


def model_foreward(sample_batched,model,criterion):


    data = sample_batched["s"].float() ## s(22)   f(6)  v(4)   sf(28)   sv(26)  sfv(32)
    f = sample_batched["f"].float()
    v = sample_batched["v"].float()

    label = sample_batched["label"]
    label = label.type(torch.LongTensor)

    ###.cuda()表示将本存储到CPU的网络及其参数存储到GPU！
    label = label.to(device) ############################################################cuda()
    label = torch.autograd.Variable(label, requires_grad=False)


    score = model(data,f,v)

    loss = criterion(score,label)     #计算loss

    acc = get_acc(score, label)

    return score,loss, acc


def get_acc(score, labels):
    score = score.cpu().data.numpy()
    labels = labels.cpu().data.numpy()
    outputs = np.argmax(score, axis=1)
    return np.sum(outputs==labels)/float(labels.size)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":

    tra_acc = []
    tra_loss = []
    test_acc = []
    test_loss = []

    print("\nhyperparamter......")
    args = parser.parse_args()
    print(args)

    print("test_subject_id: ", args.test_subject_id)

   #folder for saving trained model...
    # change this path to the fold where you want to save your pre-trained model
    model_fold = "./model/DHS_ID-{}_dp-{}_lr-{}_dc-{}/".format(args.test_subject_id,args.dp_rate, args.learning_rate, args.data_cfg)
    try:
        os.mkdir(model_fold)
    except:
        pass
    train_loader, val_loader = init_data_loader(args.test_subject_id,args.data_cfg)
    print(train_loader)
    print(val_loader)

    #.........inital model
    print("\ninit model.............")
    model = init_model(args.data_cfg)

    model_solver = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    #........set loss
    criterion = torch.nn.CrossEntropyLoss()
    #
    train_data_num = 2660
    test_data_num = 140
    iter_per_epoch = int(train_data_num / args.batch_size)

    #parameters recording training log
    max_acc = 0
    no_improve_epoch = 0
    n_iter = 0

    #***********training#***********
    for epoch in range(args.epochs):    #########300
        print("\ntraining.............")
        '''model.train() 和 model.eval() 一般在模型训练和评价的时候会加上这两句，
        主要是针对由于model 在训练时和评价时 Batch Normalization 和 Dropout 方法模式不同；
        因此，在使用PyTorch进行训练和测试时一定注意要把实例化的model指定train/eval；'''
        model.train()
        start_time = time.time()
        train_acc = 0
        train_loss = 0
        for i, sample_batched in enumerate(train_loader):
            n_iter += 1
            #print("training i:",i)
            if i + 1 > iter_per_epoch:
                continue
            score,loss, acc = model_foreward(sample_batched, model, criterion)

            model.zero_grad()
            loss.backward()
            #clip_grad_norm_(model.parameters(), 0.1)
            model_solver.step()


            train_acc += acc
            train_loss += loss
            #print(i)
        train_acc /= float(i + 1)
        train_loss /= float(i + 1)

        tra_acc.append(train_acc)
        tra_loss.append(train_loss)

        print("*** DHS  Epoch: [%2d] time: %4.4f, "
              "cls_loss: %.4f  train_ACC: %.6f ***"
              % (epoch + 1,  time.time() - start_time,
                 train_loss.data, train_acc))
        start_time = time.time()

        #adjust_learning_rate(model_solver, epoch + 1, args)
        #print(print(model.module.encoder.gcn_network[0].edg_weight))

        #***********evaluation***********
        with torch.no_grad():
            val_loss = 0
            acc_sum = 0
            model.eval()
            for i, sample_batched in enumerate(val_loader):
                #print("testing i:", i)
                label = sample_batched["label"]
                score, loss, acc = model_foreward(sample_batched, model, criterion)
                val_loss += loss

                if i == 0:
                    score_list = score
                    label_list = label
                else:
                    score_list = torch.cat((score_list, score), 0)
                    label_list = torch.cat((label_list, label), 0)


            val_loss = val_loss / float(i + 1)
            val_cc = get_acc(score_list,label_list)

            test_acc.append(val_cc)
            test_loss.append(val_loss)


            print("*** DHS  Epoch: [%2d], "
                  "val_loss: %.6f,"
                  "val_ACC: %.6f ***"
                  % (epoch + 1, val_loss, val_cc))

            #save best model
            if val_cc > max_acc:
                max_acc = val_cc
                no_improve_epoch = 0
                val_cc = round(val_cc, 10)

                #################模型保存，生成  .pth文件
                #################  .pth  的torch调用：如下
                '''import torch
                import torchvision.models as models
                # pretrained=True就可以使用预训练的模型
                net = models.squeezenet1_1(pretrained=False)
                pthfile = r'E:\anaconda\app\envs\luo\Lib\site-packages\torchvision\models\squeezenet1_1.pth'
                net.load_state_dict(torch.load(pthfile))'''
                torch.save(model.state_dict(),
                           '{}/epoch_{}_acc_{}.pth'.format(model_fold, epoch + 1, val_cc))
                print("performance improve, saved the new model......best acc: {}".format(max_acc))
            else:
                no_improve_epoch += 1
                print("no_improve_epoch: {} best acc {}".format(no_improve_epoch,max_acc))

            if no_improve_epoch > args.patiences:
                print("stop training....")
                break

    '''dataframe = pd.DataFrame({'train_acc': tra_acc,
                              'test_acc': test_acc
                              })
    dataframe.to_csv("DHG_ID-{}_dc-{}.csv".format(args.test_subject_id, args.data_cfg), index=False, header=None,
                     sep=',')'''