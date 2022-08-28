import torch
from util.SHREC_parse_data import *
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
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument("-b", "--batch_size", type=int, default=32)  # 16
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
parser.add_argument('--cuda', default=True, help='enables cuda')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')  # 1000
parser.add_argument('--patiences', default=100, type=int,
                    help='number of epochs to tolerate the no improvement of val_loss')  # 1000
parser.add_argument('--data_cfg', type=int, default=0,
                    help='0 for 14 class, 1 for 28')
parser.add_argument('--dp_rate', type=float, default=0.2,
                    help='dropout rate')  # 1000


def init_data_loader(data_cfg):
    ##########调用SHREC_parse_data.py 中的函数 get_train_test_data（）
    train_data, test_data = split_train_test(data_cfg)
    print("数据载入完成!!!")

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
        num_workers=args.workers, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    return train_loader, val_loader

def init_model(data_cfg):
    if data_cfg == 0:
        class_num = 14
    elif data_cfg == 1:
        class_num = 28

    model = DG_STA(class_num, args.dp_rate)
    model = torch.nn.DataParallel(model).to(device) ######################################################cuda()

    return model


def model_foreward(sample_batched,model,criterion):


    data = sample_batched["s"].float() ## s(22)   f(6)  v(4)   sf(28)   sv(26)  sfv(32)
    f = sample_batched["f"].float()
    v = sample_batched["v"].float()

    label = sample_batched["label"]
    label = label.type(torch.LongTensor)
    label = label.to(device) ##########################################################cuda()
    label = torch.autograd.Variable(label, requires_grad=False)


    score = model(data,f,v)

    loss = criterion(score,label)

    acc = get_acc(score, label)

    return score,loss, acc



def get_acc(score, labels):
    score = score.cpu().data.numpy()
    labels = labels.cpu().data.numpy()
    outputs = np.argmax(score, axis=1)
    return np.sum(outputs==labels)/float(labels.size)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def plot_confusion_matrix(cm, savename, classes, title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=120)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.01:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()


def plot_Matrix(cm, classes, title=None, cmap=plt.cm.Blues):
    plt.rc('font', family='Times New Roman', size='8')  # 设置字体样式、大小

    # 按行进行归一化
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
    str_cm = cm.astype(np.str).tolist()
    for row in str_cm:
        print('\t'.join(row))
    # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) == 0:
                cm[i, j] = 0

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax) # 侧边的颜色条带

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Actual',
           xlabel='Predicted')

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 将x轴上的lables旋转45度
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 标注百分比信息
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) > 0:
                ax.text(j, i, format(int(cm[i, j] * 100 + 0.5), fmt) + '%',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('cm.jpg', dpi=300)
    # show confusion matrix
    #plt.savefig(savename, format='png')
    plt.show()

if __name__ == "__main__":
    tra_acc = []
    tra_loss = []
    test_acc = []
    test_loss = []

    print("\nhyperparamter......")
    args = parser.parse_args()
    print(args)

   #fold for saving trained model...
    #change this path to the fold where you want to save your pre-trained model
    model_fold = "./model/SHREC_dp-{}_lr-{}_dc-{}/".format(args.dp_rate, args.learning_rate, args.data_cfg)
    try:
        os.mkdir(model_fold)
    except:
        pass



    train_loader, val_loader = init_data_loader(args.data_cfg)


    #.........inital model
    print("\ninit model.............")
    model = init_model(args.data_cfg)
    model_solver = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    #........set loss
    criterion = torch.nn.CrossEntropyLoss()


    #
    train_data_num = 1960
    test_data_num = 840
    iter_per_epoch = int(train_data_num / args.batch_size)

    #parameters recording training log
    max_acc = 0
    no_improve_epoch = 0
    n_iter = 0

    #***********training#***********
    val_reallabel=[]
    val_prelabel=[]
    for epoch in range(args.epochs):
        print("\ntraining.............")
        model.train()
        start_time = time.time()
        train_acc = 0
        train_loss = 0
        for i, sample_batched in enumerate(train_loader):
            n_iter += 1
            #print("training i:",i)
            if i + 1 > iter_per_epoch:
                continue
            score,loss, acc= model_foreward(sample_batched, model, criterion)

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

        print("*** SHREC  Epoch: [%2d] time: %4.4f, "
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
                    score_list = torch.cat((score_list, score), 0)    #torch.cat ( (A, B), dim=0)接受一个由两个（或多个）tensor组成的元组，按行拼接
                    label_list = torch.cat((label_list, label), 0)

            val_loss = val_loss / float(i + 1)
            val_cc = get_acc(score_list,label_list)

            test_acc.append(val_cc)
            test_loss.append(val_loss)


            print("*** SHREC  Epoch: [%2d], "
                  "val_loss: %.6f,"
                  "val_ACC: %.6f ***"
                  % (epoch + 1, val_loss, val_cc))

            #save best model
            if val_cc > max_acc:
                max_acc = val_cc
                no_improve_epoch = 0
                val_cc = round(val_cc, 10)
                score_list = score_list.cpu().data.numpy()
                label_list = label_list.cpu().data.numpy()
                bestreallabel = label_list
                bestprelabel = np.argmax(np.array(score_list), axis=1)


                torch.save(model.state_dict(),
                           '{}/epoch_{}_acc_{}.pth'.format(model_fold, epoch + 1, val_cc))
                print("performance improve, saved the new model......best acc: {}".format(max_acc))

            else:
                no_improve_epoch += 1
                print("no_improve_epoch: {} best acc {}".format(no_improve_epoch,max_acc))

            if no_improve_epoch > args.patiences:
                print("stop training....")
                break

    if args.data_cfg==0:
        classes = ['Grab', 'Tap', 'Expand', 'Pinch', 'Rotation CW', 'Rotation CCW', 'Swipe Right', 'Swipe Left',
                   'Swipe Up', 'Swipe Down', 'Swipe X', 'Swipe+', 'Swipe V', 'Shake']
    else:
        classes = ['G(1)', 'G(2)', 'T(1)', 'T(2)', 'E(1)', 'E(2)', 'P(!)', 'P(2)', 'R-CW(1)', 'R-CW(2)', 'R-CCW(1)',
                   'R-CCW(2)', 'S-R(1)', 'S-R(2)', 'S-L(1)', 'S-L(2)',
                   'S-U(1)', 'S-U(2)', 'S-D(1)', 'S-D(2)', 'S-X(1)', 'S-X(2)', 'S-+(1)', 'S-+(2)', 'S-V(1)', 'S-V(2)',
                   'Sh(1)', 'Sh(2)']
    '''dataframe = pd.DataFrame({'reallabel':  bestreallabel,
                              'prelabel': bestprelabel
                              })
    dataframe.to_csv("位置嵌入SHRE-{}.csv".format(args.data_cfg), index=False, header=None,
                     sep=',')'''