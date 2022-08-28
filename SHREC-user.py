from util.SHREC_parse_data import *
from util.Mydataset import *
import time
import argparse
from model.network import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument("-b", "--batch_size", type=int, default=32)
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
parser.add_argument('--cuda', default=True, help='enables cuda')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')  # 1000
parser.add_argument('--patiences', default=50, type=int,
                    help='number of epochs to tolerate the no improvement of val_loss')  # 1000
parser.add_argument('--data_cfg', type=int, default=0,
                    help='0 for 14 class, 1 for 28')
parser.add_argument('--dp_rate', type=float, default=0.2,
                    help='dropout rate')  # 1000


def init_data_loader(data_cfg):
    train_data, test_data = split_train_test(data_cfg)

    print("loading data!!!")

    train_dataset = Hand_Dataset(train_data, use_data_aug = True, time_len = 8)
    test_dataset = Hand_Dataset(test_data, use_data_aug = False, time_len = 8)

    print("train data num: ",len(train_dataset))
    print("test data num: ",len(test_dataset))

    print("batch size:", args.batch_size)
    print("workers:", args.workers)

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
    #print('data.shape',data.shape)
    f = sample_batched["f"].float()
    v = sample_batched["v"].float()

    label = sample_batched["label"]
    label = label.type(torch.LongTensor)
    label = label.to(device) ##########################################################cuda()
    label = torch.autograd.Variable(label, requires_grad=False)

    '''from thop import profile
    flops, params = profile(model, inputs=(data,f,v))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')'''

    score = model(data,f,v)

    loss = criterion(score,label)

    acc = get_acc(score, label)

    return score,loss, acc

def get_acc(score, labels):
    score = score.cpu().data.numpy()
    labels = labels.cpu().data.numpy()
    outputs = np.argmax(score, axis=1)
    return np.sum(outputs==labels)/float(labels.size)

def plot_Matrix(cm, classes, title=None, cmap=plt.cm.Blues):
    plt.rc('font', family='Times New Roman', size='5')

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    '''print("Normalized confusion matrix")
    str_cm = cm.astype(np.str).tolist()
    Y=[]
    i=0
    for row in str_cm:
        Y.append(float(row[i]))
        i +=1
        #print('\t'.join(row))
    average =sum(Y) / len(Y)
    print('average_acc:',average)'''

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) == 0:
                cm[i, j] = 0

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)


    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Actual',
           xlabel='Predicted')

    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)


    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) > 0:
                ax.text(j, i, format(int(cm[i, j] * 100 + 0.5), fmt) + '%',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('cm.jpg', dpi=880)
    # show confusion matrix
    #plt.savefig(savename, format='png')
    #plt.show()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    args = parser.parse_args()

    train_loader, val_loader = init_data_loader(args.data_cfg)

    # ........set loss
    criterion = torch.nn.CrossEntropyLoss()
    # ***********evaluation***********
    with torch.no_grad():
        val_loss = 0
        acc_sum = 0

        # .........inital model
        print("\nload model.............")
        model = init_model(args.data_cfg)
        model.load_state_dict(torch.load('./model/0.9559.pth', map_location=torch.device('cpu')))  #cpu
        #model.load_state_dict(torch.load('./model/0.9559.pth'))  #gpu
        model.eval()


        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))



        t1 = time.time()
        for i, sample_batched in enumerate(val_loader):
            # print("testing i:", i)
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
        val_cc = get_acc(score_list, label_list)

        print("*** time: %4.4f, "
              "val_loss: %.6f,"
              "val_ACC: %.6f ***"
              % (time.time() - t1, val_loss, val_cc))

        score_list = score_list.cpu().data.numpy()
        label_list = label_list.cpu().data.numpy()

        bestreallabel = label_list
        bestprelabel = np.argmax(np.array(score_list), axis=1)

        if args.data_cfg == 0:
            classes = ['Grab', 'Tap', 'Expand', 'Pinch', 'Rotation CW', 'Rotation CCW', 'Swipe Right', 'Swipe Left',
                       'Swipe Up', 'Swipe Down', 'Swipe X', 'Swipe+', 'Swipe V', 'Shake']
        else:
            classes = ['G(1)', 'G(2)', 'T(1)', 'T(2)', 'E(1)', 'E(2)', 'P(1)', 'P(2)', 'R-CW(1)', 'R-CW(2)', 'R-CCW(1)',
                       'R-CCW(2)', 'S-R(1)', 'S-R(2)', 'S-L(1)', 'S-L(2)',
                       'S-U(1)', 'S-U(2)', 'S-D(1)', 'S-D(2)', 'S-X(1)', 'S-X(2)', 'S-+(1)', 'S-+(2)', 'S-V(1)',
                       'S-V(2)',
                       'Sh(1)', 'Sh(2)']

        np.seterr(divide='ignore', invalid='ignore')
        cm = confusion_matrix(bestreallabel, bestprelabel)
        #print(cm)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        str_cm = cm.astype(np.str).tolist()
        Y = []
        i = 0
        for row in str_cm:
            Y.append(float(row[i]))
            i += 1
            # print('\t'.join(row))
        average = sum(Y) / len(Y)
        print('average_acc:', average)
        plot_Matrix(cm, classes, title=None, cmap=plt.cm.Blues)