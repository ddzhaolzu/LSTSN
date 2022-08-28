import random
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from random import randint,shuffle
import math

class Hand_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data, time_len, use_data_aug):
        """
        Args:
            data: a list of video and it's label
            time_len: length of input video
            use_data_aug: flag for using data augmentation
        """
        self.use_data_aug = use_data_aug
        self.data = data

        self.time_len = time_len
        self.compoent_num = 22


    def __len__(self):
        return len(self.data)   #

    def __getitem__(self, ind):
        #print("ind:",ind)
        data_ele = self.data[ind]
        #hand skeleton
        skeleton = data_ele["skeleton"]   #shape:[valid_frame,22,3]   video
        skeleton = np.array(skeleton)

        if self.use_data_aug:
            skeleton = self.data_aug(skeleton)    ############
        # sample time_len frames from whole video
        data_num = skeleton.shape[0]  #########
        idx_list = self.sample_frame(data_num)##############

        skeleton = [skeleton[idx] for idx in idx_list]
        skeleton = np.array(skeleton)   #shape:[time_len,22,3]   video

        ##
        c = skeleton.sum(axis=1) / skeleton.shape[1]  ###(8,3)
        #c -= c[0]  ###(8,3)
        #
        f = []
        v = []
        for i in range(8):
            f.append(skeleton[i][5])
            f.append(skeleton[i][9])
            f.append(skeleton[i][13])
            f.append(skeleton[i][17])
            f.append(skeleton[i][21])
            f.append(c[i])

            v.append(skeleton[i][9] - skeleton[i][5])
            v.append(skeleton[i][13] - skeleton[i][5])
            v.append(skeleton[i][17] - skeleton[i][5])
            v.append(skeleton[i][21] - skeleton[i][5])
        f = np.array(f)  ###(8,6,3)
        f = f.reshape((8, 6, 3))    #
        f -= skeleton[0][1]  ###(8,6,3)
        #####
        v = np.array(v)
        v = v.reshape((8, 4, 3))

        #normalize by palm center
        skeleton -= skeleton[0][1]  #
        #####torch.from_numpy()
        skeleton = torch.from_numpy(skeleton).float()  #shape:[time_len,22,3]   video
        f = torch.from_numpy(f).float()  ###(8,6,3)
        v = torch.from_numpy(v).float()  ##(8, 4, 3)

        sf=torch.cat((skeleton, f), 1)  ##[time_len,  22+6 , 3]   (8,28,3)
        sv=torch.cat((skeleton, v), 1)  ##[time_len,  22+4 , 3]   (8,26,3)
        sfv=torch.cat((sf, v), 1)  ##[time_len,  22+6+4 , 3]   (8,32,3)
        
        '''
        Example:
         a = numpy.array([1, 2, 3])
         t = torch.from_numpy(a)
         >>> t
         tensor([ 1, 2, 3])
         >>> t[0] = -1
         >>> a
         array([-1, 2, 3])
        '''
        #print(skeleton.shape)
        # label
        label = data_ele["label"] - 1 #

        sample = {'s': skeleton, 'f': f, 'v': v, 'sf':sf, 'sv':sv, 'sfv':sfv, "label": label}

        return sample

    def data_aug(self, skeleton):

        ##########
        def scale(skeleton):
            ratio = 0.2
            low = 1 - ratio
            high = 1 + ratio
            factor = np.random.uniform(low, high)
            video_len = skeleton.shape[0]
            for t in range(video_len):
                for j_id in range(self.compoent_num):
                    skeleton[t][j_id] *= factor
            skeleton = np.array(skeleton)
            return skeleton

        ##########
        def shift(skeleton):
            low = -0.1
            high = -low
            offset = np.random.uniform(low, high, 3)
            video_len = skeleton.shape[0]
            for t in range(video_len):
                for j_id in range(self.compoent_num):
                    skeleton[t][j_id] += offset   #
            skeleton = np.array(skeleton)
            return skeleton

        #############
        def noise(skeleton):
            low = -0.1
            high = -low
            #select 4 joints
            all_joint = list(range(self.compoent_num))  #
            shuffle(all_joint)  #
            selected_joint = all_joint[0:4]  #

            for j_id in selected_joint:
                noise_offset = np.random.uniform(low, high, 3)
                for t in range(self.time_len):
                    skeleton[t][j_id] += noise_offset
            skeleton = np.array(skeleton)
            return skeleton

        ###############
        def time_interpolate(skeleton):
            skeleton = np.array(skeleton)
            video_len = skeleton.shape[0]

            r = np.random.uniform(0, 1)

            result = []
            for i in range(1, video_len):
                displace = skeleton[i] - skeleton[i - 1] #d_t = s_t+1 - s_t
                displace *= r
                result.append(skeleton[i -1] + displace) # r*disp

            while len(result) < self.time_len:
                result.append(result[-1]) #padding
            result = np.array(result)
            return result

        # og_id = np.random.randint(3)
        aug_num = 4
        ag_id = randint(0, aug_num - 1)  #

        if ag_id == 0:
            skeleton = scale(skeleton)
        elif ag_id == 1:
            skeleton = shift(skeleton)
        elif ag_id == 2:
            skeleton = noise(skeleton)
        elif ag_id == 3:
            skeleton = time_interpolate(skeleton)
        return skeleton

    ########个index
    def sample_frame(self, data_num):
        #sample #time_len frames from whole video
        sample_size = self.time_len
        each_num = (data_num - 1) / (sample_size - 1)
        idx_list = [0, data_num - 1]
        for i in range(sample_size):
            index = round(each_num * i)  #
            if index not in idx_list and index < data_num:
                idx_list.append(index)
        idx_list.sort()

        while len(idx_list) < sample_size:
            idx = randint(0, data_num - 1)
            if idx not in idx_list:
                idx_list.append(idx)
        idx_list.sort()

        return idx_list