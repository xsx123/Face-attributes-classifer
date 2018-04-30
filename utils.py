import numpy as np
from scipy import misc
import random

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

def get_batch(img_paths,label_attrative,label_male,label_smiling,label_young,start,end):

    batch_img_paths = img_paths[start:end]
    nrof_samples = len(batch_img_paths)
    batch_data = []
    for i in range(nrof_samples):
        img = misc.imread(batch_img_paths[i])
        if img.ndim == 2:
            img = to_rgb(img)
        img = prewhiten(img)
        batch_data.append(img)

    batch_label_attrative = label_attrative[start:end]
    batch_label_male = label_male[start:end]
    batch_label_smiling = label_smiling[start:end]
    batch_label_young = label_young[start:end]

    return batch_data,batch_label_attrative,batch_label_male,batch_label_smiling,batch_label_young

# def shuffle(data):
#     random.shuffle(data)
#     return data

def shuffle(train_img_paths,train_atractive_label,train_male_label,train_smiling_label,train_young_label):
    shuffle_list = list(zip(train_img_paths,train_atractive_label,train_male_label,train_smiling_label,train_young_label))
    random.shuffle(shuffle_list)
    train_img_paths,train_atractive_label,train_male_label,train_smiling_label,train_young_label = zip(*shuffle_list)
    return train_img_paths,train_atractive_label,train_male_label,train_smiling_label,train_young_label




def load_img_path():
    img_paths = []
    dd = np.load('../real_atrri_parts_balance_4.npy')
    ll = len(dd)
    for i in range(ll):
        img_paths.append(dd[i][0])
    return img_paths


def load_data_label():
    img_paths = load_img_path()
    atractive_label = np.load('../data/atractive_label.npy')
    male_label = np.load('../data/male_label.npy')
    smiling_label = np.load('../data/smiling_label.npy')
    young_label = np.load('../data/young_label.npy')

    all_len = len(young_label)
    test_len = int(0.15*all_len)
    # train_len = all_len - test_len

    test_img_paths = img_paths[:test_len]
    train_img_paths = img_paths[test_len:]

    test_atractive_label = atractive_label[:test_len]
    train_atractive_label=atractive_label[test_len:]

    test_male_label = male_label[:test_len]
    train_male_label = male_label[test_len:]

    test_smiling_label = smiling_label[:test_len]
    train_smiling_label = smiling_label[test_len:]

    test_young_label = young_label[:test_len]
    train_young_label = young_label[test_len:]

    return train_img_paths,train_atractive_label,train_male_label,train_smiling_label,train_young_label,\
           test_img_paths,test_atractive_label,test_male_label,test_smiling_label,test_young_label

if __name__ == '__main__':
    img_paths = load_img_path()
    print('OK')