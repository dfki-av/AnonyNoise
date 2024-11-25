import torch 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import seaborn as sns
import os 

def fliplr(img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(3)-1,-1,-1).long().cuda()  # N x C x H x W
        img_flip = img.index_select(3,inv_idx)
        return img_flip
           
def id_evaluate(qf, ql, qe, qfid, gf, gl, ge, gfid):
    query = qf
    score = np.dot(gf,query)
           
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    query_index = []
    for i in range(len(gl)):
        if gl[i] == ql:
            query_index.append(i)
    file_emo_index = []
    for i in range(len(gfid)):
        if gfid[i] == qfid or ge[i] == qe:
            file_emo_index.append(i)
    good_index = np.setdiff1d(query_index, file_emo_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, file_emo_index)
    junk_index = np.append(junk_index2, junk_index1)
    CMC_tmp = id_compute_mAP(index, good_index, junk_index)
    return CMC_tmp

def id_compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc

def plot_confusion_matrix(cm, unique_labels_emo, tensorboard, current_epoch):
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels_emo, yticklabels=unique_labels_emo, center=np.max(cm)//2)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    
    tensorboard.add_figure('conf_matrix', fig, current_epoch)
    plt.close()
    
def plot_event_frame(ev_tmp, ev_tmp2, i, tensorboard, current_epoch, name = ''):
    ev = ev_tmp.view(int(ev_tmp.size(0)/2), 2, ev_tmp.size(1), ev_tmp.size(2))
    width, height = ev_tmp.size(1), ev_tmp.size(2)
    event_data = torch.sum(ev, dim= 0).permute(1, 2, 0)
    
    tmp = torch.zeros(width, height, 1)
    event_data = torch.cat((event_data, tmp), dim = 2) 
    max_value = torch.max(event_data)*0.9
    event_data = torch.clamp(event_data, max=max_value)/max_value
    
    ev2 = ev_tmp2.view(int(ev_tmp2.size(0)/2), 2, ev_tmp2.size(1), ev_tmp2.size(2))
    event_data2 = torch.sum(ev2, dim= 0).permute(1, 2, 0)
    event_data2 = torch.cat((event_data2, tmp), dim = 2)#/torch.max(event_data2)
    max_value = torch.max(event_data2)*0.9
    event_data2 = torch.clamp(event_data2, max=max_value)/max_value
    
    fig, ax = plt.subplots(figsize=(2 * event_data2.shape[1]/100, event_data2.shape[0]/100), dpi=100)
    ax.set_position([0, 0, 1, 1])

    plt.subplot(1, 2, 1)
    plt.imshow(event_data.detach().cpu().numpy())
    plt.title('Original')    
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(event_data2.detach().cpu().numpy())
    plt.title('Anonymized')
    plt.axis('off')
    tensorboard.add_figure(f'recon/image{i}', fig, current_epoch)
    plt.close()
        

