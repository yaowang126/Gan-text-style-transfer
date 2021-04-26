import pickle
import numpy as np
import torch
import os

def get_seq(path):
    file=open(path,"rb")
    file=pickle.load(file) 

    number_seq = []
    for i, poet in enumerate(file):
        for j, seq in enumerate(poet):
            number_seq.append(seq)

    return number_seq
        
class Dictionary(object):

    def __init__(self):
        self.word2idx = {} 
        self.idx2word = {} 
        self.idx = 0
    
    def add_word(self, word):
        if not word in self.word2idx: 
            self.word2idx[word] = self.idx 
            self.idx2word[self.idx] = word 
            self.idx += 1
    
    def __len__(self):
        return len(self.word2idx) 
    
class Corpus(object):
    
    def __init__(self):
        self.dictionary = Dictionary()
 
    def get_dict(self, seq_list,mask = False):
        for seq in (seq_list):
            words = ['<bos>']+ [word for word in seq] + ['<eos>']
            for word in words:
                self.dictionary.add_word(word)
        if mask:
            self.dictionary.add_word('mask')
        return self.dictionary.word2idx, self.dictionary.idx2word
    
    def get_idx_seq(self, seq_list):
        idx_seq = []
        for seq in (seq_list):
            idx_seq.append([self.dictionary.word2idx[word] for word in ['<bos>']+[word1 for word1 in seq]+['<eos>']])
        return idx_seq
 
def mask_noise(x,prob,idx_mask):
    r = x.shape[0]
    c = x.shape[1]
    if prob == 0.: 
        return x

    assert 0 < prob < 1
    keep = torch.rand(r,c) >= prob
    keep = keep.long().cuda()
    keep[:,0] = 1.
    keep[:,-1] = 1.
    keep[:,6] = 1.
    keep[:,-2] = 1.
    unkeep = 1-keep
    mask = torch.ones(r,c)*idx_mask
    mask = mask.long().cuda()
    seq_mask = torch.mul(x,keep) + torch.mul(mask,unkeep)

    return seq_mask, keep

def  del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path,i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)