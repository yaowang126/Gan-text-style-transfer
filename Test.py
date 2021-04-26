from Util import *
from Model import *
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

five_seq = get_seq(r"F:/Columbia/DL_syst_perf/project/five")
seven_seq = get_seq(r"F:/Columbia/DL_syst_perf/project/seven")

corpus = Corpus()

word2idx, idx2word = corpus.get_dict(five_seq+seven_seq,mask=True)
idx_five_seq = corpus.get_idx_seq(five_seq)
idx_seven_seq = corpus.get_idx_seq(seven_seq)

vocab_size = len(word2idx)

test_5 = torch.tensor(idx_five_seq[500000:500100], dtype = torch.long)
test_7 = torch.tensor(idx_seven_seq[500000:500100], dtype = torch.long)

embed_size = 128   
hidden_size = 512
n_layers = 1     
vocab_size = len(word2idx)

enc5 = Encoder(vocab_size, embed_size, hidden_size, n_layers)
dec5 = Decoder(vocab_size, embed_size, hidden_size, n_layers)
enc7 = Encoder(vocab_size, embed_size, hidden_size, n_layers)
dec7 = Decoder(vocab_size, embed_size, hidden_size, n_layers)
atten5 = Attention(hidden_size)
atten7 = Attention(hidden_size)
model57 = Seq2Seq(enc5, dec7,atten5,17)
model75 = Seq2Seq(enc7, dec7,atten7,13)
ae5 = Autoencoder(enc5,dec5,atten5,13)
ae7 = Autoencoder(enc7,dec5,atten7,17)


#ae5.load_state_dict(torch.load('F:/Columbia/DL_syst_perf/Poetry/models/ae5.pt'))
ae7.load_state_dict(torch.load('F:/Columbia/DL_syst_perf/Poetry/models/ae7.pt'))
#model57 = model57.cuda()
#model57 = model75.cuda()
ae5 = ae5.cuda()
ae7 = ae7.cuda()
test_5 = test_5.cuda()
test_7 = test_7.cuda()
outputae7 = ae7(test_7[:,:13],test_7[:,1:],teacher_force=0)


#outputae7 = ae7(test_7[:,:17],test_7[:,1:],teacher_force=0)
#output57, tryonehot57, tryprob57 = model57(test_5,test_7,gumbel=0.0,teacher_force=0.0)
#output57 = torch.argmax(output57,dim=2,keepdim=False)
outputae7 = torch.argmax(outputae7,dim=2,keepdim=False)
output = outputae7.cpu()
output = output.tolist()
poem = [[idx2word[idx] for idx in line] for line in output]
poem = [''.join(line) for line in poem]

f = open('F:/Columbia/DL_syst_perf/Poetry/output/ae7_dropout.txt','w',encoding='utf-8')
for i, line  in enumerate(poem):
    f.write(line)
    f.write(seven_seq[500000+i])
    f.write('\n')
f.close()
