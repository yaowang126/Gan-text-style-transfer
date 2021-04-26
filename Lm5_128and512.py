from Util import *
from Model import Lstm
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils import shuffle
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('F:/Columbia/DL_syst_perf/Poetry/tensorboard_lm5')

five_seq = get_seq(r"F:/Columbia/DL_syst_perf/project/five")
seven_seq = get_seq(r"F:/Columbia/DL_syst_perf/project/seven")

corpus = Corpus()
word2idx, idx2word = corpus.get_dict(five_seq+seven_seq,mask=True)
idx_five_seq = corpus.get_idx_seq(five_seq)
idx_seven_seq = corpus.get_idx_seq(seven_seq)
idx_five_seq = shuffle(idx_five_seq)
idx_seven_seq = shuffle(idx_seven_seq)

embed_size = 128   
hidden_size = 512
num_layers = 1     
num_epochs = 20
batch_size = 256  
learning_rate = 1e-4
vocab_size = len(word2idx)
src = [seq[:-1] for seq in idx_five_seq]
trg = [seq[1:] for seq in idx_five_seq]
train_src = torch.tensor(src[:500000], dtype = torch.long)
train_trg = torch.tensor(trg[:500000], dtype = torch.long)
train_dataset = TensorDataset(train_src, train_trg)
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

test_src = torch.tensor(src[500000:], dtype = torch.long)
test_trg = torch.tensor(trg[500000:], dtype = torch.long)
test_dataset = TensorDataset(test_src, test_trg)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

lm5 = Lstm(vocab_size,embed_size,hidden_size,num_layers,drop_out=0.25)
lm5.cuda()
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(lm5.parameters(), lr=learning_rate)

#train

val_loss_min = 0.
for epoch in range(num_epochs):
    lm5.train()
    for i,(inputs, targets) in enumerate(train_loader):
        
        inputs = torch.zeros(inputs.shape[0], inputs.shape[1], len(word2idx)).scatter_(dim=-1,
                              index=torch.unsqueeze(inputs,-1),
                               value=1)
        inputs = inputs.cuda()
        targets = targets.cuda()
        outputs,_ = lm5(inputs,None)
        optimizer.zero_grad()
        loss = criterion(outputs,targets.reshape(-1))
        loss.backward()
        #clip_grad_norm_(model.parameters(),0.5)
        optimizer.step()
        
        writer.add_scalar('training loss',loss, epoch * len(train_loader) + i) 

    
    lm5.eval()
    val_loss = 0.
    with torch.no_grad():
        for i,(inputs, targets) in enumerate(test_loader):
            inputs = torch.zeros(inputs.shape[0], inputs.shape[1], len(word2idx)).scatter_(dim=-1,
                       index=torch.unsqueeze(inputs,-1),
                       value=1)
            inputs = inputs.cuda()
            targets = targets.cuda()
            with torch.no_grad():
                outputs, hidden = lm5(inputs,None)
            loss = criterion(outputs,targets.reshape(-1))
            val_loss += loss.item()*inputs.shape[0]

    val_loss = val_loss / len(test_dataset)
    writer.add_scalar('validation loss',val_loss, epoch)
    if val_loss_min == 0 or val_loss < val_loss_min:
        val_loss_min = val_loss
        torch.save(lm5.state_dict(), "F:/Columbia/DL_syst_perf/Poetry/models/lm5_lstm_dropout.th")
        print('save')
 
