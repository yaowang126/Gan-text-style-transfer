from Util import *
from Model import  Embedin, Embedout, Encoder, Decoder, Attention, Autoencoder, Discriminator, Discriminator2, Lstm
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


torch.backends.cudnn.enabled = True


def get_dictionary():
    five_seq = get_seq("data/five")
    seven_seq = get_seq("data/seven")
    all_seq = five_seq + seven_seq
    
    corpus = Corpus()
    word2idx, idx2word = corpus.get_dict(all_seq,mask=True)
    idx_five_seq = corpus.get_idx_seq(five_seq)
    idx_seven_seq = corpus.get_idx_seq(seven_seq)
    return five_seq, seven_seq, idx_five_seq, idx_seven_seq, word2idx, idx2word


def get_traindata(idx_five_seq, idx_seven_seq,batch_size):
    train_5 = torch.tensor(idx_five_seq[:500000], dtype = torch.long)
    train_7 = torch.tensor(idx_seven_seq[:500000], dtype = torch.long)
    train_dataset = TensorDataset(train_5, train_7)
    train_loader = DataLoader(train_dataset,batch_size = batch_size,shuffle=True)
    
    test_5 = torch.tensor(idx_five_seq[500000:500010], dtype = torch.long)
    test_7 = torch.tensor(idx_seven_seq[500000:500010], dtype = torch.long)
    test_5 = test_5.cuda()
    test_7 = test_7.cuda()
    test_set = (test_5,test_7)
    
    return train_loader, test_set

def get_model():

    embedin = Embedin(vocab_size,embed_size)
    embedout = Embedout(vocab_size,hidden_size)
    
    enc = Encoder(embedin, embed_size, hidden_size, n_layers)
    dec5 = Decoder(embedin, embed_size, hidden_size, n_layers)
    #enc7 = Encoder(embedin, embed_size, hidden_size, n_layers)
    dec7 = Decoder(embedin, embed_size, hidden_size, n_layers)
    atten = Attention(hidden_size)
    #atten7 = Attention(hidden_size)
    ae5 = Autoencoder(enc,dec5,atten,embedout,13)
    ae7 = Autoencoder(enc,dec7,atten,embedout,17)
    
    discriminator = Discriminator(hidden_size)
    discriminator2 = Discriminator2(vocab_size,embed_size,hidden_size)
    
    seq2seq57 = Autoencoder(enc,dec7,atten,embedout,17)
    seq2seq75 = Autoencoder(enc,dec5,atten,embedout,13)
    
    lm5 = Lstm(vocab_size,embed_size,hidden_size,n_layers,drop_out=0)
    lm7 = Lstm(vocab_size,embed_size,hidden_size,n_layers,drop_out=0)
    lm5.load_state_dict(torch.load('models/lm5_lstm_dropout.th'))
    lm7.load_state_dict(torch.load('models/lm7_lstm_dropout.th'))
    
    ae5 = ae5.cuda()
    ae7 = ae7.cuda()
    discriminator = discriminator.cuda()
    discriminator2 = discriminator2.cuda()
    seq2seq57 = Autoencoder(enc,dec7,atten,embedout,17)
    seq2seq75 = Autoencoder(enc,dec5,atten,embedout,13)
    lm5 = lm5.cuda()
    lm7 = lm7.cuda()
    return ae5, ae7, discriminator, discriminator2, seq2seq57, seq2seq75, lm5, lm7
#define metrics



def get_optim():
    optimizer5 = torch.optim.RMSprop([{'params': ae5.parameters()}],lr=learning_rate)
    optimizer7 = torch.optim.RMSprop([{'params': ae7.parameters()}],lr=learning_rate)
    optimizerdis = torch.optim.RMSprop([{'params':discriminator.parameters()}],lr=learning_rate)
    optimizerdis2 = torch.optim.RMSprop([{'params':discriminator2.parameters()}],lr=learning_rate)
    optimizer57 = torch.optim.RMSprop([{'params':seq2seq57.parameters()}],lr=learning_rate)
    optimizer75 = torch.optim.RMSprop([{'params':seq2seq75.parameters()}],lr=learning_rate)

    return optimizer5, optimizer7, optimizerdis, optimizerdis2, optimizer57, optimizer75
       
        
        
  #train discriminator
#---------------------------------------------------------------------------------------------
def discriminator_train(inputs5,targets5,inputs7,targets7):    
    discriminator.train()
    ae5.eval()
    ae7.eval()
    encode5,_,_ = ae5.encoder(inputs5)
    encode5 = encode5.detach()
    dis5 = discriminator(encode5)
    encode7,_,_ = ae7.encoder(inputs7)
    encode7 = encode7.detach()  
    dis7 = discriminator(encode7)
    len5 = torch.tensor(13,dtype=torch.long)
    len7 = torch.tensor(17,dtype=torch.long)
    
    dis_loss = -dis5.mean() + dis7.mean()
    
    optimizerdis.zero_grad()
    dis_loss.backward()
    clip_grad_norm_(discriminator.parameters(),0.5)
    optimizerdis.step()    
    for parm in discriminator.parameters():
        parm.data.clamp_(-0.01,0.01)
    writer.add_scalar('dis loss',dis_loss, epoch * len(train_loader) + i)        
    
def autoencoders_train(inputs5,targets5,inputs7,targets7):  #train autoencoders
#---------------------------------------------------------------------------------------------
    crossentropy = nn.CrossEntropyLoss()
    
    ae5.train()
    ae7.train()
    
    inputs5_noise,_ = mask_noise(inputs5,0.2,vocab_size-1)
    outputae5 = ae5(inputs5_noise,targets5,tf)
    outputae5 = outputae5.reshape(outputae5.shape[0]*outputae5.shape[1],outputae5.shape[2])
    targets5 = targets5.reshape(-1)
    loss_ae5 = crossentropy(outputae5,targets5)
    
    inputs7_noise,_ = mask_noise(inputs7,0.2,vocab_size-1)
    outputae7 = ae7(inputs7_noise,targets7,teacher_force=tf)
    outputae7 = outputae7.reshape(outputae7.shape[0]*outputae7.shape[1],outputae7.shape[2])
    targets7 = targets7.reshape(-1)
    loss_ae7 = crossentropy(outputae7,targets7)
    
    optimizer5.zero_grad()
    optimizer7.zero_grad()
    
    ae_loss = loss_ae5 + loss_ae7
    ae_loss.backward()
    clip_grad_norm_(ae5.parameters(),0.5)
    clip_grad_norm_(ae7.parameters(),0.5)
    optimizer5.step()
    optimizer7.step()

    writer.add_scalar('ae5 loss',loss_ae5, epoch * len(train_loader) + i)
    writer.add_scalar('ae7 loss',loss_ae7, epoch * len(train_loader) + i)
#train ae5 and ae7 for cheating
#---------------------------------------------------------------------------------------------
def discriminator_cheating_train(inputs5,targets5,inputs7,targets7): 
    ae7.train()
    discriminator.eval()
    encode7,_,_ = ae7.encoder(inputs7)
    dis7 = discriminator(encode7)
    cheat_loss = -dis7.mean()
    #cheat_loss = -crossentropy(dis5,disy5) - crossentropy(dis7,disy7)
    
    optimizer7.zero_grad()
    cheat_loss.backward()
    optimizer7.step()

    
#train discriminator2
#-----------------------------------------------------------------------------------------------
def discriminator2_train(inputs5,targets5,inputs7,targets7):             
    discriminator2.train()
    seq2seq57.eval()
    trans57 = seq2seq57(inputs5,targets5,teacher_force=0)
    trans57 = trans57.detach()
    inputs7_onehot = torch.zeros(inputs7.shape[0], inputs7.shape[1], len(word2idx)).cuda()
    inputs7_onehot = inputs7_onehot.scatter_(dim=-1,index=torch.unsqueeze(inputs7,-1),value=1)
    real7,_ = lm7(inputs7_onehot)
    disg7 = discriminator2(trans57)
    disr7 = discriminator2(real7)
    dis7_loss = -disr7.mean() + disg7.mean()
  
    optimizerdis2.zero_grad()
    dis7_loss.backward()
    optimizerdis2.step()
    for parm in discriminator2.parameters():
        parm.data.clamp_(-0.01,0.01)
    #for name, param in discriminator2.named_parameters():
    #    if name.startswith('Linear'):
    #        param.data.clamp_(-0.01,0.01)
    writer.add_scalar('dis7 loss',dis7_loss, epoch * len(train_loader) + i)

#train seq2seq57 for cheating
#----------------------------------------------------------------------------------------------
def discriminator2_cheating_train(inputs5,targets5,inputs7,targets7):
    seq2seq57.train()
    discriminator2.eval()
    trans57 = seq2seq57(inputs5,targets5,teacher_force=0)
    disg7 = discriminator2(trans57)
    cheat7_loss = -disg7.mean()
    optimizer57.zero_grad()
    cheat7_loss.backward()
    optimizer57.step()
        
            
            
#add tensorboard
#------------------------------------------------------------------------------------------------          

def test_translate(test_set):
    test_5,test_7 = test_set

    poem_output=''
    ae5.eval()
    trans5 = ae5(test_5[:,:13],test_5[:,1:],teacher_force=0)
    trans5 = torch.argmax(trans5,dim=2,keepdim=False)
    output5 =  trans5.cpu()
    output5 = output5.tolist()
    poem5 = [[idx2word[idx] for idx in line] for line in output5]
    poem5 = [''.join(line) for line in poem5]      
   
    seq2seq57.eval()
    trans57 = seq2seq57(test_5[:,:13],test_5[:,1:],teacher_force=0)
    trans57 = torch.argmax(trans57,dim=2,keepdim=False)
    output57 = trans57.cpu()
    output57 = output57.tolist()
    poem57 = [[idx2word[idx] for idx in line] for line in output57]
    poem57 = [''.join(line) for line in poem57]
    for i,line in enumerate(poem5):
        poem_output = poem_output+poem5[i]+'||'+poem57[i]+'||'+five_seq[500000+i]+'||(T^T)'

   
    writer.add_text('output', poem_output, epoch * len(train_loader) + i)

def save_model():   
    torch.save(ae5.state_dict(),'models/ae5.pt')
    torch.save(ae7.state_dict(),'models/ae7.pt')
    torch.save(discriminator.state_dict(),'models/discriminator.pt')
    #torch.save(discriminator2.state_dict(),'models/discriminator2_2disc.pt')

# main-------------------------------------------------------------------------------------------------------
    
five_seq, seven_seq, idx_five_seq, idx_seven_seq, word2idx, idx2word = get_dictionary()

vocab_size = len(word2idx)    
embed_size = 128   
hidden_size = 512
n_layers = 1     
num_epochs = 10   
batch_size = 1024
learning_rate = 1e-4

train_loader, test_set = get_traindata(idx_five_seq, idx_seven_seq, batch_size)

ae5, ae7, discriminator, discriminator2, seq2seq57, seq2seq75, lm5, lm7 = get_model()

optimizer5, optimizer7, optimizerdis, optimizerdis2, optimizer57, optimizer75 = get_optim()
 


tensorboard_path = 'tensorboard_file'
del_file(tensorboard_path)
writer = SummaryWriter(tensorboard_path)

for epoch in range(num_epochs):
    print(epoch)
    
    if epoch < num_epochs - 1:
        tf = 1/(epoch+1)
    else:
        tf = 0
    
    for i,(train5, train7) in enumerate(train_loader):
        inputs5, targets5 = train5[:,:13], train5[:,1:]
        inputs7, targets7 = train7[:,:17], train7[:,1:]            
        inputs5, targets5, inputs7, targets7 = inputs5.cuda(), targets5.cuda(), inputs7.cuda(), targets7.cuda()
        
        discriminator_train(inputs5,targets5,inputs7,targets7)
        autoencoders_train(inputs5,targets5,inputs7,targets7)
        if i%5 == 0:
            discriminator_cheating_train(inputs5,targets5,inputs7,targets7)
       # discriminator2_train(inputs5,targets5,inputs7,targets7)
       # if i%3 ==0:
       #     discriminator2_cheating_train(inputs5,targets5,inputs7,targets7)
            
        test_translate(test_set)
        
    save_model()

