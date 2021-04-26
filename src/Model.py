import torch
from torch import nn
import random

class Lstm(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size,num_layers,drop_out):
        super(Lstm,self).__init__()
        self.embed = nn.Linear(vocab_size,embed_size,bias = False)
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers,batch_first=True)
        self.dropout = nn.Dropout(drop_out) 
        self.linear = nn.Linear(hidden_size,vocab_size) 
        
        self.crossentropy_reduce = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, x):
        x = self.embed(x)
        output,(hidden,cell) = self.lstm(x)
        output = self.dropout(output)
        output = self.linear(output) 
        
        return output,(hidden,cell)
    
class Embedin(nn.Module):
    def __init__(self,vocab_size,embed_size):
        super(Embedin,self).__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size,embed_size)
    
    def forward(self,x):
        x = self.embed(x)
        return x

class Embedout(nn.Module):
    def __init__(self,vocab_size,hidden_size):
        super(Embedout,self).__init__()
        self.embed = nn.Linear(hidden_size,vocab_size)
    
    def forward(self,x):
        x = self.embed(x)
        return x
        

    

class Encoder(nn.Module):
    def __init__(self, embedin, embed_size, hidden_size, n_layers):
        super(Encoder,self).__init__()
        self.embedin = embedin
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, bidirectional=False, batch_first=True)

    def forward(self, x):
        x = self.embedin(x)
        output,(hidden,cell) = self.lstm(x)

        return output, hidden,cell
    
class Decoder(nn.Module):
    def __init__(self, embedin, embed_size, hidden_size, n_layers):
        super(Decoder,self).__init__()
        self.embedin = embedin
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, batch_first=True)
        
    def forward(self, x, hidden,cell):
        #x (batch_size,seq_len = 1)
        x = self.embedin(x)
        output, (hidden, cell) = self.lstm(x, (hidden,cell))      #output(batch_size,seq_len =1,hidden_size) 
        #output = self.linear(output)
        return output, hidden, cell


 
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        
        self.hidden_size = hidden_size
        self.linear_out = nn.Linear(2 * hidden_size, hidden_size)
        self.softmax2 = nn.Softmax(dim=2)
    
    def forward(self, encoder_output, decoder_output):
        #softmax2 = nn.Softmax(dim=2)
        # encoder_output: [batch_size, 13, hidden_size]
        # decoder_output: [batch_size, 1, hidden_size]

        encoder_output_trans = encoder_output.transpose(1, 2)# [batch_size, hidden_size, 13]
        attn = torch.bmm(decoder_output, encoder_output_trans)  # [batch_size, 1, 13]
        attn = self.softmax2(attn)
        context = torch.bmm(attn, encoder_output)   # [batch_size, 1, hidden_size]
        output = torch.cat((context, decoder_output), dim=2)  # [batch_size, 1, 2*hidden_size]
        output = torch.tanh(self.linear_out(output)) # [batch_size, 1, hidden_size]
    
        return output, attn
    
class Autoencoder(nn.Module):
    def __init__(self,encoder,decoder,attention,embedout,max_len):
        super(Autoencoder,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
        self.embedout = embedout
        self.max_len = max_len
    
   
    def forward(self,x, targets, teacher_force):
        #x:[batch_size,seq_len]
        batch_size = x.shape[0]
        max_len = self.max_len
        vocab_size = self.decoder.embedin.vocab_size
        encoder_output,hidden,cell = self.encoder(x)   
        outputs = torch.zeros(max_len,batch_size,vocab_size)
        outputs = outputs.cuda()
        # inputs -> batch_size * <bos>
        inputs = torch.zeros(batch_size,vocab_size)
        #Let position <eos> = 1
        inputs[:,0] = 1
        inputs_idx = torch.argmax(inputs, dim = 1,keepdim = False)
        inputs_idx = inputs_idx.cuda() 
        inputs_idx = inputs_idx.long() #input_idx->(batch_size)
        inputs_idx = inputs_idx.unsqueeze(1)
        for t in range(max_len):
            output, hidden,cell = self.decoder(inputs_idx, hidden,cell)
            output, _ = self.attention(encoder_output,hidden.transpose(0,1))
            ####delete this line
            output = self.embedout(output)
            output = output.squeeze(1)
            outputs[t] = output #output (batch_size,vocab_size)
            inputs_idx = torch.argmax(output,dim=1,keepdim=False)
            inputs_idx = inputs_idx.cuda()
            inputs_idx = inputs_idx.long()
            inputs_idx = inputs_idx.unsqueeze(1)
            use_teacher_force = random.random() <= teacher_force 
            inputs_idx = (targets[:,t].unsqueeze(1) if use_teacher_force else inputs_idx)
    

        outputs = outputs.transpose(0,1)
        return outputs
    
class Discriminator(nn.Module):
    def __init__(self,hidden_size):
        super(Discriminator,self).__init__()
        self.linear1 = nn.Linear(hidden_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,int(hidden_size/2))
        self.linear3 = nn.Linear(int(hidden_size/2),int(hidden_size/4))
        self.linear_out = nn.Linear(int(hidden_size/4),1)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self,encoder_output):
        x = encoder_output.reshape(encoder_output.shape[0]*encoder_output.shape[1],encoder_output.shape[2])
        
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_out(x)
        return x
    
class Discriminator2(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size):
        super(Discriminator2,self).__init__()
        #self.embed = nn.Embedding(vocab_size,embed_size)
        self.embed = nn.Linear(vocab_size,embed_size,bias=False)
        self.lstm = nn.LSTM(embed_size, hidden_size, 1, batch_first=True)
        self.linear1 = nn.Linear(hidden_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,int(hidden_size/2))
        self.linear3 = nn.Linear(int(hidden_size/2),int(hidden_size/4))
        self.linear_out = nn.Linear(int(hidden_size/4),1)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self,x):
        #if embed_matrix == True:
            #x = x @ self.embed.weight
        #elif embed_matrix == False:
            #x = self.embed(x)
        x = self.embed(x)
        output,(x,cell) = self.lstm(x)
        x = x.transpose(0,1)
        x = x.squeeze(1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_out(x)
        return x
        
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder,attention,embedout,max_len):
        super(Seq2Seq,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
        self.max_len = max_len
        self.embedout = embedout


    def forward(self, x, targets , teacher_force = 0.5):
        #x: (batch size,seq_len)
        #trg: (batch size,seq_len)
        batch_size = x.shape[0]
        max_len = self.max_len
        vocab_size = self.decoder.embedin.vocab_size
        
        outputs = torch.zeros(max_len,batch_size,vocab_size)
        outputs = outputs.cuda()
        inputss = torch.zeros(max_len+1,batch_size,vocab_size)
        inputss = inputss.cuda()
        probs = torch.zeros(max_len,batch_size)
        encoder_output,hidden,cell = self.encoder(x)
        #inputs -> batch_size * <bos>
        inputs = torch.zeros(batch_size,vocab_size)
        #Let position <eos> = 1
        inputs[:,0] = 1
        inputs_idx = torch.argmax(inputs, dim = 1,keepdim = False)
        inputs_idx = inputs_idx.cuda() 
        inputs_idx = inputs_idx.long() #input_idx->(batch_size)
        inputs_idx = inputs_idx.unsqueeze(1)
        inputs = inputs.cuda()
        inputss[0] = inputs        
        
        for t in range(max_len):
            output, hidden,cell = self.decoder(inputs_idx, hidden,cell)
            output, _ = self.attention(encoder_output,hidden.transpose(0,1))
            output = self.embedout(output)
            output = output.squeeze(1)
            outputs[t] = output
            inputs_idx = torch.argmax(output,dim=1,keepdim=False)
            inputs_idx = inputs_idx.cuda()
            inputs_idx = inputs_idx.long()
            inputs_idx = inputs_idx.unsqueeze(1)
            use_teacher_force = random.random() <= teacher_force
            inputs_idx = (targets[:,t].unsqueeze(1) if use_teacher_force else inputs_idx)
        
            
        outputs = outputs.transpose(0,1)
        inputss = inputss.transpose(0,1)
        probs = probs.transpose(0,1)
        
        return outputs



    
