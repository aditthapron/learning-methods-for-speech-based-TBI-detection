import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import recall_score,precision_score,accuracy_score

from models import GRU_MLP,SincNet_global_2
from data_preparation import *

# setting seed
torch.manual_seed(seed)
np.random.seed(seed)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# loss function
cost = nn.BCELoss()
m = nn.Sigmoid()

#initialize model
CNN_net=SincNet_global_2(wlen,fs)

if pt_file!='none':
    checkpoint_load = torch.load(pt_file)
    CNN_net.load_state_dict(checkpoint_load['CNN_model_par'])
else:
    raise('Path to pre-trained weights is not specified.')

if use_cuda:
    CNN_net.cuda()
for param in CNN_net.parameters():
    param.requires_grad = False

DNN_net = GRU_MLP(CNN_net.out_dim)
if use_cuda:
    DNN_net.cuda() 

optimizer_CNN = optim.RMSprop(CNN_net.parameters(), lr=lr*0.1,alpha=0.95, eps=1e-8) 
optimizer_DNN = optim.RMSprop(DNN_net.parameters(), lr=lr,alpha=0.95, eps=1e-8) 
scheduler_CNN = StepLR(optimizer_CNN, step_size=1, gamma=0.98) 
scheduler_DNN = StepLR(optimizer_DNN, step_size=1, gamma=0.98)

training_set = Dataset_(batch_size,wav_lst_tr,snt_tr,wlen,0.2,tran_lst_tr)
dataloader = DataLoader(training_set, batch_size=batch_size,
                        shuffle=False, num_workers=8, drop_last=True)
early_stop=0
best_loss=np.inf
fine_tuning=False

#GRU memory
Subject_h = dict()
fine_tuning_epoch=-1
for epoch in range(N_epochs):
    if fine_tuning:
        CNN_net.train()
    else:
        CNN_net.eval()
    DNN_net.train()

    loss_sum=0
    err_sum=0
    c=0
    for local_batch, local_labels,subj in dataloader:
        # Transfer to GPU
        inp, lab = local_batch.float().to(device), local_labels.float().to(device)
        #resize
        inp = inp.view(inp.shape[0]*chunk_n,wlen)
        feature = CNN_net(inp)

        #format latent space to stack up 39 of 200 ms to represent 4s
        feature = feature.view(inp.shape[0]//chunk_n,chunk_n,CNN_net.out_dim,CNN_net.out_filt_len).permute(0,2,1,3)
        feature = feature.reshape(inp.shape[0]//chunk_n,CNN_net.out_dim,chunk_n*CNN_net.out_filt_len).permute(0,2,1)

        #get memory
        if subj in Subject_h:
            h_in = Subject_h[subj]
        else:
            #if new subject initilize to be zero
            h_in =  torch.zeros((1, 4, 8)).to(device)
        pout,h_out= DNN_net(feature,h_in)
        #save memory
        Subject_h[subj] = h_out
        
        #estimate loss
        pred = torch.round(pout) 
        loss = cost(pout, lab.float())
        err = torch.mean((pred!=lab).float())

        #model optimization
        if fine_tuning:
            optimizer_CNN.zero_grad()
        optimizer_DNN.zero_grad()      
        loss.backward(retain_graph=True)
        if fine_tuning:
            optimizer_CNN.step()
        optimizer_DNN.step()

        loss_sum=loss_sum+loss.detach()
        err_sum=err_sum+err.detach()
        c=c+1


    loss_tot=loss_sum/c
    err_tot=err_sum/c
    
    #reduce learning rate
    if fine_tuning:
        scheduler_CNN.step()
    scheduler_DNN.step()
    
    # Validation and checkpoint  
    if epoch%N_eval_epoch==0:
                        
        CNN_net.eval()
        DNN_net.eval()

        loss_sum=0
        err_sum=0
        err_sum_snt=0
        
        with torch.no_grad():  
            acc,prec,recall,pred_all,label_all=[],[],[],[],[]
            Subject_h_val=dict()
            for i in range(snt_te):                      
                signal, fs = librosa.load(wav_lst_te[i],16000)
                subj = wav_lst_te[i].split('/')[-1]
                signal=signal/np.max(np.abs(signal))
                select_ind = np.zeros(len(signal),dtype=bool)
                time = np.array(cha_read(tran_lst_te[i])).astype(int)
                time = (time*16000//1000)
                for t in time:
                    select_ind[t[0]:t[1]]=1
                signal = signal[select_ind]
                
                signal=torch.from_numpy(signal).float().cuda().contiguous()
                if tran_lst_te[i].split('/')[-2] =='TB':
                    lab_batch = 1 
                else:
                    lab_batch = 0
            
                # split signals into chunks
                beg_samp=0
                end_samp=wlen*chunk_n

                N_fr=int((signal.shape[0]-wlen*chunk_n)/(wshift))
                sig_arr=torch.zeros([Batch_dev,chunk_n*wlen]).float().cuda().contiguous()                
                count_fr=0
                count_fr_tot=0
                temp=[]
                
                while end_samp<signal.shape[0]:
                    try:
                        sig_arr[count_fr,:]=signal[beg_samp:end_samp]
                    except:
                        print(sig_arr.size())
                        print(wlen)
                        print(count_fr)
                    beg_samp=beg_samp+wshift
                    end_samp=beg_samp+wlen*chunk_n
                    count_fr=count_fr+1
                    count_fr_tot=count_fr_tot+1
                    if count_fr==Batch_dev:
                        inp=Variable(sig_arr)
                        inp = inp.view(Batch_dev*chunk_n,wlen)
                        feature = CNN_net(inp)
                        feature = feature.view(Batch_dev,chunk_n,CNN_net.out_dim,CNN_net.out_filt_len).permute(0,2,1,3)
                        feature = feature.reshape(Batch_dev,CNN_net.out_dim,chunk_n*CNN_net.out_filt_len).permute(0,2,1)
                        
                        #get memory
                        if subj in Subject_h_val:
                            h_in = Subject_h_val[subj]
                        else:
                            #if new subject initilize to be zero
                            h_in =  torch.zeros((1, 4, 8)).to(device)
                        pout,h_out= DNN_net(feature,h_in)
                        #save memory
                        Subject_h[subj] = h_out
                        pout=pout.view(-1)

                        lab= Variable((torch.zeros(Batch_dev)+lab_batch).cuda().contiguous().float())                        
                        loss = cost(pout, lab.float())
                        err = torch.mean((pred!=lab.float()).float())
                        loss_sum = loss_sum+loss.detach()/N_fr
                        err_sum=err_sum+err.detach()/N_fr
                        pred = torch.round(pout) 
                        pred_all.append(pred.detach().cpu().numpy())
                        label_all.append(lab.detach().cpu().numpy())
                        count_fr=0
                        sig_arr=torch.zeros([Batch_dev,chunk_n*wlen]).float().cuda().contiguous()
            

            acc.append(accuracy_score(np.hstack(label_all),np.hstack(pred_all)))
            recall.append(recall_score(np.hstack(label_all),np.hstack(pred_all)))
            prec.append(precision_score(np.hstack(label_all),np.hstack(pred_all)))
            err_tot_dev_snt=0
            loss_tot_dev=loss_sum/snt_te
            err_tot_dev=err_sum/snt_te

        
        print("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f" % (epoch, loss_tot,err_tot,loss_tot_dev,err_tot_dev,err_tot_dev_snt))
        print("ACC:{} +- {}\tPrec:{} +- {}\tRecall:{} +- {}".format(np.mean(acc),np.std(acc),np.mean(prec),np.std(prec),np.mean(recall),np.std(recall)))
        with open(output_folder+"/res_"+str(argv[1])+".res", "a") as res_file:
            res_file.write("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f\n" % (epoch, loss_tot,err_tot,loss_tot_dev,err_tot_dev,err_tot_dev_snt))   
            res_file.write("ACC:{} +- {}\tPrec:{} +- {}\tRecall:{} +- {}".format(np.mean(acc),np.std(acc),np.mean(prec),np.std(prec),np.mean(recall),np.std(recall)))
        if loss_tot_dev < best_loss:
            checkpoint={'CNN_model_par': CNN_net.state_dict(),
                        'DNN_model_par': DNN_net.state_dict()}
            torch.save(checkpoint,output_folder+'/model_raw_'+str(argv[1])+'.pkl')
            early_stop=0
            best_loss = loss_tot_dev
        else:
            early_stop = early_stop + 1

        #begin fine-tuning
        if ~fine_tuning and early_stop==10:
            fine_tuning=True
            early_stop=0
            fine_tuning_epoch=epoch

            for param in CNN_net.parameters():
                param.requires_grad = True

            #set early stage of fine tuning
            optimizer_CNN = optim.RMSprop(CNN_net.parameters(), lr=1e-5,alpha=0.95, eps=1e-8) 
            optimizer_DNN = optim.RMSprop(DNN_net.parameters(), lr=1e-5,alpha=0.95, eps=1e-8) 
            scheduler_CNN = StepLR(optimizer_CNN, step_size=1, gamma=6.31) 
            scheduler_DNN = StepLR(optimizer_DNN, step_size=1, gamma=6.31)

        if epoch == fine_tuning_epoch+10:
            #return to normal learning rate
            optimizer_CNN = optim.RMSprop(CNN_net.parameters(), lr=1e-3,alpha=0.95, eps=1e-8) 
            optimizer_DNN = optim.RMSprop(DNN_net.parameters(), lr=1e-3,alpha=0.95, eps=1e-8) 
            scheduler_CNN = StepLR(optimizer_CNN, step_size=1, gamma=0.95) 
            scheduler_DNN = StepLR(optimizer_DNN, step_size=1, gamma=0.95)
        #Early-stopping execution
        if fine_tuning and early_stop==20:
            break
    
    else:
        print("epoch %i, loss_tr=%f err_tr=%f" % (epoch, loss_tot,err_tot))


