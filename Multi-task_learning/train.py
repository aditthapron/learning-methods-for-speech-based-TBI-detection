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
MSEloss = nn.MSELoss()
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

DNN_net = GRU_MLP(CNN_net.out_dim)
if use_cuda:
    DNN_net.cuda() 

DNN_net_WOZ = GRU_MLP(CNN_net.out_dim)
if use_cuda:
    DNN_net_WOZ.cuda() 

optimizer_CNN = optim.RMSprop(CNN_net.parameters(), lr=lr*0.1,alpha=0.95, eps=1e-8) 
optimizer_DNN = optim.RMSprop(DNN_net.parameters(), lr=lr,alpha=0.95, eps=1e-8) 
scheduler_CNN = StepLR(optimizer_CNN, step_size=1, gamma=0.98) 
scheduler_DNN = StepLR(optimizer_DNN, step_size=1, gamma=0.98)

training_set = Dataset_(batch_size,wav_lst_tr,snt_tr,wlen,0.2,tran_lst_tr)
dataloader = DataLoader(training_set, batch_size=batch_size,
                        shuffle=False, num_workers=8, drop_last=True)

training_set_WOZ = Dataset_WOZ(batch_size,WOZ_train,len(WOZ_train),wlen,0.2)
dataloader_WOZ = DataLoader(training_set_WOZ, batch_size=batch_size,
                        shuffle=False, num_workers=8, drop_last=True)
early_stop=0
best_loss=np.inf
fine_tuning=False

#GRU memory
Subject_h = dict()
Subject_h_WOZ = dict()

# weights for Gradnorm
Weightloss1 = torch.tensor(torch.FloatTensor([1]), requires_grad=True)
Weightloss2 = torch.tensor(torch.FloatTensor([1]), requires_grad=True)
params = [Weightloss1, Weightloss2]
opt2 = torch.optim.Adam(params, lr=0.001)
Gradloss = nn.L1Loss()

for epoch in range(N_epochs):
    dataloader_iterator = iter(dataloader_WOZ)
    CNN_net.train()
    DNN_net.train()

    loss_sum=0
    err_sum=0
    c=0
    for local_batch, local_labels,subj in dataloader:
        try:
            local_batch_WOZ, local_labels_WOZ,subj_WOZ = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(dataloader_WOZ)
            local_batch_WOZ, local_labels_WOZ,subj_WOZ = next(dataloader_iterator)
        # Transfer to GPU
        inp, lab = local_batch.float().to(device), local_labels.float().to(device)
        inp_WOZ, lab_WOZ =  local_batch_WOZ.float().to(device), local_labels_WOZ.float().to(device)
        
        ###TBI part
        inp = inp.view(inp.shape[0]*chunk_n,wlen)
        feature = CNN_net(inp)

        #format latent space to stack up 39 of 200 ms to represent 4s
        feature = feature.view(inp.shape[0]//chunk_n,chunk_n,CNN_net.out_dim,CNN_net.out_filt_len).permute(0,2,1,3)
        feature = feature.reshape(inp.shape[0]//chunk_n,CNN_net.out_dim,chunk_n*CNN_net.out_filt_len).permute(0,2,1)

        #get memory
        h_in=torch.zeros((1, 4, 8)).to(device)
        for S in range(batch_size)
            if subj[S] in Subject_h:
                h_in[:,S,:] = Subject_h[subj]
            else:
                #if new subject initilize to be zero
                h_in[:,S,:] = torch.zeros((1, 1, 8)).to(device)
        pout,h_out= DNN_net(feature,h_in)
        #save memory
        for S in range(batch_size):
            Subject_h[subj[S]] = h_out[:,S,:]
        

        
        #estimate loss
        pred = torch.round(pout) 
        l1 = cost(pout, lab.float())
        err = torch.mean((pred!=lab).float())

        ### Secondary dataset part
        inp_WOZ = inp_WOZ.view(inp_WOZ.shape[0]*chunk_n,wlen)
        feature = CNN_net(inp_WOZ)
        feature = feature.view(inp_WOZ.shape[0]//chunk_n,chunk_n,CNN_net.out_dim,CNN_net.out_filt_len).permute(0,2,1,3)
        feature = feature.reshape(inp_WOZ.shape[0]//chunk_n,CNN_net.out_dim,chunk_n*CNN_net.out_filt_len).permute(0,2,1)
        h_in=torch.zeros((1, 4, 8)).to(device)
        for S in range(batch_size)
            if subj_WOZ[S] in Subject_h_WOZ:
                h_in[:,S,:] = Subject_h_WOZ[subj_WOZ]
            else:
            #if new subject initilize to be zero
                h_in[:,S,:] = torch.zeros((1, 1, 8)).to(device)
        pout,h_out= DNN_net_WOZ(feature,h_in)
        #save memory
        for S in range(batch_size):
            Subject_h_WOZ[subj_WOZ[S]] = h_out[:,S,:]

        #Gradnorm
        l2 = cost(MSEloss, lab_WOZ.float())
        loss = torch.div(torch.add(l1,l2), 2)
        if epoch == 0:
            l01 = l1.data  
            l02 = l2.data
        optimizer_CNN.zero_grad()
        optimizer_DNN.zero_grad()
        optimizer_DNN_WOZ.zero_grad()      
        loss.backward(retain_graph=True)

        G1R = torch.autograd.grad(l1, CNN_net.parameters()[0], retain_graph=True, create_graph=True)
        G1 = torch.norm(G1R[0], 2)
        G2R = torch.autograd.grad(l2, CNN_net.parameters()[0], retain_graph=True, create_graph=True)
        G2 = torch.norm(G2R[0], 2)
        G_avg = torch.div(torch.add(G1, G2), 2)
        # Calculating relative losses 
        lhat1 = torch.div(l1,l01)
        lhat2 = torch.div(l2,l02)
        lhat_avg = torch.div(torch.add(lhat1, lhat2), 2)
        # Calculating relative inverse training rates for tasks 
        inv_rate1 = torch.div(lhat1,lhat_avg)
        inv_rate2 = torch.div(lhat2,lhat_avg)
        # Calculating the constant target for Eq. 2 in the GradNorm paper
        C1 = G_avg*(inv_rate1)**alph
        C2 = G_avg*(inv_rate2)**alph
        C1 = C1.detach()
        C2 = C2.detach()

        opt2.zero_grad()
        Lgrad = torch.add(Gradloss(G1, C1),Gradloss(G2, C2))
        Lgrad.backward()

        opt2.step()
        optimizer_CNN.step()
        optimizer_DNN.step()
        optimizer_DNN_WOZ.step()
        # Renormalizing the losses weights
        coef = 2/torch.add(Weightloss1, Weightloss2)
        params = [coef*Weightloss1, coef*Weightloss2]

        loss_sum=loss_sum+loss.detach()
        err_sum=err_sum+err.detach()
        c=c+1


    loss_tot=loss_sum/c
    err_tot=err_sum/c
    
    #reduce learning rate
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
                        h_in=torch.zeros((1, Batch_dev, 8)).to(device)
                        for S in range(Batch_dev)
                            if subj[S] in Subject_h_val:
                                h_in[:,S,:] = Subject_h_val[subj]
                            else:
                                #if new subject initilize to be zero
                                h_in[:,S,:] = torch.zeros((1, 1, 8)).to(device)
                        pout,h_out= DNN_net(feature,h_in)
                        #save memory
                        for S in range(Batch_dev):
                            Subject_h_val[subj[S]] = h_out[:,S,:]

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

    
    else:
        print("epoch %i, loss_tr=%f err_tr=%f" % (epoch, loss_tot,err_tot))


