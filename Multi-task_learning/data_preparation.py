import glob
import os
import sys
import librosa
import numpy as np
from torch.utils.data import DataLoader,Dataset

# Path to pretrained weights
pt_file= '/home/aditthapron/WASH_work/SincNet/weight/model_Librispeech.pkl'
output_folder='exp/SincNet_coelho_transfer_learning/'

#signal length
fs=16000
cw_len=200
cw_shift=100

# chunk_n * cw_shift + (cw_len-cw_shift) needs to equal 4000ms (4s)
chunk_n=int( (4000-cw_len+cw_shift) / cw_shift)

wlen=int(fs*cw_len/1000.00)
wshift=int(fs*cw_shift/1000.00)

#Model hyper parameter
lr=0.001
batch_size=4
N_epochs=2000

N_eval_epoch=2
seed=42
Batch_dev=4
alph = 0.22


#Testing data are excluded from these directories
path = '/home/aditthapron/WASH_work/TBIBank/tbi/English/Coelho/'
tran_healthy = np.array(sorted(glob.glob(path + 'transcript/N/*.cha')))
healthy = np.array(sorted(glob.glob(path+'N/*.wav')))

path = '/home/aditthapron/WASH_work/TBIBank/tbi/English/Coelho/'
tran_TBI = np.array(sorted(glob.glob(path + 'transcript/TB/*.cha')))
TBI = np.array(sorted(glob.glob(path+'TB/*.wav')))

np.random.seed(seed)
healthy_select = np.arange(len(healthy))
healthy_select = np.delete(healthy_select,[23])
tbi_select = np.arange(len(TBI))
tbi_select = np.delete(tbi_select,[26, 29, 40, 44, 51])
np.random.shuffle(healthy_select)
np.random.shuffle(tbi_select)

#train/test split
i=int(sys.argv[1])
#number of fold in 10-fold training.

healthy_val=healthy_select[len(healthy_select)//10*i:len(healthy_select)//10*(i+1)]
healthy_train=healthy_select[~np.isin(healthy_select,healthy_val)]
tbi_val=tbi_select[len(tbi_select)//10*i:len(tbi_select)//10*(i+1)]
tbi_train=tbi_select[~np.isin(tbi_select,tbi_val)]

# test list
wav_lst_tr = np.hstack([healthy[healthy_train] , TBI[tbi_train]])
tran_lst_tr = np.hstack([tran_healthy[healthy_train] , tran_TBI[tbi_train]])
wav_lst_te = np.hstack([healthy[healthy_val] , TBI[tbi_val]])
tran_lst_te = np.hstack([tran_healthy[healthy_val] , tran_TBI[tbi_val]])
snt_tr=len(wav_lst_tr)
snt_te=len(wav_lst_te)

# Folder creation
try:
    os.stat(output_folder)
except:
    os.mkdir(output_folder) 

### Source dataset
path_WOZ = '/home/aditthapron/WASH_work/WOZ/wav/processed/'
files_WOZ = glob.glob(path_WOZ+'/*.wav')
np.random.seed(seed)
WOZ_val=files_WOZ[len(files_WOZ)//10*i:len(files_WOZ)//10*(i+1)]
WOZ_train=files_WOZ[~np.isin(np.arange(len(files_WOZ)),WOZ_val)]

def cha_read(file):
    # text = []
    time = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('*'):
                try:
                    if line.find('PAR')!=-1 and line.find('\x15')!=-1:
                        # text.append(line.split('\x15')[0].split('PAR:\t')[1])
                        time.append(line.split('\x15')[1])
                except:
                    pass
    return [s.split('_') for s in time]

class Dataset_(Dataset):
    def __init__(self, batch_size,wav_lst,N_snt,wlen,fact_amp,tran_lst):

        self.batch_size = batch_size
        self.wav_lst = wav_lst
        self.N_snt = N_snt
        self.wlen = wlen
        self.fact_amp = fact_amp
        self.tran_lst  = tran_lst
        self.chunk_n = 39

    def __len__(self):
        return len(self.wav_lst)

    def __getitem__(self, idx):
        snt_id_arr=np.random.randint(self.N_snt)
        signal, fs = librosa.load(self.wav_lst[snt_id_arr],16000)
        signal=signal/np.max(np.abs(signal))
        select_ind = np.zeros(len(signal),dtype=bool)
        time = np.array(cha_read(self.tran_lst[snt_id_arr])).astype(int)
        time = (time*16000//1000)
        for t in time:
            select_ind[t[0]:t[1]]=1
        signal = signal[select_ind]
        
        snt_len=signal.shape[0]
        snt_beg=np.random.randint(snt_len-self.wlen*self.chunk_n-1)
        snt_end=snt_beg+self.wlen*self.chunk_n

        sig_batch = np.zeros((self.chunk_n,self.wlen))
        for n in range(self.chunk_n):
            sig_batch[n] = signal[snt_beg:snt_beg+self.wlen]
            snt_beg += wshift

        #adding some noise
        sig_batch=sig_batch*np.random.uniform(1.0-self.fact_amp,1+self.fact_amp,1)

        if self.wav_lst[snt_id_arr].split('/')[-2] =='TB':
            lab_batch = 1 
        else:
            lab_batch = 0

        return sig_batch,lab_batch,self.wav_lst[snt_id_arr].split('/')[-1]

class Dataset_WOZ(Dataset):
    def __init__(self, batch_size,wav_lst,N_snt,wlen,fact_amp):
        self.batch_size = batch_size
        self.wav_lst = wav_lst
        self.N_snt = N_snt
        self.wlen = wlen
        self.fact_amp = fact_amp
        self.chunk_n = 39

    def __len__(self):
        return len(self.wav_lst)

    def __getitem__(self, idx):
        snt_id_arr=np.random.randint(self.N_snt)
        signal, fs = librosa.load(self.wav_lst[snt_id_arr],16000)
        signal=signal/np.max(np.abs(signal))
               
        snt_len=signal.shape[0]
        snt_beg=np.random.randint(snt_len-self.wlen*self.chunk_n-1)
        snt_end=snt_beg+self.wlen*self.chunk_n

        sig_batch = np.zeros((self.chunk_n,self.wlen))
        for n in range(self.chunk_n):
            sig_batch[n] = signal[snt_beg:snt_beg+self.wlen]
            snt_beg += wshift

        #adding some noise
        sig_batch=sig_batch*np.random.uniform(1.0-self.fact_amp,1+self.fact_amp,1)

        lab_batch = float(self.wav_lst[snt_id_arr].split('/')[-2])

        return sig_batch,lab_batch,self.wav_lst[snt_id_arr].split('/')[-1]