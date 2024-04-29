import time
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "../models/transformer"))

import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from .trainer import Trainer
from src.models.transformer import Transformer, Informer, Reformer, Flowformer, Flashformer, \
    iTransformer, iInformer, iReformer, iFlowformer, iFlashformer
from src.models.transformer.utils.tools import adjust_learning_rate

MODEL_DICT = {
    'Transformer': Transformer,
    'Informer': Informer,
    'Reformer': Reformer,
    'Flowformer': Flowformer,
    'Flashformer': Flashformer,
    'iTransformer': iTransformer,
    'iInformer': iInformer,
    'iReformer': iReformer,
    'iFlowformer': iFlowformer,
    'iFlashformer': iFlashformer,
}

def stamp_transform(timestamps):
    df_stamp = pd.DataFrame(timestamps, columns=["date"])
    # df_stamp['date'] = pd.to_datetime(df_stamp.date)
    # df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
    # df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
    # df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday, 1)
    # df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
    # df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
    # df_stamp['second'] = df_stamp.date.apply(lambda row: row.second, 1)

    # print(df_stamp)
    # df_stamp.drop(['date'], 1)
    # print(df_stamp)
    return df_stamp

# Transform the original GResearchCryptoData into GResearchCryptoTransformerData
# GResearchCryptoTransformerData
class GRCTData(Dataset):
    def __init__(self, data, params) -> None:
        self.params = params
        self.seq_len, self.label_len, self.pred_len = self.params.seq_len, self.params.label_len, self.params.pred_len
        self.step_size = 60 
        self._transformerize(data)

    def _transformerize(self, data):
        self.feats, self.targets = data.feats, data.targets
        self.timestamps = stamp_transform(self.feats.index)

    def __len__(self):
        return (len(self.feats) - self.seq_len - self.pred_len + 1) // self.step_size
    
    def __getitem__(self, index):
        rand_start = np.random.randint(0, self.step_size)
        s_begin = min((index * self.step_size + rand_start),  (len(self.feats) - self.seq_len - self.pred_len + 1))
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.feats.iloc[s_begin:s_end].to_numpy()
        seq_y = self.targets.iloc[r_begin:r_end].to_numpy()
        seq_x_mark = self.timestamps.iloc[s_begin:s_end].to_numpy()
        seq_y_mark = self.timestamps.iloc[r_begin:r_end].to_numpy()

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
# GResearchCryptoTransformerInfnerenceDataseq_y = self.targets.iloc[r_begin:r_end].to_numpy()
class GRCTIData(Dataset):
    def __init__(self, data, params) -> None:
        self.params = params
        self.seq_len, self.label_len, self.pred_len = self.params.seq_len, self.params.label_len, self.params.pred_len
        self.feats, self.targets = data
        self.timestamps = stamp_transform(self.feats.index)

    def __len__(self):
        return (len(self.feats) - self.seq_len - self.label_len) // self.pred_len 
    
    def __getitem__(self, index):
        s_begin = index * self.pred_len
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.feats.iloc[s_begin:s_end].to_numpy()
        seq_y = self.targets.iloc[r_begin:r_end].to_numpy()
        seq_x_mark = self.timestamps.iloc[s_begin:s_end].to_numpy()
        seq_y_mark = self.timestamps.iloc[r_begin:r_end].to_numpy()

        return seq_x, seq_y, seq_x_mark, seq_y_mark

def data_provider(data, params, shuffle, drop_last, is_inference=False):
    if is_inference:
        t_data = GRCTIData(data, params)
    else:
        t_data = GRCTData(data, params)
    data_loader = DataLoader(
                    t_data,
                    batch_size=params.batch_size,
                    shuffle=shuffle,
                    num_workers=params.num_workers,
                    drop_last=drop_last)
    return t_data, data_loader

class TransformerTrainer(Trainer):
    def __init__(self, h_params) -> None:
        model_type = h_params.model
        self.params = h_params
        print("Transformer Params:", h_params)
        self.device = self._acquire_device()
        self.model = self._build_model(model_type).to(self.device)
        self.optimizer = self._select_optimizer()

    def _acquire_device(self):
        if self.params.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.params.gpu) if not self.params.use_multi_gpu else self.params.devices
            device = torch.device('cuda:{}'.format(self.params.gpu))
            print('Use GPU: cuda:{}'.format(self.params.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    def _build_model(self, model_type):
        model = MODEL_DICT[model_type].Model(self.params).float()

        if self.params.use_multi_gpu and self.params.use_gpu:
            model = nn.DataParallel(model, device_ids=self.params.device_ids)
        return model

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.params.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def train(self, data, params, valid_data=None):
        train_data, train_loader = data_provider(data, self.params, shuffle=True, drop_last=False)
        train_steps = len(train_loader)
        criterion = self._select_criterion()

        if self.params.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        time_now = time.time()
        for epoch in range(self.params.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                self.optimizer.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.params.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.params.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.params.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.params.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.params.features == 'MS' else 0
                        outputs = outputs[:, -self.params.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.params.pred_len:, f_dim:].to(self.device)
                        loss = 1e3 * criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    # print("Input Sizes:", batch_x.size(), batch_x_mark.size(), dec_inp.size(), batch_y_mark.size())
                    if self.params.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                    
                    f_dim = -1 
                    outputs = outputs[:, -self.params.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.params.pred_len:, f_dim:].to(self.device)
                    
                    # print("Preds", outputs.size(), batch_y.size())
                    # ss
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.params.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.params.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} ".format(
                epoch + 1, train_steps, train_loss))

            adjust_learning_rate(self.optimizer, epoch + 1, self.params)
        
    def inference(self, x_data):
        inf_data, inf_loader = data_provider(x_data, self.params, shuffle=False, drop_last=False, is_inference=True)
        self.model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(inf_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.params.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.params.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.params.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.params.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.params.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 
                batch_y = batch_y[:, -self.params.pred_len:, f_dim:]
                batch_y = batch_y.detach().cpu().numpy()
                outputs = outputs[:, -self.params.pred_len:, f_dim:]
                outputs = outputs.detach().cpu().numpy()

                preds.append(outputs.flatten())
                targets.append(batch_y.flatten())

        preds = np.concatenate(preds, axis=0)
        # preds = preds.flatten
        targets = np.concatenate(targets, axis=0)
        # targets = np.array(targets)
        # targets = targets.flatten()
        # print("Preds Shape", preds.shape, targets.shape)
        return preds, targets
    