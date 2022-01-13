from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch import optim
import torch.nn as nn
import torch
import os
import numpy as np
from logging_helper import Logger
import utils
from config import HParams
hp = HParams()
logger_instance = Logger('model')
logger = logger_instance.logger


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# adaptive lr
def lr_decay(optimizer):
    """Decay learning rate by a factor of lr_decay"""
    for param_group in optimizer.param_groups:
        if param_group['lr'] > hp.min_lr:
            param_group['lr'] *= hp.lr_decay
    return optimizer

# encoder and decoder modules


class EncoderRNN(nn.Module):
    def __init__(self):
        super(EncoderRNN, self).__init__()
        # bidirectional lstm:
        if hp.use_recurrent_dropout and hp.is_training:
            # active dropout:
            self.lstm = nn.LSTM(5, hp.enc_hidden_size, bidirectional=True,
                                dropout=hp.recurrent_dropout_prob, batch_first=True,)
        else:
            self.lstm = nn.LSTM(5, hp.enc_hidden_size,
                                bidirectional=True, batch_first=True,)

        self.fc = nn.Linear(hp.enc_hidden_size * 2, hp.output_dim)

    def forward(self, inputs, batch_size, hidden_cell=None):
        if hidden_cell is None:
            # then must init with zeros
            hidden = torch.zeros(2, batch_size, hp.enc_hidden_size).to(
                device)  # num_layers * num_directions, batch, hidden_size
            cell = torch.zeros(2, batch_size, hp.enc_hidden_size).to(device)
            hidden_cell = (hidden, cell)
        _, (hidden, cell) = self.lstm(inputs, hidden_cell)
        # hidden is (2, batch_size, hidden_size), we want (batch_size, 2*hidden_size):
        hidden_forward, hidden_backward = torch.split(hidden, 1, 0)
        hidden_cat = torch.cat(
            [hidden_forward.squeeze(0), hidden_backward.squeeze(0)], 1)

        # fc - sigmoid
        output = self.fc(hidden_cat)
        output = F.softmax(output, dim=1)

        return output


class Model():
    def __init__(self):
        self.encoder = EncoderRNN().to(device)
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), hp.lr)
        self.eta_step = hp.eta_min

    def train(self, epoch, batch_iter):
        self.encoder.train()
        total_loss = 0

        for batch_idx, batch in enumerate(batch_iter):

            # load batch data:
            _3_strokes, pad_5_strokes, sequence_lengths, labels = batch
            pad_5_strokes = torch.from_numpy(pad_5_strokes).float()
            labels = torch.tensor(labels).long()
            batch_size = pad_5_strokes.size(0)

            # to cuda:
            inputs = pad_5_strokes.to(device)
            true_labels = labels.to(device)

            # encode:
            output = self.encoder(inputs, batch_size)

            # compute losses:
            loss = F.cross_entropy(output, true_labels)
            total_loss += loss

            # gradient step
            loss.backward()

            # gradient cliping
            nn.utils.clip_grad_norm_(self.encoder.parameters(), hp.grad_clip)
            # optim step
            self.encoder_optimizer.step()
            self.encoder_optimizer = lr_decay(self.encoder_optimizer)

        # some print and save:
        if epoch % 1 == 0:
            logger.info(f'epoch={epoch}, loss={total_loss/batch_idx:.3f}')

        if epoch % 1 == 0:
            self.save(epoch)

    def eval(self, batch_iter):
        self.encoder.eval()
        predictions, true_labels = [], []

        for batch_idx, batch in enumerate(batch_iter):
            # load batch data:
            _3_strokes, pad_5_strokes, sequence_lengths, labels = batch
            pad_5_strokes = torch.from_numpy(pad_5_strokes).float()
            labels = torch.tensor(labels).long()
            batch_size = pad_5_strokes.size(0)

            # to cuda:
            inputs = pad_5_strokes.to(device)

            # encode:
            with torch.no_grad():
                output = self.encoder(inputs, batch_size)

            pred = output.max(1)[1]
            predictions.append(pred.to('cpu').numpy())
            true_labels.append(labels.to('cpu').numpy())

        predictions = [
            item for sublist in predictions for item in sublist]  # y_pred
        true_labels = [
            item for sublist in true_labels for item in sublist]  # y_true

        return true_labels, predictions

    def save(self, epoch):
        #sel = np.random.rand()
        torch.save(self.encoder.state_dict(),
                   os.path.join(hp.checkpoint, f'encoderRNN_epoch_{epoch}.pth'))

    def load(self, model_path):
        saved_encoder = torch.load(model_path)
        self.encoder.load_state_dict(saved_encoder)


def predict(strokes, model_epoch=None):
    # data to dataloader:
    processed_strokes = utils.processed_strokes(strokes, hp)
    pred_set = utils.DataLoader(
        strokes=[processed_strokes], labels=[0],
        batch_size=1,
        max_seq_length=hp.max_seq_len,
        random_scale_factor=0.0,
        augment_stroke_prob=0.0,
        sampler=True, shuffle=False)
    pred_set.normalize()

    # load model:
    pred_model = Model()
    if model_epoch is None:
        # Todo: get max epoch model_path
        model_path = [f for f in os.listdir(
            hp.checkpoint) if f.find('pth') > -1]
        model_path = model_path[0]
        print(f'Model Load:{model_path}')
    else:
        model_path = f'encoderRNN_epoch_{model_epoch}.pth'
    pred_model.load(os.path.join(hp.checkpoint, model_path))

    # predict:
    y_tru, y_pred = pred_model.eval(pred_set)
    return y_pred[0]


if __name__ == "__main__":

    # load data:
    train_set, valid_set, test_set, model_params, eval_model_params, sample_model_params = utils.load_dataset(
        hp)

    # train model:
    train_model = Model()
    for epoch in range(hp.epoch):
        hp = model_params
        train_model.train(epoch, train_set)

        if epoch % 1 == 0:
            hp = eval_model_params
            y_tru, y_pred = train_model.eval(valid_set)
            logger.info(
                f"""[epoch={epoch}] accuracy:{accuracy_score(y_tru, y_pred):.3f} | precision:{precision_score(y_tru, y_pred, average='micro'):.3f}""")

    '''
    model.load('encoder.pth')
    '''
