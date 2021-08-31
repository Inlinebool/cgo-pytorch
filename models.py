from queue import PriorityQueue

import torch
from loguru import logger
from torch import nn
from torch.nn.utils import weight_norm
from torch.nn.utils.rnn import pack_padded_sequence


class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 do_batch_norm=False,
                 do_weight_norm=False,
                 dropout=None,
                 activation=None):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        if do_batch_norm and do_weight_norm:
            logger.warning(
                "batch norm and weight norm enabled at the same time!")
        if do_weight_norm:
            logger.info("weight norm enabled.")
            self.linear = weight_norm(self.linear)
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        if do_batch_norm:
            logger.info("batch norm enabled.")
            self.bn = nn.BatchNorm1d(out_dim)
        else:
            self.bn = None
        if not activation:
            logger.warning("no activation function!")
        self.activation = activation

    def forward(self, input):
        out = self.linear(input)
        if self.bn:
            out = self.bn(out)
        if self.activation:
            out = self.activation(out)
        if self.dropout:
            out = self.dropout(out)
        return out


class ObjectClassifier(torch.nn.Module):
    def __init__(self,
                 feature_dim,
                 att_dim,
                 linear_dims,
                 num_of_classes,
                 dropout=0.5):
        super().__init__()
        self.fc_att0 = FullyConnectedLayer(feature_dim[1],
                                           att_dim,
                                           do_weight_norm=True,
                                           dropout=dropout,
                                           activation=nn.ReLU())
        self.fc_att1 = FullyConnectedLayer(att_dim,
                                           1,
                                           do_weight_norm=True,
                                           activation=nn.Softmax(dim=1))
        self.fc_pred_first = FullyConnectedLayer(feature_dim[1],
                                                 linear_dims[0],
                                                 do_batch_norm=True,
                                                 activation=nn.ReLU())
        self.fc_pred_last = FullyConnectedLayer(linear_dims[-1],
                                                num_of_classes)
        self.fc_pred_middle = nn.ModuleList()
        for i in range(len(linear_dims) - 1):
            self.fc_pred_middle.append(
                FullyConnectedLayer(linear_dims[i],
                                    linear_dims[i + 1],
                                    do_batch_norm=True,
                                    activation=nn.ReLU()))

    def forward(self, inputs):
        image_features, = inputs
        att_0 = self.fc_att0(image_features)
        alphas = self.fc_att1(att_0)
        reduced_features = torch.sum(alphas * image_features, dim=1)
        h = self.fc_pred_first(reduced_features)
        for layer in self.fc_pred_middle:
            h = layer(h)
        h = self.fc_pred_last(h)
        return h


class LanguageModel(torch.nn.Module):
    def __init__(self,
                 vocal_size,
                 embed_dim,
                 hidden_dim,
                 feature_dim,
                 att_dim,
                 device,
                 dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocal_size, embed_dim, padding_idx=0)

        self.hidden_dim = hidden_dim
        self.lstm_att = nn.LSTMCell(hidden_dim + feature_dim[1] + embed_dim,
                                    hidden_dim)

        self.fc_att_hidden = FullyConnectedLayer(
            hidden_dim,
            att_dim,
            do_weight_norm=True,
        )
        self.fc_att_image = FullyConnectedLayer(
            feature_dim[1],
            att_dim,
            do_weight_norm=True,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc_att_out = FullyConnectedLayer(att_dim,
                                              1,
                                              do_weight_norm=True,
                                              activation=nn.Softmax(dim=1))

        self.lstm_lang = nn.LSTMCell(hidden_dim + feature_dim[1], hidden_dim)
        self.fc_pred1 = FullyConnectedLayer(
            hidden_dim,
            vocal_size,
            do_weight_norm=True,
        )

        self.device = device

    def step(self, image_features, prev_pred, att_h0, att_c0, lang_h0,
             lang_c0):
        mean_features = torch.mean(image_features, dim=1)
        input_embed = self.embedding(prev_pred)
        att_input = torch.cat([lang_h0, mean_features, input_embed], dim=1)

        att_h1, att_c1 = self.lstm_att(att_input, (att_h0, att_c0))

        att_h = self.fc_att_hidden(att_h1).unsqueeze(1)
        att_v = self.fc_att_image(image_features)
        att_in = self.dropout(self.relu(att_h + att_v))
        alphas = self.fc_att_out(att_in)
        reduced_features = torch.sum(alphas * image_features, dim=1)

        lang_input = torch.cat([reduced_features, att_h1], dim=1)
        lang_h1, lang_c1 = self.lstm_lang(lang_input, (lang_h0, lang_c0))

        pred = self.fc_pred1(self.dropout(lang_h1))

        return pred, att_h1, att_c1, lang_h1, lang_c1

    def init_hidden(self, batch_size):
        att_h0 = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        att_c0 = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        lang_h0 = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        lang_c0 = torch.zeros(batch_size, self.hidden_dim, device=self.device)

        return att_h0, att_c0, lang_h0, lang_c0

    def forward(self, inputs):
        image_features, seq, seq_length = inputs
        seq_length += 1  # won't input <end>
        seq_length = seq_length.cpu().to(torch.int64)
        packed_seq = pack_padded_sequence(seq,
                                          seq_length,
                                          batch_first=True,
                                          enforce_sorted=False)
        batchs = torch.split(packed_seq.data, packed_seq.batch_sizes.tolist())
        sorted_features = image_features[packed_seq.sorted_indices]
        att_h, att_c, lang_h, lang_c = self.init_hidden(len(seq))
        preds = []
        for batch in batchs:
            batch_size = len(batch)
            batch_features = sorted_features[:batch_size]
            batch_att_h0 = att_h[:batch_size]
            batch_att_c0 = att_c[:batch_size]
            batch_lang_h0 = lang_h[:batch_size]
            batch_lang_c0 = lang_c[:batch_size]
            pred, att_h1, att_c1, lang_h1, lang_c1 = self.step(
                batch_features, batch, batch_att_h0, batch_att_c0,
                batch_lang_h0, batch_lang_c0)
            preds.append(pred)
            att_h = att_h1
            att_c = att_c1
            lang_h = lang_h1
            lang_c = lang_c1
        return torch.cat(preds)

    def decode(self, inputs, end, beam=1):
        # in forward(), inputs are in batches, seq is the gold label
        # in decode(), batch size is always 1, seq is the guiding sequence.
        # for lstm-left, guiding sequence is the guiding object, and asumed
        # sequence, in reverse order.
        # for lstm-right, guiding sequence is the output of lstm-left, in
        # reverse order.
        image_features, seq, seq_length = inputs
        seq_length += 1
        att_h, att_c, lang_h, lang_c = self.init_hidden(1)
        # use log softmax, so that we can add the scores (instead of multiply)
        softmax = nn.LogSoftmax(dim=0)
        top_k = PriorityQueue()
        ended = 0
        guiding_length = seq_length[0]

        # prepare model states with input sequence
        for i in range(guiding_length - 1):
            pred, att_h, att_c, lang_h, lang_c = self.step(
                image_features, seq[0, i].view(-1), att_h, att_c, lang_h,
                lang_c)

        # start actual decoding (no guidng sequence left)
        pred, att_h, att_c, lang_h, lang_c = self.step(
            image_features, seq[0, guiding_length - 1].view(-1), att_h, att_c,
            lang_h, lang_c)
        pred = pred.view(-1)
        pred = softmax(pred)
        top_k_pred = torch.argsort(pred, descending=True)[:beam]
        for candidate in top_k_pred:
            # negate the score, because priority queue is in ascending order
            top_k.put((-pred[candidate], [candidate]))

        decoded_length = 0
        while decoded_length < 50:
            new_top_k = PriorityQueue()
            ended = 0
            for _ in range(beam):
                score, pred_seq = top_k.get()
                if pred_seq[-1] == end:
                    new_top_k.put((score, pred_seq))
                    ended += 1
                    continue
                pred, att_h, att_c, lang_h, lang_c = self.step(
                    image_features, pred_seq[-1].view(-1), att_h, att_c,
                    lang_h, lang_c)
                pred = pred.view(-1)
                pred = softmax(pred)
                top_k_pred = torch.argsort(pred, descending=True)[:beam]
                for candidate in top_k_pred:
                    new_top_k.put(
                        (-pred[candidate] + score, pred_seq + [candidate]))
            top_k = new_top_k
            decoded_length += 1
            if ended == beam:
                break
        _, final_pred_seq = top_k.get()
        return [x.item() for x in final_pred_seq]
