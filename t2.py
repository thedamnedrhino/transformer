import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import warnings
import spacy
import datetime
import torchtext.data.batch
from nltk.translate.bleu_score import sentence_bleu
from torch.autograd import Variable
from torchtext import data, datasets



import io


BATCH_SIZE = 12000
TRAIN_EPOCHS = 10
# ++++++ BEGIN main functionalities ++++++++++
def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# ----- END main functionalities -----------------

# ++++++ BEGIN Code utils +++++++++++++++++++++++++++++++++
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# ------ END Code utils --------------------------------

# ++++++ BEGIN functional utils ++++++++++++++++++
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


# ------ END functional utils -------------------


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Positional encoding gains even more importance when we're using
    multi-head attention - it's critical that we take into account
    where each attention is coming from
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


# ++++++++++++++++++++ BEGIN training ++++++++++++++++++++++
class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0.
    tokens = 0
    total_elapsed = 0.
    all_tokens_per_sec = []
    for i, batch in enumerate(data_iter):
        ntokens = batch.ntokens.float()
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, ntokens)
        total_loss += loss
        total_tokens += ntokens
        tokens += ntokens
        if True or i % 50 == 1:
            elapsed = time.time() - start
            tokens_per_sec = tokens.float() / elapsed
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / batch.ntokens.float(), tokens_per_sec))
            total_elapsed += elapsed
            all_tokens_per_sec.append(tokens_per_sec)
            start = time.time()
            tokens = 0
    return total_loss / total_tokens, total_elapsed, sum(all_tokens_per_sec) / len(all_tokens_per_sec)


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x: torch.Tensor, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()  # type: torch.Tensor
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1).long(), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0 and sum(mask.size()) > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


# ---------------- END training -----------------------

# +++++++++++++++ BEGIN test run ++++++++++++++++++++++
class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm.float()
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm.float().item()


def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1

        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


def test_run():
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    times = []
    all_tps = []  # array of tokens per sec
    losses = []
    for epoch in range(10):
        model.train()
        loss, t, tokens_per_sec = run_epoch(data_gen(V, 30, 10), model,
                                            SimpleLossCompute(model.generator, criterion, model_opt))
        times.append(t)
        all_tps.append(tokens_per_sec)
        losses.append(loss)
        model.eval()
        # model.train()
        # print(run_epoch(data_gen(V, 30, 5), model,
        #                 SimpleLossCompute(model.generator, criterion, None)))
        # print(time.time() - t)
        # model.eval()

    # model.eval()
    print("average time: {}".format(sum(times) / len(times)))
    print("average tps: {}".format(sum(all_tps) / len(all_tps)))
    print("average last 3 losses: {}".format(sum(losses[:-3]) / 3))
    print("last loss: {}".format(losses[len(losses) - 1]))
    src = Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))
    src_mask = Variable(torch.ones(1, 1, 10))
    print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))


# --------------- END test run ------------------------


# --------------- BEGIN real training --------------------
class MyIterator(data.Iterator):
    def create_batches(self):
        if not hasattr(self, 'limit'):
            self.limit = None
        if self.train:
            def pool(d, random_shuffler):
                for p in torchtext.data.batch(d, self.batch_size * 100):
                    p_batch = torchtext.data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    yielded = 0
                    for b in random_shuffler(list(p_batch)):
                        if self.limit is None:
                            yield b
                        else:
                            if yielded >= self.limit:
                                break
                            yield b
                            yielded += self.batch_size

            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            yielded = 0
            batch = torchtext.data.batch(self.data(), self.batch_size, self.batch_size_fn)
            for b in batch:
                if self.limit is not None:
                    if yielded >= self.limit:
                        break
                    yielded += self.batch_size
                self.batches.append(sorted(b, key=self.sort_key))

    def set_limit(self, limit):
        self.limit = limit


def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)


class MultiGPULossCompute:
    "A multi-gpu loss compute and train function."

    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion,
                                               devices=[d.index for d in devices])
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size

    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator,
                                          devices=[d.index for d in self.devices])
        out_scatter = nn.parallel.scatter(out,
                                          target_gpus=[d.index for d in self.devices])
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets,
                                      target_gpus=[d.index for d in self.devices])

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions
            out_column = [[Variable(o[:, i:i + chunk_size].data,
                                    requires_grad=self.opt is not None)]
                          for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss.
            y = [(g.contiguous().view(-1, g.size(-1)),
                  t[:, i:i + chunk_size].contiguous().view(-1))
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # Sum and normalize loss
            l = nn.parallel.gather(loss,
                                   target_device=self.devices[0].index)
            if i % 50 == 1:
                print('\\\\\\\\\\\\\\\\')
                print(torch.cuda.memory_allocated(0))
                print(torch.cuda.memory_allocated(1))
            l = l.sum()[0] / normalize
            # total += l.data[0]
            total += l.item()
            if i % 50 == 1:
                print('++++++++++++++++')
                print(torch.cuda.memory_allocated(0))
                print(torch.cuda.memory_allocated(1))

            # Backprop loss to output of transformer
            if self.opt is not None:
                l.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

            if i % 50 == 1:
                print('-----------------------------')
                print(torch.cuda.memory_allocated(0))
                print(torch.cuda.memory_allocated(1))
        # Backprop all loss through transformer.
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad,
                                    target_device=self.devices[0].index)
            o1.backward(gradient=o2)
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return total * normalize


BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"


def get_tokenizer(lang):
    if lang is 'en':
        spacy_en = spacy.load('en')
        def tokenize_en(text):
            words =  [tok.text for tok in spacy_en.tokenizer(text)]
            return words
        return tokenize_en
    elif lang is 'de':
        spacy_de = spacy.load('de')
        def tokenize_de(text):
            return [tok.text for tok in spacy_de.tokenizer(text)]

        return tokenize_de
    elif lang is 'fr':
        spacy_fr = spacy.load('fr')
        def tokenize_fr(text):
            return [tok.text for tok in spacy_fr.tokenizer(text)]
        return tokenize_fr
    elif lang is 'es' or 'eu':
        spacy_es = spacy.load('es')
        def tokenize_es(text):
            return [tok.text for tok in spacy_es.tokenizer(text)]
        return tokenize_es
    return str.split


def load_data(lang1 = 'de', lang2 = 'en', directory = None):

    lang1_tokenizer = get_tokenizer(lang1)
    lang2_tokenizer = get_tokenizer(lang2)

    SRC = data.Field(tokenize=lang1_tokenizer, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=lang2_tokenizer, init_token=BOS_WORD,
                     eos_token=EOS_WORD, pad_token=BLANK_WORD)

    MAX_LEN = 100

    if directory:
        train, val = datasets.TranslationDataset(path = directory, exts=('.'+lang1, '.'+lang2), fields=(SRC,TGT),
                                                 filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(
                                                     vars(x)['trg']) <= MAX_LEN).split()
    else:
        train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'),
                                                 fields=(SRC, TGT),
                                                 filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN)
    MIN_FREQ = 2
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)
    return train, val, SRC, TGT  # todo  find out exactly what each of these variables are


def train_multi_gpu(num_gpu, output_model, in_model=None, data = load_data(), limit = None):

    device_ids = list(range(num_gpu))
    devices = [torch.device("cuda:{}".format(i)) for i in device_ids]
    train, val, SRC, TGT = data
    pad_idx = TGT.vocab.stoi[BLANK_WORD]
    if in_model:
        model = load_model(in_model, len(SRC.vocab), len(TGT.vocab))
    else: 
        model = make_model(len(SRC.vocab), len(TGT.vocab))
        model.cuda()
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda()
    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=devices[0],
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=devices[0],
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)

    if limit is not None:
        train_iter.set_limit(limit)
        valid_iter.set_limit(limit)

    model_par = nn.DataParallel(model, device_ids=device_ids)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    import time
    t = time.time()

    for epoch in range(TRAIN_EPOCHS):
        print("Epoch {}:".format(epoch))
        model_par.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter),
                  model_par,
                  MultiGPULossCompute(model.generator, criterion,
                                      devices=devices, opt=model_opt))
        model_par.eval()
        t = time.time()
        loss, elasped_time, avg_tokens_sec  = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
                         model_par,
                         MultiGPULossCompute(model.generator, criterion,
                                             devices=devices, opt=None))
        print("Validation loss: {} \n Time for eval: {}".format(loss.item(),elasped_time))
        torch.save(model.state_dict(), output_model)
    return model

def load_model(model_file, src_len, tgt_len):
    model = make_model(src_len, tgt_len)
    with io.open(model_file, "rb") as file:
        model.load_state_dict(torch.load(file))
    model.cuda()
    return model


def validate(model, num_gpu = torch.cuda.device_count(), data = load_data()):
    device_ids = list(range(num_gpu))
    devices = [torch.device("cuda:{}".format(i)) for i in device_ids]

    train, val, SRC, TGT = data
    model.eval()
    BATCH_SIZE = 12000
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=devices[0],
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)
    scores = []
    for i, batch in enumerate(valid_iter):
        src = batch.src.transpose(0, 1)[:1]
        src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
        out = greedy_decode(model, src, src_mask,
                            max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
        print("Translation:", end="\t")
        out_list = []
        tar_list = []
        for i in range(1, out.size(1)):
            sym = TGT.vocab.itos[out[0, i]]
            if sym == "</s>": break
            out_list.append(sym)

        print(' '.join(out_list))
        print()
        print("Target:", end="\t")
        for i in range(1, batch.trg.size(0)):
            sym = TGT.vocab.itos[batch.trg.data[i, 0]]
            if sym == "</s>": break
            tar_list.append(sym)
        print(' '.join(tar_list))
        print()
        score = sentence_bleu([out_list],tar_list)
        scores.append(score)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print("score: ", score)
    print("average score: {}".format(sum(scores) / float(len(scores))))


# +++++++++++++++ END real training ++++++++++++++++++++
if __name__ == '__main__':
    import optparse
    optparser = optparse.OptionParser()
    # optparser.add_option("-d", "--datadir", dest="datadir", default="data", help="data directory (default=data)")
    optparser.add_option("-l", "--limit", type = "int", dest="limit", default=None, help="limit the number of training and evaluation samples")
    optparser.add_option("-o", "--model-output", dest="model_output", default=str(datetime.date.today()), help="output file for the model, defaults to model-{random_number}")
    optparser.add_option("-s", "--batch-size", dest="batch_size", default=12000, help="batch size, default: 12000")
    optparser.add_option("-e", "--epochs", dest="epochs", default=10, help="number of epochs for training")
    optparser.add_option("-b", "--basic", dest="basic", default=False, action='store_true', help="whether to use the auto-generated one-to-one integer training, this is just a sanity test")
    optparser.add_option("-v", "--validate", dest="validate", default=False, action='store_true', help="run the model found in the file with dataset")
    optparser.add_option("-i", "--inputmodel", dest="model_input", default=None, help="load model to input")
    optparser.add_option("-d", "--data-dir", dest="data_dir", default=None, help="Data directory to load dataset from")
    optparser.add_option("-1", "--lang1", dest="lang1", default=None, help="Input language extention name")
    optparser.add_option("-2", "--lang2", dest="lang2", default=None, help="Input language extention name")
    (opts, _) = optparser.parse_args()
    BATCH_SIZE=int(opts.batch_size)
    if opts.data_dir:
        print("loading data from {}".format(opts.data_dir))
        data = load_data(lang1=opts.lang1, lang2=opts.lang2, directory=opts.data_dir)
    else:
        data = load_data()

    if opts.basic:
        test_run()
    else:
        limit = opts.limit
        TRAIN_EPOCHS = int(opts.epochs)
        model_output_file = opts.model_output
        model_input_file = opts.model_input
        model = train_multi_gpu(torch.cuda.device_count(),
                        model_output_file, model_input_file,
                        data=data,
                        limit=int(limit) if limit is not None else limit)
        if opts.validate:
            validate(model, data=data)


