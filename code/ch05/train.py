import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from dataset import ptb
from simple_rnnlm import SimpleRnnlm


def generate(model, start_id, sample_size=20):
    word_ids = [start_id]
    model.reset_state()
    x = np.array(start_id).reshape(1, 1)
    for _ in range(sample_size):
        score = model.forward(x, None)
        p = np.exp(score[0, -1])
        p /= p.sum()
        next_id = np.random.choice(len(p), p=p)
        word_ids.append(next_id)
        x = np.array(next_id).reshape(1, 1)
    return word_ids

# 设定超参数
batch_size = 10
wordvec_size = 100
hidden_size = 100  # RNN的隐藏状态向量的元素个数
time_size = 5  # RNN的展开大小
lr = 0.1
max_epoch = 100

# 读入训练数据
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_size = 1000  # 缩小测试用的数据集
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)
xs = corpus[:-1]  # 输入
ts = corpus[1:]  # 输出（监督标签）

# 生成模型
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

trainer.fit(xs, ts, max_epoch, batch_size, time_size)
trainer.plot()