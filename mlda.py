import logging

import numpy as np
import scipy as sp

import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MLDA:

    def __init__(self, n_topic: int, n_iter=2000, alpha=0.1, eta=0.01, random_state=None, refresh=10):
        """

        :param n_topic: 主题数目
        :param n_iter: 迭代次数
        :param alpha: 文档主题分布超参数
        :param eta: 主题单词分布超参数
        :param random_state: 随机种子
        :param refresh: 循环多少次输出当前日志
        """
        self.n_topic = n_topic
        self.n_iter = n_iter
        self.alpha = alpha
        self.eta = eta
        # if random_state is None, check_random_state(None) does nothing
        self.random_state = random_state

        if alpha <= 0 or eta <= 0:
            raise ValueError('alpha and eta must be greater than zero')

        # random number that are reused
        rng = utils.check_random_state(random_state)
        self._rands = rng.rand(1024 ** 2 // 8)  # 1MiB of random variates

    def fit(self, corpus):
        """

        :param corpus: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features. Sparse matrix allowed.
        :return:
        """
        random_state = utils.check_random_state(self.random_state)
        rands = self._rands.copy()

        self._initialize(corpus) # 初始化所有有关信息
        for iter in range(self.n_iter):

            random_state.shuffle(rands)
            if iter % self.re

    def _initialize(self, X: np.array):
        n_doc, n_term = X.shape
        n_word = int(X.sum())

        n_topics = self.n_topic
        n_iter = self.n_iter
        logger.info('n_documents: {}'.format(n_doc))
        logger.info('n_terms: {}'.format(n_term))
        logger.info('n_words: {}'.format(n_word))
        logger.info('n_topics: {}'.format(n_topics))
        logger.info('n_iter: {}'.format(n_iter))

        self.nzw_ = nzw_ = np.zeros((n_topics, n_term), dtype=np.int)  # 主题单词统计量
        self.ndz_ = ndz_ = np.zeros((n_doc, n_topics), dtype=np.int)  # 文档主题统计量
        self.nz_ = nz_ = np.zeros(n_topics, dtype=np.int)

        # WS, DS. ZS 分别表示单词标号集合，文档标号集合，赋予主题标号集合
        self.WS, self.DS = WS, DS = utils.matrix_to_lists(X)
        self.ZS = ZS = np.random.randint(n_topics, size=n_word)

        for i in range(n_word):
            w, d, z = WS[i], DS[i], ZS[i]
            nzw_[z, w] += 1
            ndz_[d, z] += 1
            nz_[z] += 1


if __name__ == '__main__':
    model = MLDA(n_topic=20, random_state=10)
    corpus = np.array([[1, 3, 4, 5, 5], [3, 0, 0, 1, 2]], dtype=np.int)
    model.fit(corpus)
