import logging

import numpy as np
import scipy as sp

import utils

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MLDA:

    def __init__(self, n_topic: int, n_iter=2000, alpha=0.1, beta=0.01,
                 random_state=None, refresh=10):
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
        self.beta = beta
        # if random_state is None, check_random_state(None) does nothing
        self.random_state = random_state
        self.refresh = refresh

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

        self._initialize(corpus)  # 初始化所有有关信息
        for iter in range(self.n_iter):

            random_state.shuffle(rands)
            if iter % self.refresh == 0:
                ll = self.loglikelihood()
                logger.info('<{}> log likelihood: {:.0f}'.format(iter, ll))

            self._sample_topics(rands)
        ll = self.loglikelihood()
        logger.info('<{}> log likelihood: {:.0f}'.format(iter, ll))

        # 计算文档主题分布和主题单词分布
        self.doc_topic_ = (self.ndz_ + self.alpha).astype(float)
        self.doc_topic_ /= np.sum(self.doc_topic_, axis=1)[:, np.newaxis]
        self.topic_word_ = (self.nzw_ + self.beta).astype(float)
        self.topic_word_ /= np.sum(self.topic_word_, axis=1)[:,np.newaxis]

        # 删除计算过程中的中间值，以节约空间
        del self.TS
        del self.DS
        del self.ZS
        return self



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

        self.nzt_ = nzt_ = np.zeros(
            (n_topics, n_term), dtype=np.int)  # 主题单词统计量
        self.ndz_ = ndz_ = np.zeros((n_doc, n_topics), dtype=np.int)  # 文档主题统计量
        self.nz_ = nz_ = np.zeros(n_topics, dtype=np.int)

        # TS, DS. ZS 分别表示单词(Term)标号集合，文档标号集合，赋予主题标号集合
        self.TS, self.DS = TS, DS = utils.matrix_to_lists(X)
        self.ZS = ZS = np.random.randint(n_topics, size=n_word)

        for i in range(n_word):
            t, d, z = TS[i], DS[i], ZS[i]
            nzw_[z, t] += 1
            ndz_[d, z] += 1
            nz_[z] += 1

    def loglikelihood(self):
        """
        Calculate complete log likelihood, log p(w,z)
        Formula used is log p(w,z) = log p(w|z) + log p(z)
        :return: log likelihood
        """
        print('hello world')

    def _sample_topics(self, rands):
        """
        sample topics base on store information
        :param rands 存储的随机数 
        """
        n_topics, n_term = self.nzt_.shape

        n_word = self.ZS.size
        TS = self.TS
        ZS = self.ZS
        DS = self.DS
        nzt_ = self.nzt_
        nz_ = self.nz_
        ndz_ = self.ndz_
        beta = self.beta
        alpha = self.alpha
        n_rand = rands.size

        for i in range(n_word):
            d, z, t = DS[i], ZS[i], TS[i]
            nzt_[z, t] -= 1
            nz_[z] -= 1
            ndz_[d, z] -= 1
            
            # 计算采样的分布
            p = (nzt_[:,t] + beta) / (nz_ + n_term * beta) * (ndz_[d] + alpha)
            p_sum = np.array([p[0: col+1] for col in range(p.size)])
            r = rands(i % n_rand) * p_sum[-1]
            new_z = np.searchsorted(p_sum, r)

            # 根据新主题，统计相关信息
            nzt_[new_z, t] += 1
            nz_[new_z] += 1
            ndz_[d, new_z] += 1
            

if __name__ == '__main__':
    model = MLDA(n_topic=20, random_state=10)
    corpus = np.array([[1, 3, 4, 5, 5], [3, 0, 0, 1, 2]], dtype=np.int)
    model.fit(corpus)
