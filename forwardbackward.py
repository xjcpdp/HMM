__author__ = 'Jiecheng Xu'
__AndrewID__ = 'jiechenx'

import numpy as np
import sys

# toy_data/toytrain.txt toy_data/toy_index_to_word.txt toy_data/toy_index_to_tag.txt hmmprior.txt hmmemit.txt hmmtrans.txt predicted.txt metrics.txt
# handout/testwords.txt handout/index_to_word.txt handout/index_to_tag.txt hmmprior.txt hmmemit.txt hmmtrans.txt predicted.txt metrics.txt


class hmm_predicter:
    alpha = None
    beta = None
    A = None
    B = None
    prior = None
    test_word = []
    index2tag = {}
    index2word = {}
    word2index = {}
    tag2index = {}
    test_tag = []
    predict_tag = []
    likelihood = []
    avg_log_likelihood = None
    accuracy = None
    log_alpha = None
    log_like = []

    def load_data(self, test_in:str, index_to_word:str, index_to_tag:str, hmmprior:str, hmmemit:str, hmmtrans:str):
        # read index_to_word file and populate the dictionary index2word
        with open(index_to_word) as word_index:
            input = word_index.readlines()
            for i in range(len(input)):
                self.index2word[input[i].strip()] = i
                self.word2index[i] = input[i].strip()

        # read index_to_tag file and populate index2tag
        with open(index_to_tag) as tag_index:
            input = tag_index.readlines()
            for i in range(len(input)):
                self.index2tag[input[i].strip()] = i
                self.tag2index[i] = input[i].strip()

        # read test_input file and encode into index form using above dictionaries
        with open(test_in) as file:
            input = file.readlines()
            for line in input:
                line = line.strip().split(' ')
                temp_word = []
                temp_tag = []
                for item in line:
                    item = item.strip().split("_")
                    temp_word.append(self.index2word[item[0]])
                    temp_tag.append(self.index2tag[item[1]])
                self.test_word.append(temp_word)
                self.test_tag.append(temp_tag)
        # read hmmprior into prior
        self.prior = np.loadtxt(hmmprior)
        self.prior = self.prior.reshape(self.prior.shape[0],1)
        # read hmmemit into B
        self.B = np.loadtxt(hmmemit)
        # read hmmtrans into A
        self.A = np.loadtxt(hmmtrans)
        return


    def forward_backward(self):
        for i in range(len(self.test_word)):
            word = self.test_word[i]
            tag = self.test_tag[i]
            # print(word, tag)
            self.forward(word)
            self.backward(word)
            self.predict()
        return


    def forward(self, word:str):
        # initialize alpha
        self.alpha = np.zeros((len(word),len(self.index2tag)))
        self.log_alpha = np.zeros((len(word),len(self.index2tag)))
        for j in range(len(self.index2tag)):
            self.alpha[0][j] = self.prior[j]*self.B[j][word[0]]
            self.log_alpha[0][j] = np.log(self.alpha[0][j])
        for t in range(len(self.alpha)-1):
            t = t + 1
            for j in range(len(self.index2tag)):
                sum = 0
                temp = []
                for k in range(len(self.index2tag)):
                    sum += self.alpha[t-1][k]*self.A[k][j]
                    temp.append(self.log_alpha[t-1][k] + np.log(self.A[k][j]))
                self.alpha[t][j] = self.B[j][word[t]] * sum
                log_sum = self.log_sum_exp(temp)
                self.log_alpha[t][j] = np.log(self.B[j][word[t]]) + log_sum
        return


    def backward(self, word:str):
        # initialize beta
        self.beta = np.zeros((len(word),len(self.index2tag)))
        self.beta[self.beta.shape[0]-1] = 1
        for t in range(self.beta.shape[0]-2,-1,-1):
            temp = (np.multiply(self.B.transpose()[word[t+1]], self.beta[t+1]))
            temp = temp.reshape(temp.shape[0],1)
            self.beta[t] = np.dot(self.A, temp).flat
        return


    def predict(self):
        px = sum(self.alpha[self.alpha.shape[0]-1])
        log_px = self.log_sum_exp(self.log_alpha[self.alpha.shape[0]-1])
        predict = []
        for i in range(self.alpha.shape[0]):
            alpha_t = self.alpha[i]
            beta_t = self.beta[i]
            p_yt = np.multiply(alpha_t,beta_t) / px
            predict_label = np.argmax(p_yt)
            predict.append(predict_label)
        self.predict_tag.append(predict)
        self.likelihood.append(px)
        self.log_like.append(log_px)
        return


    def log_sum_exp(self, px:list):
        m = max(px)
        sum = 0
        for i in range(len(px)):
            sum += np.exp(px[i] - m)
        return m + np.log(sum)


    def calc_avg_log_likelihood(self):
        # self.avg_log_likelihood = sum(np.log(self.likelihood)) / len(self.likelihood)
        self.avg_log_likelihood = sum(self.log_like)/ len(self.log_like)
        return


    def calc_accuracy(self):
        count = 0
        total = 0
        for i in range(len(self.test_tag)):
            test_tag = self.predict_tag[i]
            actual_tag = self.test_tag[i]
            for j in range(len(test_tag)):
                if test_tag[j] != actual_tag[j]:
                    count += 1
            total += len(test_tag)
        self.accuracy = 1 - (count / total)
        return


    def write_out(self, predict_files:str, metric_out:str):
        # decode and write files in to destination
        with open(predict_files, 'w+') as fl:
            for i in range(len(self.test_word)):
                index_word = self.test_word[i]
                index_tag = self.predict_tag[i]
                result = ''
                for j in range(len(index_word)):
                    result +=  self.word2index[index_word[j]] + "_" + self.tag2index[index_tag[j]]
                    result += ' '
                result = result.strip()
                result += '\n'
                fl.write(result)

        # write metric into destination
        with open(metric_out, 'w+') as fl:
            s = "Average Log-Likelihood: {}\nAccuracy: {}".format(self.avg_log_likelihood,self.accuracy)
            fl.write(s)
        return


# Main function starts here!
if __name__ == '__main__':
    test_in = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]
    predicted_files = sys.argv[7]
    metric_out = sys.argv[8]

    hp = hmm_predicter()
    hp.load_data(test_in,index_to_word,index_to_tag,hmmprior,hmmemit,hmmtrans)
    hp.forward_backward()
    hp.calc_avg_log_likelihood()
    hp.calc_accuracy()
    hp.write_out(predicted_files, metric_out)
    print(hp.avg_log_likelihood)
    print(hp.accuracy)
    # print(hp.log_alpha)
    # print(hp.alpha)
    print(hp.log_like)