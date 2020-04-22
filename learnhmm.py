__author__ = 'Jiecheng Xu'
__AndrewID__ = 'jiechenx'

import numpy as np
import sys

# toy_data/toytrain.txt toy_data/toy_index_to_word.txt toy_data/toy_index_to_tag.txt hmmprior.txt hmmemit.txt hmmtrans.txt
# handout/trainwords.txt handout/index_to_word.txt handout/index_to_tag.txt hmmprior.txt hmmemit.txt hmmtrans.txt

class hmmlearner:
    prior = [] # pi
    emit = None # B
    trans = None # A
    train_word = []
    train_tag = []
    index2word = {}
    index2tag = {}


    def populate_emit(self):
        self.emit = np.zeros((len(self.index2tag),len(self.index2word)))
        for i in range(len(self.train_word)):
            for j in range(len(self.train_tag[i])):
                word = self.train_word[i][j]
                tag = self.train_tag[i][j]
                self.emit[tag][word] += 1
        self.emit += 1
        for i in range(len(self.emit)):
            total = sum(self.emit[i])
            self.emit[i] /= total
        return


    def populate_trans(self):
        self.trans = np.zeros((len(self.index2tag),len(self.index2tag)))
        for line in self.train_tag:
            for i in range(len(line)-1):
                first = line[i]
                next = line[i+1]
                self.trans[first][next] += 1
        self.trans += 1
        for i in range(len(self.trans)):
            total = sum(self.trans[i])
            self.trans[i] /= total
        return


    def populate_prior(self):
        all_tags = []
        for i in self.index2tag.values():
            count = 0
            for line in self.train_tag:
                tag = line[0]
                if tag == i:
                    count+=1
            all_tags.append(count+1)
        total = sum(all_tags)
        for count in all_tags:
            self.prior.append(count/total)
        return


    def process_input(self, train_input:str, index_to_word:str, index_to_tag:str):

        # read index_to_word file and populate the dictionary index2word
        with open(index_to_word) as word_index:
            input = word_index.readlines()
            for i in range(len(input)):
                self.index2word[input[i].strip()] = i

        # read index_to_tag file and populate index2tag
        with open(index_to_tag) as tag_index:
            input = tag_index.readlines()
            for i in range(len(input)):
                self.index2tag[input[i].strip()] = i

        # read train_input file and encode into index form using above dictionaries
        with open(train_input) as file:
            input = file.readlines()
            # input = input[:10]
            for line in input:
                line = line.strip().split(' ')
                temp_word = []
                temp_tag = []
                for item in line:
                    item = item.strip().split("_")
                    temp_word.append(self.index2word[item[0]])
                    temp_tag.append(self.index2tag[item[1]])
                self.train_word.append(temp_word)
                self.train_tag.append(temp_tag)
        return


    def writeout(self,prior:str, emit:str, trans:str):
        np.savetxt(prior,self.prior) # write prior to destination
        np.savetxt(emit,self.emit) # write emit to destination
        np.savetxt(trans,self.trans) # write trans to destination
        return


# Main function starts here!!
if __name__ == '__main__':

    # read all parameters
    train_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]

    hmmlearner = hmmlearner()
    hmmlearner.process_input(train_input,index_to_word,index_to_tag)
    hmmlearner.populate_prior()
    hmmlearner.populate_trans()
    hmmlearner.populate_emit()
    hmmlearner.writeout(hmmprior,hmmemit,hmmtrans)