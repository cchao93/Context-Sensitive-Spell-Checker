""" Victor Chia-Chi Chao
    COMP150 NLP
    Problem Set 3
    10/28/15
"""

import sys
from collections import defaultdict
from math import log, exp
import math

from nltk.corpus import treebank
from nltk.tag.util import untag  # Untags a tagged sentence. 

unknown_token = "<UNK>"  # unknown word token.
start_token = "<S>"  # sentence boundary token.
end_token = "</S>"  # sentence boundary token.

""" Remove trace tokens and tags from the treebank as these are not necessary.
"""
def TreebankNoTraces():
    return [[x for x in sent if x[1] != "-NONE-"] for sent in treebank.tagged_sents()]

def PreprocessText(corpus, vocabulary):
    new_corpus = list()
    for sent in corpus:
        new_sent = list()
        new_sent.append((start_token, start_token))
        for tagged_word in sent:
            if tagged_word[0] not in vocabulary or vocabulary[tagged_word[0]] <= 1:
                new_sent.append((unknown_token, tagged_word[1]))
            else:
                new_sent.append(tagged_word)
        new_sent.append((end_token, end_token))
        new_corpus.append(new_sent)

    return new_corpus

def BuildVocab(corpus):
    vocabulary = defaultdict(int)
    for sent in corpus:
        for tagged_word in sent:
            vocabulary[tagged_word[0]] += 1
    return vocabulary
        
class BigramHMM:
    def __init__(self):
        """ Implement:
        self.transitions, the A matrix of the HMM: a_{ij} = P(t_j | t_i)
        self.emissions, the B matrix of the HMM: b_{ii} = P(w_i | t_i)
        self.dictionary, a dictionary that maps a word to the set of possible tags
        """
        self.transitions = defaultdict(float)
        self.emissions = defaultdict(float)
        self.dictionary = defaultdict(lambda:list())
        
    def Train(self, training_set):
        """ 
        1. Estimate the A matrix a_{ij} = P(t_j | t_i)
        2. Estimate the B matrix b_{ii} = P(w_i | t_i)
        3. Compute the tag dictionary 
        """
        unigram_counts = defaultdict(int)
        for sent in training_set:
            for tagged_word in sent:
                unigram_counts[tagged_word[1]] += 1
        self.EstimateTransitions(training_set, unigram_counts)
        self.EstimateEmissions(training_set, unigram_counts)
        self.ComputeTagDictionary(training_set)


    def EstimateTransitions(self, training_set, unigram_counts):
        for sent in training_set:
            prev = 0
            for word_index in xrange(1, len(sent)):
                bigram = (sent[prev][1], sent[word_index][1])
                self.transitions[bigram] += 1
                prev += 1
        for bigram, freq in self.transitions.iteritems():
            self.transitions[bigram] = (float)(freq) / (float)(unigram_counts[bigram[0]])

    def EstimateEmissions(self, training_set, unigram_counts):
        for sent in training_set:
            for bigram in sent:
                self.emissions[bigram] += 1
        for bigram, freq in self.emissions.iteritems():
            self.emissions[bigram] = (float)(freq) / (float)(unigram_counts[bigram[1]])

    def ComputeTagDictionary(self, training_set):
        for sent in training_set:
            for tagged_word in sent:
                if tagged_word[1] not in self.dictionary[tagged_word[0]]:
                    self.dictionary[tagged_word[0]].append(tagged_word[1])

    def ComputePercentAmbiguous(self, data_set):
        """ Compute the percentage of tokens in data_set that have more than one tag according to self.dictionary. """
        print self.dictionary[unknown_token]
        ambiguous_token_count = 0
        num_tokens = 0
        for sent in data_set:
            num_tokens += len(sent)
            for tagged_word in sent:
                if len(self.dictionary[tagged_word[0]]) > 1:
                    ambiguous_token_count += 1
        return ((float)(ambiguous_token_count) / (float)(num_tokens)) * 100.00

    def JointProbability(self, sent):
        """ Compute the joint probability of the words and tags of a tagged sentence. """
        jointProb = 1.0
        for word_index in xrange(1, len(sent)):
            tag_bigram = (sent[word_index - 1][1], sent[word_index][1])
            word_bigram = sent[word_index]
            jointProb *= self.transitions[tag_bigram] * self.emissions[word_bigram]
        return jointProb


    def BestPath(self, sent, backptr, index = None, tag = end_token):
        if index == 0: return [tag]
        if index == None: index = len(sent) - 1
        prev_tag = backptr[index][tag]
        prev_best_path = self.BestPath(sent, backptr, index - 1, prev_tag)
        return prev_best_path + [tag]

    def Viterbi(self, sent):  # test sentence is untagged, unk'ed and padded.
        T = len(sent)
        trellis = [{} for i in range(T)]
        backptr = [{} for i in range(T)]

        # initialize the trellis
        trellis[0][start_token] = 1.0  # log(1.0)
        backptr[0][start_token] = None

        # compute the trellis
        for i in range(1, T):
            word = sent[i]
            tags = self.dictionary[word]
            assert(tags)  # not the empty set
            for tag in tags:
                max_path_log_prob = None  # all real values are greater than None.
                max_path_prev_tag = None
                for prev_tag, prev_path_log_prob in trellis[i-1].iteritems():
                    path_log_prob = prev_path_log_prob * self.transitions[(prev_tag, tag)] * self.emissions[(word, tag)]
                    if path_log_prob > max_path_log_prob and (max_path_log_prob or path_log_prob != float('-inf')):
                        max_path_log_prob = path_log_prob
                        max_path_prev_tag = prev_tag
                trellis[i][tag] = max_path_log_prob
                backptr[i][tag] = max_path_prev_tag

        best_path_prob = trellis[T-1][end_token]
        best_path = self.BestPath(sent, backptr)
        return best_path


    def Test(self, test_set):
        """ Use Viterbi and predict the most likely tag sequence for every sentence. Return a re-tagged test_set. """
        new_test_set = list()
        for sent in test_set:
            new_test_set.append(zip(untag(sent), self.Viterbi(untag(sent))))
        return new_test_set

def MostCommonClassBaseline(training_set, test_set):
    """ Implement the most common class baseline for POS tagging. Return the test set tagged according to this baseline. """
    tagged_word_freq = dict()
    most_freq_tag = dict()
    new_test_set = list()

    for sent in training_set:
        for tagged_word in sent:
            if tagged_word not in tagged_word_freq.keys():
                tagged_word_freq[tagged_word] = 1
            else:
                tagged_word_freq[tagged_word] += 1

    for tagged_word, freq in tagged_word_freq.iteritems():
        if tagged_word[0] not in most_freq_tag.keys():
            most_freq_tag[tagged_word[0]] = (tagged_word[1], freq)
        elif freq > most_freq_tag[tagged_word[0]][1]:
            most_freq_tag[tagged_word[0]] = (tagged_word[1], freq)

    for sent in test_set:
        new_sent = list()
        for tagged_word in sent:
            if tagged_word[0] in most_freq_tag.keys():
                new_sent.append((tagged_word[0], most_freq_tag[tagged_word[0]][0]))
            else:
                new_sent.append(tagged_word)
        new_test_set.append(new_sent)

    return new_test_set


def ComputeAccuracy(test_set, test_set_predicted):
    """ Using the gold standard tags in test_set, compute the sentence and tagging accuracy of test_set_predicted. """
    correct_sent_count = 0
    correct_tag_count = 0
    num_sents = len(test_set)
    num_words = 0

    for i in xrange(0, num_sents):
        if test_set[i] == test_set_predicted[i]:
            correct_sent_count += 1
    sent_accuracy = ((float)(correct_sent_count) / (float)(num_sents)) * 100.00
    print "Percent sentence accuracy in test set is %.2f%%." %sent_accuracy
    
    for i in xrange(0, num_sents):
        for j in xrange(1, (len(test_set[i]) - 1)):
            num_words += 1
            #if test_set[i][j] != (start_token, start_token)
            #and test_set[i][j] != (end_token, end_token)
            if test_set[i][j] == test_set_predicted[i][j]:
                correct_tag_count += 1
    tagging_accuracy = ((float)(correct_tag_count) / (float)(num_words)) * 100.00
    print "Percent tagging accuracy in test set is %.2f%%." %tagging_accuracy

def main():
    treebank_tagged_sents = TreebankNoTraces()  # Remove trace tokens. 
    training_set = treebank_tagged_sents[:50000]  # This is the train-test split that we will use. 
    test_set = treebank_tagged_sents[-3000:]
    
    vocabulary = BuildVocab(training_set)

    """ Transform the data sets by eliminating unknown words and adding sentence boundary tokens.
    """
    training_set_prep = PreprocessText(training_set, vocabulary)
    test_set_prep = PreprocessText(test_set, vocabulary)
    
    """ Print the first sentence of each data set.
    """
    print " ".join(untag(training_set_prep[0]))  # See nltk.tag.util module.
    print " ".join(untag(test_set_prep[0]))

    """ Estimate Bigram HMM from the training set, report level of ambiguity.
    """
    bigram_hmm = BigramHMM()
    bigram_hmm.Train(training_set_prep)
    #print "Percent tag ambiguity in training set is %.2f%%." %bigram_hmm.ComputePercentAmbiguous(training_set_prep)
    #print "Joint probability of the first sentence is %s." %bigram_hmm.JointProbability(training_set_prep[0])
    
    """ Implement the most common class baseline. Report accuracy of the predicted tags.
    """
    #test_set_predicted_baseline = MostCommonClassBaseline(training_set_prep, test_set_prep)
    #print "--- Most common class baseline accuracy ---"
    #ComputeAccuracy(test_set_prep, test_set_predicted_baseline)

    """ Use the Bigram HMM to predict tags for the test set. Report accuracy of the predicted tags.
    """
    test_set_predicted_bigram_hmm = bigram_hmm.Test(test_set_prep)
    print "--- Bigram HMM accuracy ---"
    ComputeAccuracy(test_set_prep, test_set_predicted_bigram_hmm)    

if __name__ == "__main__": 
    main()