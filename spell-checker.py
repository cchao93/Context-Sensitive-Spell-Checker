from nltk.corpus import brown, treebank
from nltk.tag.util import untag
import nltk
from collections import defaultdict
from random import randint

unknown_token = "<UNK>"
start_token = "<S>"
end_token = "</S>"


def BuildConfusionSets():
    confusion_sets = []
    txt_file = open("confusion_sets_large.dat", "r")
    for line in txt_file:
		confusion_sets.append(line[:-1].split(", "))
    return confusion_sets

def IsInConfusionSet(word, confusion_sets):
    for c_set in confusion_sets:
        if word in c_set: return c_set
    return None

def SimulateSpellingErrors(test_set, confusion_sets):
    simulated_test_set = []
    for sent in test_set:
        simulated_sent = []
        for word in sent:
            c_set = IsInConfusionSet(word[0], confusion_sets)
            if c_set != None:
                error_index = randint(0, len(c_set) - 1)
                simulated_sent.append((c_set[error_index], word[1]))
            else:
                simulated_sent.append(word)
        simulated_test_set.append(simulated_sent)
    return simulated_test_set

def TreebankNoTraces():
    return [[x for x in sent if x[1] != "-NONE-"] for sent in treebank.tagged_sents()]

def BuildVocabulary(corpus):
	vocabulary = defaultdict(int)
	for sent in corpus:
		for word in untag(sent):
			# no need to do the whole "if word not in vocabulary" thing b/c of defaultdict()
			vocabulary[word] += 1
	return vocabulary

def AddConfusionSetsToVocabulary(confusion_sets):
    vocabulary = defaultdict(int)
    for c_set in confusion_sets:
        for word in c_set:
            vocabulary[word] += 1

    return vocabulary

def PreprocessTaggedCorpus(corpus, vocabulary):
	processed_corpus = list()
	for sent in corpus:
		processed_sent = list()
		processed_sent.append((start_token, start_token))
		processed_sent.append((start_token, start_token))
		for word in sent:
			if word[0] not in vocabulary or vocabulary[word[0]] <= 1:
				processed_sent.append((unknown_token, word[1]))
			else:
				processed_sent.append(word)
		processed_sent.append((end_token, end_token))
		processed_sent.append((end_token, end_token))
		processed_corpus.append(processed_sent)
	return processed_corpus

def UntagCorpus(corpus):
	untagged_corpus = list()
	for sent in corpus:
		untagged_sent = untag(sent)
		untagged_corpus.append(untagged_sent)
	return untagged_corpus

def MostCommonWordBaseline(test_set, vocabulary, confusion_sets):
    predicted_test_set = []
    for sent in test_set:
        predicted_sent = []
        for word in sent:
            c_set = IsInConfusionSet(word, confusion_sets)
            if c_set != None:
                max_freq = None
                predicted_word = None
                for ambiguous_word in c_set:
                    if ambiguous_word in vocabulary and vocabulary[ambiguous_word] > max_freq:
                        max_freq = vocabulary[ambiguous_word]
                        predicted_word = ambiguous_word
                predicted_sent.append(predicted_word)
            else:
                predicted_sent.append(word)
        predicted_test_set.append(predicted_sent)
    return predicted_test_set


def ComputeAccuracy(test_set, test_set_predicted):
    correct_sent_count = 0
    correct_word_count = 0
    num_sents = len(test_set)
    num_words = 0

    for i in xrange(0, num_sents):
        if test_set[i] == test_set_predicted[i]:
            correct_sent_count += 1
    sent_accuracy = ((float)(correct_sent_count) / (float)(num_sents)) * 100.00
    print "Percent sentence accuracy in test set is %.2f%%." %sent_accuracy
    
    for i in xrange(0, num_sents):
        for j in xrange(2, (len(test_set[i]) - 2)):
            num_words += 1
            if test_set[i][j] == test_set_predicted[i][j]:
                correct_word_count += 1
    word_accuracy = ((float)(correct_word_count) / (float)(num_words)) * 100.00
    print "Percent word accuracy in test set is %.2f%%." %word_accuracy

def Precision(test_set, test_set_simulated, test_set_predicted):
    tp = 0.0
    fp = 0.0
    num_sents = len(test_set)
    precision = 0.0

    for i in xrange(0, num_sents):
        predicted_sent = untag(test_set_predicted[i])
        for j in xrange(2, len(test_set[i]) - 2):
            if test_set[i][j] == test_set_simulated[i][j] and test_set[i][j] == predicted_sent[j]:
                tp += 1
            if test_set[i][j] != test_set_simulated[i][j] and test_set[i][j] == predicted_sent[j]:
                fp += 1

    precision = tp / (tp + fp)

    return precision

def recall(test_set, test_set_simulated, test_set_predicted):
    pass

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
        predicted_test_set = list()
        for sent in test_set:
            predicted_test_set.append(zip(untag(sent), self.Viterbi(untag(sent))))
        return predicted_test_set


class TrigramHMM:
    def __init__(self):
        self.transitions = defaultdict(float)
        self.emissions = defaultdict(float)
        self.dictionary = defaultdict(lambda:list())

    def GetUnigramCounts(self, training_set):
    	unigram_counts = defaultdict(int)
        for sent in training_set:
            for word in sent:
                unigram_counts[word[1]] += 1
        return unigram_counts

    def GetBigramCounts(self, training_set):
    	bigram_counts = defaultdict(int)
    	for sent in training_set:
            prev = 0
            for curr in xrange(1, len(sent)):
                bigram = (sent[prev][1], sent[curr][1])
                bigram_counts[bigram] += 1
                prev += 1
        return bigram_counts

    def EstimateTransitions(self, training_set):
        bigram_counts = self.GetBigramCounts(training_set)
        for sent in training_set:
            prev = 1
            prev2 = 0
            for curr in xrange(2, len(sent)):
                trigram = (sent[prev2][1], sent[prev][1], sent[curr][1])
                self.transitions[trigram] += 1
                prev += 1
                prev2 += 1
        for trigram, freq in self.transitions.iteritems():
            bigram = (trigram[0], trigram[1])
            self.transitions[trigram] = (float)(freq) / (float)(bigram_counts[bigram])

	"""
    def EstimateTransitions(self, training_set):
        bigram_counts = defaultdict(int)
        unigram_counts = defaultdict(int)
        for sent in training_set:
            prev = 1
            prev2 = 0
            for curr in xrange(2, len(sent)):
                trigram = (sent[prev2][1], sent[prev][1], sent[curr][1])
                self.transitions[trigram] += 1
                bigram = (sent[prev2][1], sent[prev][1])
                bigram_counts[bigram] += 1
                unigram = sent[prev2][1]
                unigram_counts[unigram] += 1
                prev += 1
                prev2 += 1
            bigram_counts[(sent[len(sent) - 2][1], sent[len(sent) - 1][1])] += 1
            unigram_counts[sent[len(sent) - 2][1]] += 1
            unigram_counts[sent[len(sent) - 1][1]] += 1
        word_count = 0
        for key in unigram_counts.keys():
            word_count += unigram_counts[key]
        for trigram, freq in self.transitions.iteritems():
            bigram = (trigram[0], trigram[1])
            self.transitions[trigram] = -1 #(0.4 * ((float)(freq) / (float)(bigram_counts[bigram]))) + (0.3 * ((float)(bigram_counts[(trigram[1], trigram[2])]) / (float)(unigram_counts[trigram[1]]))) + (0.3 * ((float)(unigram_counts[trigram[2]]) / (float)(word_count)))
	"""

    def EstimateEmissions(self, training_set):
        unigram_counts = self.GetUnigramCounts(training_set)
        for sent in training_set:
            for bigram in sent:
                self.emissions[bigram] += 1
        for bigram, freq in self.emissions.iteritems():
            self.emissions[bigram] = (float)(freq) / (float)(unigram_counts[bigram[1]])

    def ComputeTagDictionary(self, training_set):
        for sent in training_set:
            for word in sent:
                if word[1] not in self.dictionary[word[0]]:
                    self.dictionary[word[0]].append(word[1])

    def Train(self, training_set):
    	self.EstimateTransitions(training_set)
    	self.EstimateEmissions(training_set)
    	self.ComputeTagDictionary(training_set)

    def Viterbi(self, sent):
        T = len(sent)
        trellis = [{} for i in range(T)]
        backptr = [{} for i in range(T)]

        trellis[0][(start_token, start_token)] = 1.0
        backptr[0][(start_token, start_token)] = None

        for i in range(1, T):
            prev_word = sent[i - 1]
            prev_tags = self.dictionary[prev_word]
            word = sent[i]
            tags = self.dictionary[word]
            assert(prev_tags)
            assert(tags)
            for prev_tag in prev_tags:
            	for tag in tags:
                	max_path_prob = None
                	max_path_prev2_tag = None
                	for (prev2_tag, prev1_tag), prev_path_prob in trellis[i - 1].iteritems():
                		if prev_tag == prev1_tag:
                  			path_prob = prev_path_prob * self.transitions[(prev2_tag, prev_tag, tag)] * self.emissions[(word, tag)]
                   			if path_prob > max_path_prob and (max_path_prob or path_prob != float("-inf")):
								max_path_prob = path_prob
                       			max_path_prev2_tag = prev2_tag
                	trellis[i][(prev_tag, tag)] = max_path_prob
                	backptr[i][(prev_tag, tag)] = max_path_prev2_tag
        #best_path_prob = trellis[T - 1][end_token]
        best_path = self.Backtrace(sent, backptr)
        return best_path

    def Backtrace(self, sent, backptr, index = None, prev_tag = end_token, tag = end_token):
        if index == 0:
        	return [tag]
        if index == None:
        	index = len(sent) - 1
        prev2_tag = backptr[index][(prev_tag, tag)]
        prev_best_path = self.Backtrace(sent, backptr, index - 1, prev2_tag, prev_tag)
        return prev_best_path + [tag]

    def Test(self, test_set):
        predicted_test_set = list()
        for sent in test_set:
            predicted_test_set.append(zip(untag(sent), self.Viterbi(untag(sent))))
        return predicted_test_set

    def TestContextWords(self, test_set):
        predicted_test_set = list()
        for sent in test_set:
            predicted_test_set.append(zip(sent), self.Viterbi(sent))
        return predicted_test_set 

class ContextWords:
    def __init__(self, k, min_occurrences, vocabulary, confusion_sets):
        self.k = k
        self.min_occurrences = min_occurrences
        self.vocabulary = vocabulary
        self.confusion_sets = confusion_sets
        self.context_probs = defaultdict(float)

    def ComputeContextProbs(self, training_set):
        for sent in training_set:
            for i in xrange(0, len(sent)):
                word = sent[i]
                if IsInConfusionSet(word, self.confusion_sets) != None:
                    for j in xrange(i - self.k, i):
                        if j not in xrange(0, len(sent)): break
                        context_word = sent[j]
                        self.context_probs[(context_word, word)] += 1
                    for j in xrange(i + 1, i + 1 + self.k):
                        if j not in xrange(0, len(sent)): break
                        context_word = sent[j]
                        self.context_probs[(context_word, word)] += 1
        to_delete = []
        for bigram, freq in self.context_probs.iteritems():
            word_count = self.vocabulary[bigram[1]]
            if freq < self.min_occurrences or (word_count - freq) < self.min_occurrences:
                to_delete.append(bigram)
            else:
                self.context_probs[bigram] = (float)(freq) / (float)(word_count)
        for bigram in to_delete:
                del self.context_probs[bigram]

    #def PruneContextWords(self):
    
    def Train(self, training_set):
        self.ComputeContextProbs(training_set)

    def Test(self, test_set):
        predicted_test_set = list()
        for sent in test_set:
            predicted_sent = list()
            for i in xrange(0, len(sent)):
                word = sent[i]
                c_set = IsInConfusionSet(word, self.confusion_sets)
                if c_set != None:
                    c_dict = self.ConfusionSetToDict(c_set)
                    for ambiguous_word in c_dict.keys():
                        for j in xrange(i - self.k, i):
                            if j not in xrange(0, len(sent)): break
                            context_word = sent[j]
                            bigram = (context_word, ambiguous_word)
                            if bigram in self.context_probs and self.context_probs[bigram] != 0.0:
                                c_dict[ambiguous_word] *= self.context_probs[bigram]
                        for j in xrange(i + 1, i + 1 + self.k):
                            if j not in xrange(0, len(sent)): break
                            context_word = sent[j]
                            bigram = (context_word, ambiguous_word)
                            if bigram in self.context_probs and self.context_probs[bigram] != 0.0:
                                c_dict[ambiguous_word] *= self.context_probs[bigram]
                    max_prob = None
                    predicted_word = None
                    for ambiguous_word, prob in c_dict.iteritems():
                        if prob > max_prob:
                            max_prob = prob
                            predicted_word = ambiguous_word
                    predicted_sent.append(predicted_word)
                else:
                    predicted_sent.append(word)
            predicted_test_set.append(predicted_sent)
        return predicted_test_set

    def ConfusionSetToDict(self, confusion_set):
        confusion_dict = defaultdict(float)
        word_count = 0
        for key in self.vocabulary.keys():
            word_count += self.vocabulary[key]
        for word in confusion_set:
            confusion_dict[word] = (float)(self.vocabulary[word]) / (float)(word_count)
        return confusion_dict


def main():
    """    
    tagged_training_set = brown.tagged_sents()[:50000]
    tagged_test_set = brown.tagged_sents()[-3000:]
    """
    treebank_tagged_sents = TreebankNoTraces()
    tagged_training_set = treebank_tagged_sents[:50000] 
    tagged_test_set = treebank_tagged_sents[-3000:]
    
    vocabulary = BuildVocabulary(tagged_training_set)
    confusion_sets = BuildConfusionSets()
    #vocabulary_confusion_sets = AddConfusionSetsToVocabulary(confusion_sets)
    #vocabulary = dict(vocabulary_training.items() + vocabulary_confusion_sets.items())

    tagged_training_set_prep = PreprocessTaggedCorpus(tagged_training_set, vocabulary)
    tagged_test_set_prep = PreprocessTaggedCorpus(tagged_test_set, vocabulary)
    tagged_test_set_prep_simulated = SimulateSpellingErrors(tagged_test_set_prep, confusion_sets)
    training_set_prep = UntagCorpus(tagged_training_set_prep)
    test_set_prep = UntagCorpus(tagged_test_set_prep)
    test_set_prep_simulated = UntagCorpus(tagged_test_set_prep_simulated)

    print " ".join(training_set_prep[0])
    print " ".join(test_set_prep_simulated[0])

    """
    test_set_predicted_baseline = MostCommonWordBaseline(test_set_prep_simulated, vocabulary, confusion_sets)
    print "--- Most common class baseline accuracy ---"
    ComputeAccuracy(test_set_prep, test_set_predicted_baseline)
    """

    context_words_spell_checker = ContextWords(3, 10, vocabulary, confusion_sets)
    context_words_spell_checker.Train(training_set_prep)
    predicted_test_set = context_words_spell_checker.Test(test_set_prep_simulated)
    print "--- Context Words accuracy ---"
    ComputeAccuracy(test_set_prep, predicted_test_set)

    """
    trigram_pos_tagger = TrigramHMM()
    trigram_pos_tagger.Train(tagged_training_set_prep)
    predicted_tagged_test_set = trigram_pos_tagger.Test(tagged_test_set_prep)
    print "--- Trigram HMM accuracy ---"
    ComputeAccuracy(tagged_test_set_prep, predicted_tagged_test_set)
    """
 
    bigram_hmm = BigramHMM()
    bigram_hmm.Train(tagged_training_set_prep)
    predicted_tagged_test_set = bigram_hmm.Test(tagged_test_set_prep)
    print "--- Bigram HMM accuracy ---"
    ComputeAccuracy(tagged_test_set_prep, predicted_tagged_test_set)
    
    """The issue with the simulated words and the BigramHMM is that some of the words
    in the confusion set are not seen in the training set so when you try to look them
    up in the tag dictionary, they are not there and so they assertion fails
    
    bigram_hmm_predicted = BigramHMM()
    bigram_hmm_predicted.Train(tagged_training_set_prep)
    predicted_tagged_test_set_predicted = bigram_hmm.Test(tagged_test_set_prep_predicted)
    print "--- Bigram HMM Simluated accuracy ---"
    ComputeAccuracy(tagged_test_set_prep_predicted, predicted_tagged_test_set_predicted)
    """

    trainingSents = treebank.tagged_sents()[:50000]
    testSents = treebank.tagged_sents()[-3000:]

    trigramTagger = nltk.TrigramTagger(trainingSents)
    evalResult = trigramTagger.evaluate(testSents)
    print "--- NLTK TrigramTagger accuracy ---"
    print "%4.2f" % (100.0 * evalResult)

    print Precision(test_set_prep, predicted_test_set, predicted_tagged_test_set)
    

if __name__ == "__main__": 
    main()