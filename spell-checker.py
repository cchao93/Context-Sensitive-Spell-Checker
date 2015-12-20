from nltk.corpus import brown, treebank
from nltk.tag.util import untag
from collections import defaultdict

unknown_token = "<UNK>"
start_token = "<S>"
end_token = "</S>"

def BuildVocabulary(corpus):
	vocabulary = defaultdict(int)
	for sent in corpus:
		for word in untag(sent):
			# no need to do the whole "if word not in vocabulary" thing b/c of defaultdict()
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

class TrigramHMM:
    def __init__(self):
        self.transitions = defaultdict(float)
        self.emissions = defaultdict(float)
        self.dictionary = defaultdict(lambda:list())

    def EstimateTransitions(self, training_set):
        bigram_counts = defaultdict(int)
        for sent in training_set:
            prev = 1
            prev2 = 0
            for curr in xrange(2, len(sent)):
                trigram = (sent[prev2][1], sent[prev][1], sent[curr][1])
                self.transitions[trigram] += 1
                bigram = (sent[prev2][1], sent[prev][1])
                bigram_counts[bigram] += 1
                prev += 1
                prev2 += 1
        for trigram, freq in self.transitions.iteritems():
            bigram = (trigram[0], trigram[1])
            self.transitions[trigram] = (float)(freq) / (float)(bigram_counts[bigram])

    def EstimateEmissions(self, training_set):
        unigram_counts = defaultdict(int)
        for sent in training_set:
            for bigram in sent:
                unigram_counts[bigram[1]] += 1
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

        trellis[0][start_token] = 1.0
        backptr[0][start_token] = None

        for i in range(1, T):
            word = sent[i]
            tags = self.dictionary[word]
            assert(tags)
            for tag in tags:
                max_path_prob = None
                max_path_prev_tag = None
                for prev_tag, prev_path_prob in trellis[i - 1].iteritems():
                    path_prob = prev_path_prob * self.transitions[(prev_tag, tag)] * self.emissions[(word, tag)]
                    if path_prob > max_path_prob and (max_path_prob or path_prob != float("-inf")):
                        max_path_prob = path_prob
                        max_path_prev_tag = prev_tag
                trellis[i][tag] = max_path_prob
                backptr[i][tag] = max_path_prev_tag
        best_path_prob = trellis[T - 1][end_token]
        best_path = self.Backtrace(sent, backptr)
        return best_path

    def Backtrace(self, sent, backptr, index = None, tag = end_token):
        if index == 0:
        	return [tag]
        if index == None:
        	index = len(sent) - 1
        prev_tag = backptr[index][tag]
        prev_best_path = self.Backtrace(sent, backptr, index - 1, prev_tag)
        return prev_best_path + [tag]

    def Test(self, test_set):
        predicted_test_set = list()
        for sent in test_set:
            predicted_test_set.append(zip(untag(sent), self.Viterbi(untag(sent))))
        return predicted_test_set

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

def TreebankNoTraces():
    return [[x for x in sent if x[1] != "-NONE-"] for sent in treebank.tagged_sents()]

def main():
    """
    tagged_training_set = brown.tagged_sents()[:50000]
    tagged_test_set = brown.tagged_sents()[-3000:]
    """
    treebank_tagged_sents = TreebankNoTraces()
    tagged_training_set = treebank_tagged_sents[:50000] 
    tagged_test_set = treebank_tagged_sents[-3000:]

    vocabulary = BuildVocabulary(tagged_training_set)

    tagged_training_set_prep = PreprocessTaggedCorpus(tagged_training_set, vocabulary)
    tagged_test_set_prep = PreprocessTaggedCorpus(tagged_test_set, vocabulary)
    training_set_prep = UntagCorpus(tagged_training_set_prep)
    test_set_prep = UntagCorpus(tagged_test_set_prep)

    print " ".join(training_set_prep[0])
    print " ".join(test_set_prep[0])

    trigram_pos_tagger = TrigramHMM()
    trigram_pos_tagger.Train(tagged_training_set_prep)
    predicted_test_set = trigram_pos_tagger.Test(tagged_test_set_prep)
    print "--- Trigram HMM accuracy ---"
    ComputeAccuracy(tagged_test_set_prep, predicted_test_set)

if __name__ == "__main__": 
    main()