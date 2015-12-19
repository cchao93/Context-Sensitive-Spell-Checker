from nltk.corpus import brown
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
		for word in sent:
			if word[0] not in vocabulary or vocabulary[word[0]] <= 1:
				processed_sent.append((unknown_token, word[1]))
			else:
				processed_sent.append(word)
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


def main():
	tagged_training_set = brown.tagged_sents()[:50000]
	tagged_test_set = brown.tagged_sents()[-3000:]

	vocabulary = BuildVocabulary(tagged_training_set)

	tagged_training_set_prep = PreprocessTaggedCorpus(tagged_training_set, vocabulary)
	tagged_test_set_prep = PreprocessTaggedCorpus(tagged_test_set, vocabulary)
	training_set_prep = UntagCorpus(tagged_training_set_prep)
	test_set_prep = UntagCorpus(tagged_test_set_prep)

	trigram_pos_tagger = TrigramHMM()
	trigram_pos_tagger.Train(tagged_training_set_prep)

	print training_set_prep[0]
	print test_set_prep[0]

if __name__ == "__main__": 
    main()