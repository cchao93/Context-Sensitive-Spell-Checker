from nltk.corpus import brown
from collections import defaultdict

unknown_token = "<UNK>"
start_token = "<S>"
end_token = "</S>"

def BuildVocabulary(corpus):
	vocabulary = defaultdict(int)
	for sent in corpus:
		for word in sent:
			# no need to do the whole "if word not in vocabulary" thing b/c of defaultdict()
			vocabulary[word] += 1
	return vocabulary

def PreprocessCorpus(corpus, vocabulary):
	processed_corpus = list()
	for sent in corpus:
		processed_sent = list()
		processed_sent.append(start_token)
		for word in sent:
			if word not in vocabulary or vocabulary[word] <= 1:
				processed_sent.append(unknown_token)
			else:
				processed_sent.append(word)
		processed_sent.append(end_token)
		processed_corpus.append(processed_sent)
	return processed_corpus

def main():
	training_set = brown.sents()[:50000]
	test_set = brown.sents()[-3000:]

	vocabulary = BuildVocabulary(training_set)
	training_set_prep = PreprocessCorpus(training_set, vocabulary)
	test_set_prep = PreprocessCorpus(test_set, vocabulary)