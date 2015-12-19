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

def main():
	tagged_training_set = brown.tagged_sents()[:50000]
	tagged_test_set = brown.tagged_sents()[-3000:]

	vocabulary = BuildVocabulary(tagged_training_set)

	tagged_training_set_prep = PreprocessTaggedCorpus(tagged_training_set, vocabulary)
	tagged_test_set_prep = PreprocessTaggedCorpus(tagged_test_set, vocabulary)
	training_set_prep = UntagCorpus(tagged_training_set_prep)
	test_set_prep = UntagCorpus(tagged_test_set_prep)

	print training_set_prep[0]
	print test_set_prep[0]

if __name__ == "__main__": 
    main()