import nltk
from nltk.corpus import brown, treebank
from nltk.tag.util import untag
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

def PruneConfusionSets(vocabulary, confusion_sets):
    pruned_sets = []
    for c_set in confusion_sets:
        pruned_c_set = []
        for word in c_set:
            if word in vocabulary and vocabulary[word] > 1:
                pruned_c_set.append(word)
        pruned_sets.append(pruned_c_set)
    return pruned_sets

def IsInConfusionSet(word, confusion_sets):
    for c_set in confusion_sets:
        if word in c_set: return c_set
    return None

def BuildStopWordsList():
    txt_file = open("stop_words.dat", "r")
    stop_words = txt_file.read().splitlines()
    return stop_words

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

def ComputeAccuracy(test_set, test_set_simulated, test_set_predicted):
    correct_sent_count = 0
    correct_word_count = 0
    generated_error_count = 0
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
            if test_set[i][j] == test_set_predicted[i][j]:
                correct_word_count += 1
    word_accuracy = ((float)(correct_word_count) / (float)(num_words)) * 100.00
    print "Percent word accuracy in test set is %.2f%%." %word_accuracy

    correct_word_count = 0
    for i in xrange(0, num_sents):
        for j in xrange(1, (len(test_set[i]) - 1)):
            if test_set[i][j] != test_set_simulated[i][j]:
                generated_error_count += 1
                if test_set[i][j] == test_set_predicted[i][j]:
                    correct_word_count += 1
    error_correction_accuracy = ((float)(correct_word_count) / (float)(generated_error_count)) * 100.00
    print "Percent errors corrected in test set is %.2f%%." %error_correction_accuracy

class BigramHMM:
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

    def EstimateTransitions(self, training_set):
        unigram_counts = self.GetUnigramCounts(training_set)
        for sent in training_set:
            prev = 0
            for curr in xrange(1, len(sent)):
                bigram = (sent[prev][1], sent[curr][1])
                self.transitions[bigram] += 1
                prev += 1
        for bigram, freq in self.transitions.iteritems():
            self.transitions[bigram] = (float)(freq) / (float)(unigram_counts[bigram[0]])

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

    def Test(self, sent):
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
                for prev_tag, prev_path_prob in trellis[i-1].iteritems():
                    path_prob = prev_path_prob * self.transitions[(prev_tag, tag)] * self.emissions[(word, tag)]
                    if path_prob > max_path_prob and (max_path_prob or path_prob != float('-inf')):
                        max_path_prob = path_prob
                        max_path_prev_tag = prev_tag
                trellis[i][tag] = max_path_prob
                backptr[i][tag] = max_path_prev_tag

        best_path_prob = trellis[T-1][end_token]
        best_path = self.Backtrace(sent, backptr)
        return (best_path_prob, best_path)

    def Backtrace(self, sent, backptr, index = None, tag = end_token):
        if index == 0: return [tag]
        if index == None: index = len(sent) - 1
        prev_tag = backptr[index][tag]
        prev_best_path = self.Backtrace(sent, backptr, index - 1, prev_tag)
        return prev_best_path + [tag]

class ContextWords:
    def __init__(self, k, min_occurrences, vocabulary, confusion_sets, stop_words):
        self.k = k
        self.min_occurrences = min_occurrences
        self.vocabulary = vocabulary
        self.confusion_sets = confusion_sets
        self.stop_words = stop_words
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

    def PruneContextWords(self):
        to_delete = []
        for bigram, freq in self.context_probs.iteritems():
            if bigram[0] in self.stop_words:
                to_delete.append(bigram)
        for bigram in to_delete:
            del self.context_probs[bigram]

    def Train(self, training_set):
        self.ComputeContextProbs(training_set)
        self.PruneContextWords()

    def Test(self, sent, i):
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
            return predicted_word
        else:
            return word

    def ConfusionSetToDict(self, confusion_set):
        confusion_dict = defaultdict(float)
        word_count = 0
        for key in self.vocabulary.keys():
            word_count += self.vocabulary[key]
        for word in confusion_set:
            confusion_dict[word] = (float)(self.vocabulary[word]) / (float)(word_count)
        return confusion_dict

class HybridModel:
    def __init__(self, k, min_occurrences, vocabulary, confusion_sets, stop_words):
        self.confusion_sets = confusion_sets
        self.bigram_pos_tagger = BigramHMM()
        self.context_words_spell_checker = ContextWords(3, 10, vocabulary, confusion_sets, stop_words)

    def Train(self, training_set):
        self.bigram_pos_tagger.Train(training_set)
        self.context_words_spell_checker.Train(UntagCorpus(training_set))
    
    def Test(self, test_set):
        predicted_test_set = []
        for sent in test_set:
            predicted_sent = []
            for i in xrange(0, len(sent)):
                word = sent[i]
                c_set = IsInConfusionSet(word, self.confusion_sets)
                c_set_stats = dict()
                if c_set != None:
                    for ambiguous_word in c_set:
                        potential_sent = sent
                        potential_sent[i] = ambiguous_word
                        (prob, tags) = self.bigram_pos_tagger.Test(potential_sent)
                        c_set_stats[ambiguous_word] = (prob, tags)
                    potential_tags = []
                    for ambiguous_word, stats in c_set_stats.iteritems():
                        potential_tags.append(stats[1][i])
                    if potential_tags[1:] == potential_tags[:-1]:
                        predicted_word = self.context_words_spell_checker.Test(sent, i)
                        predicted_sent.append(predicted_word)
                    else:
                        best_prob = None
                        best_word = None
                        for ambiguous_word, stats in c_set_stats.iteritems():
                            if stats[0] > best_prob:
                                best_prob = stats[0]
                                best_word = ambiguous_word
                        predicted_sent.append(best_word)
                else:
                    predicted_sent.append(word)
            predicted_test_set.append(predicted_sent)
        return predicted_test_set
    
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
    pruned_confusion_sets = PruneConfusionSets(vocabulary, confusion_sets)

    stop_words = BuildStopWordsList()

    tagged_test_set_simulated = SimulateSpellingErrors(tagged_test_set, pruned_confusion_sets)

    tagged_training_set_prep = PreprocessTaggedCorpus(tagged_training_set, vocabulary)
    tagged_test_set_prep = PreprocessTaggedCorpus(tagged_test_set, vocabulary)
    tagged_test_set_prep_simulated = PreprocessTaggedCorpus(tagged_test_set_simulated, vocabulary)
    
    training_set_prep = UntagCorpus(tagged_training_set_prep)
    test_set_prep = UntagCorpus(tagged_test_set_prep)
    test_set_prep_simulated = UntagCorpus(tagged_test_set_prep_simulated)

    print " ".join(training_set_prep[0])
    print " ".join(test_set_prep[0])
    print " ".join(test_set_prep_simulated[0])

    test_set_predicted_baseline = MostCommonWordBaseline(test_set_prep_simulated, vocabulary, pruned_confusion_sets)
    print "--- Most common class baseline accuracy ---"
    ComputeAccuracy(test_set_prep, test_set_prep_simulated, test_set_predicted_baseline)

    hybrid = HybridModel(3, 10, vocabulary, pruned_confusion_sets, stop_words)
    hybrid.Train(tagged_training_set_prep)
    predicted_test_set = hybrid.Test(test_set_prep_simulated)
    print "--- Hybrid method accuracy ---"
    ComputeAccuracy(test_set_prep, test_set_prep_simulated, predicted_test_set)

if __name__ == "__main__": 
    main()