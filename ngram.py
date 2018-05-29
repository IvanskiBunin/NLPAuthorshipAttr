from string import punctuation
from collections import Counter
import os
import math
import operator
import nltk
import nltk.data

project_path = "/Users/ivanski/Programming/Python/CS585/Project"
gutenberg_data = project_path + "/gutenbergdataset"
yelp_data = project_path + "/yelpdataset"
C50_data = project_path + "/C50"
function_words = ["the", "a", "and", "or", "there", "their", "he", "him", "his", "hers", "her", "she", "do", "yes", "okay"]



def cossim(dict1, dict2):
	dotprod = 0
	for key in dict1.iterkeys():
		if key in dict2:
			dotprod += (dict1[key] * dict2[key])
	len1 = 0
	len2 = 0
	for value in dict1.itervalues():
		len1 += (value * value)
	len1 = math.sqrt(len1)
	for value in dict2.itervalues():
		len2 += (value * value)
	len2 = math.sqrt(len2)
	if len1 == 0 or len2 == 0:
		return 0.
	return (dotprod / (len1 * len2))

# Classify by punctuation percentage (Number of various punctuation in document / number of words)

def get_punct_percentages(dataset, trainingSet):
	author_dicts = []
	for author in os.listdir(dataset + "/" + trainingSet):
		if author[0] == '.':
			continue
		books = os.listdir(dataset + "/" + trainingSet + "/" + author)
		numbooks = len(books)
		authorpunctperc = {}
		for f in books:
			if f[0] == '.':
				continue
			book = open(dataset + "/" + trainingSet + "/" + author + "/" + f)
			fString = book.read()
			numberWords = len(fString.split())
			counts = Counter(fString)
			punctuation_counts = {k:float(v) for k, v in counts.iteritems() if k in punctuation}
			punctuation_percentage = {k: v / numberWords for k, v in punctuation_counts.iteritems()}
			book.close()
			authorpunctperc = Counter(authorpunctperc) + Counter(punctuation_percentage)
		authorpunctperc = {k: v / numbooks for k, v in authorpunctperc.iteritems()}
		author_dicts += [(author, authorpunctperc)]
	return author_dicts

gutenberg_test = gutenberg_data + "/test"
author_correct_classifications = {}

def classify_docs(dataset, testSet, authorDict):
	totalbooks = 0.
	correctclassifications = 0.
	for author in os.listdir(dataset + "/" + testSet):
		if author[0] == ".":
			continue
		for f in os.listdir(dataset + "/" + testSet + "/" + author):
			if f[0] == ".":
				continue
			book = open(dataset + "/" + testSet + "/" + author + "/" + f)
			totalbooks += 1
			fString = book.read()
			numberWords = len(fString.split())
			counts = Counter(fString)
			punctuation_counts = {k:float(v) for k, v in counts.iteritems() if k in punctuation}
			punctuation_percentage = {k: v / numberWords for k, v in punctuation_counts.iteritems()}
			book.close()
			highestsimilarity = 0
			author_similarities = {}
			for author_tuple in authorDict:
				similarity = cossim(author_tuple[1], punctuation_percentage)
				author_similarities[author_tuple[0]] = similarity
				if similarity > highestsimilarity:
					highestsimilarity = similarity
					mostlikelyauthor = author_tuple[0]
			sorted_similarities = (sorted(author_similarities.items(), key=operator.itemgetter(1)))[::-1]
			#for item in sorted_similarities:
			#	print(item)
			#print("The most likely author for " + f + " is " + mostlikelyauthor)
			if (author == mostlikelyauthor or sorted_similarities[1][0] == author or sorted_similarities[2][0] == author or sorted_similarities[3][0] == author or sorted_similarities[4][0] == author):
				correctclassifications += 1
	return (correctclassifications / totalbooks)

gutenberg_percent = classify_docs(gutenberg_data, "test", get_punct_percentages(gutenberg_data, "train")) * 100
C50_percent = classify_docs(C50_data, "C50test", get_punct_percentages(C50_data, "C50train")) * 100

print("Classify by punctuation percentage")
print(str(gutenberg_percent) + " percent correctly classified for the gutenberg dataset")
print(str(C50_percent) + " percent correctly classified for the C50 dataset")


# Classify by average sentence length
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def get_avg_sentence_length(dataset, trainingSet):
	author_dict = {}
	for author in os.listdir(dataset + "/" + trainingSet):
		if author[0] == '.':
			continue
		books = os.listdir(dataset + "/" + trainingSet + "/" + author)
		numbooks = len(books)
		authorsentencelength = 0.
		for f in books:
			if f[0] == '.':
				continue
			book = open(dataset + "/" + trainingSet + "/" + author + "/" + f)
			fString = book.read()
			fString = fString.decode("utf-8")
			book.close()
			sentences = tokenizer.tokenize(fString)
			avgLength = 0.
			for sentence in sentences:
				avgLength += len(sentence.split())
			avgLength /= (len(sentences))
			authorsentencelength += avgLength
		authorsentencelength /= numbooks
		author_dict[author] = authorsentencelength
	return author_dict

def classify_by_sentence_length(dataset, testSet, author_dict):
	author_correct_classifications = {}
	totalbooks = 0.
	correctclassifications = 0.
	for author in os.listdir(dataset + "/" + testSet):
		if author[0] == ".":
			continue
		for f in os.listdir(dataset + "/" + testSet + "/" + author):
			if f[0] == ".":
				continue
			book = open(dataset + "/" + testSet + "/" + author + "/" + f)
			fString = book.read()
			totalbooks += 1
			fString = fString.decode("utf-8")
			book.close()
			sentences = tokenizer.tokenize(fString)
			avgLength = 0.
			for sentence in sentences:
				avgLength += len(sentence.split())
			avgLength /= (len(sentences))
			highestsimilarity = 0
			author_similarities = {}
			for otherauthor in author_dict.iterkeys():
				similarity = abs(avgLength - author_dict[otherauthor])
				author_similarities[otherauthor] = similarity
				if (highestsimilarity == 0) or (similarity < highestsimilarity):
					highestsimilarity = similarity
					mostlikelyauthor = otherauthor
			sorted_similarities = (sorted(author_similarities.items(), key=operator.itemgetter(1)))
			#for item in sorted_similarities:
			#	print(item)
			#print("The most likely author for " + f + " is " + mostlikelyauthor)
			if (author == mostlikelyauthor or sorted_similarities[1][0] == author or sorted_similarities[2][0] == author or sorted_similarities[3][0] == author or sorted_similarities[4][0] == author):
				correctclassifications += 1
	return correctclassifications / totalbooks

gutenberg_percent = (classify_by_sentence_length(gutenberg_data, "test", get_avg_sentence_length(gutenberg_data, "train"))) * 100
C50_percent = (classify_by_sentence_length(C50_data, "C50test", get_avg_sentence_length(C50_data, "C50train"))) * 100

print("Classify by average sentence length")
print(str(gutenberg_percent) + " percent of books correctly classified for the gutenberg dataset")
print(str(C50_percent) + " percent of texts correctly classified for the C50 dataset")

# Classify by character n-gram frequency

def get_ngram_frequency(dataset, trainingSet, n):
	author_dict = {}
	for author in os.listdir(dataset + "/" + trainingSet):
		if author[0] == '.':
			continue
		books = os.listdir(dataset + "/" + trainingSet + "/" + author)
		numbooks = len(books)
		numchars = 0.
		for f in books:
			if f[0] == '.':
				continue
			book = open(dataset + "/" + trainingSet + "/" + author + "/" + f)
			fString = book.read()
			book.close()
			chars = [c for c in fString]
			numchars += len(chars)
			ngrams = Counter(nltk.ngrams(chars, n))
		ngrams = {k: v / numchars for k, v in ngrams.iteritems()}
		ngrams = {k: v / numchars for k, v in ngrams.iteritems() if v > 0.0001}
		author_dict[author] = ngrams
	return author_dict


def classify_by_ngram_frequency(dataset, testSet, n, author_dict):
	totalbooks = 0.
	correctclassifications = 0.
	for author in os.listdir(dataset + "/" + testSet):
		if author[0] == ".":
			continue
		for f in os.listdir(dataset + "/" + testSet + "/" + author):
			if f[0] == ".":
				continue
			book = open(dataset + "/" + testSet + "/" + author + "/" + f)
			fString = book.read()
			totalbooks += 1
			book.close()
			chars = [c for c in fString]
			numchars = 0. + len(chars)
			textdict = {}
			ngrams = Counter(nltk.ngrams(chars, n))
			textdict = {k: v / numchars for k, v in ngrams.iteritems()}
			textdict = {k: v / numchars for k, v in textdict.iteritems() if v > 0.0001}
			highestsimilarity = 0
			author_similarities = {}
			mostlikelyauthor = ""
			for otherauthor in author_dict.iterkeys():
				similarity = cossim(textdict, author_dict[otherauthor])
				author_similarities[otherauthor] = similarity
				if similarity > highestsimilarity:
					highestsimilarity = similarity
					mostlikelyauthor = otherauthor
			sorted_similarities = (sorted(author_similarities.items(), key=operator.itemgetter(1)))[::-1]
			#for item in sorted_similarities:
			#	print(item)
			#print("The most likely author for " + f + " is " + mostlikelyauthor)
			if (author == mostlikelyauthor or sorted_similarities[1][0] == author or sorted_similarities[2][0] == author or sorted_similarities[3][0] == author or sorted_similarities[4][0] == author):
				correctclassifications += 1	
	return correctclassifications / totalbooks


gutenberg_percent = (classify_by_ngram_frequency(gutenberg_data, "test", 4, get_ngram_frequency(gutenberg_data, "train", 4))) * 100
C50_percent = (classify_by_ngram_frequency(C50_data, "C50test", 4, get_ngram_frequency(C50_data, "C50train", 4))) * 100

print("Classify by n-gram frequency")
print(str(gutenberg_percent) + " percent correctly classified for the gutenberg dataset")
print(str(C50_percent) + " percent correctly classified for the C50 dataset")


# Classify by function word distribution

def get_funcword_frequency(dataset, trainingSet):
	author_dicts = {}
	for author in os.listdir(dataset + "/" + trainingSet):
		if author[0] == '.':
			continue
		books = os.listdir(dataset + "/" + trainingSet + "/" + author)
		numbooks = len(books)
		authorfunctperc = {}
		for f in books:
			if f[0] == '.':
				continue
			book = open(dataset + "/" + trainingSet + "/" + author + "/" + f)
			fString = book.read()
			numberWords = len(fString.split())
			counts = Counter(fString.split())
			function_counts = {k:float(v) for k, v in counts.iteritems() if k in function_words}
			function_percentage = {k: v / numberWords for k, v in function_counts.iteritems()}
			book.close()
			authorfunctperc = Counter(authorfunctperc) + Counter(function_percentage)
		authorfunctperc = {k: v / numbooks for k, v in authorfunctperc.iteritems()}
		author_dicts[author] = authorfunctperc
	return author_dicts

def classify_by_funcword(dataset, testSet, author_dict):
	totalbooks = 0.
	correctclassifications = 0.
	for author in os.listdir(dataset + "/" + testSet):
		if author[0] == ".":
			continue
		for f in os.listdir(dataset + "/" + testSet + "/" + author):
			if f[0] == ".":
				continue
			book = open(dataset + "/" + testSet + "/" + author + "/" + f)
			totalbooks += 1
			fString = book.read()
			numberWords = len(fString.split())
			counts = Counter(fString.split())
			function_counts = {k:float(v) for k, v in counts.iteritems() if k in function_words}
			function_percentage = {k: v / numberWords for k, v in function_counts.iteritems()}
			book.close()
			highestsimilarity = 0
			author_similarities = {}
			for otherauthor in author_dict.iterkeys():
				similarity = cossim(author_dict[otherauthor], function_percentage)
				author_similarities[otherauthor] = similarity
				if similarity > highestsimilarity:
					highestsimilarity = similarity
					mostlikelyauthor = otherauthor
			sorted_similarities = (sorted(author_similarities.items(), key=operator.itemgetter(1)))[::-1]
			#for item in sorted_similarities:
			#	print(item)
			#print("The most likely author for " + f + " is " + mostlikelyauthor)
			if (author == mostlikelyauthor or sorted_similarities[1][0] == author or sorted_similarities[2][0] == author or sorted_similarities[3][0] == author or sorted_similarities[4][0] == author):
				correctclassifications += 1
	return (correctclassifications / totalbooks)

print("Classify by function word frequency")
gutenberg_percent = classify_by_funcword(gutenberg_data, "test", get_funcword_frequency(gutenberg_data, "train")) * 100
C50_percent = classify_by_funcword(C50_data, "C50test", get_funcword_frequency(C50_data, "C50train")) * 100

print(str(gutenberg_percent) + " correctly classified for gutenberg dataset")
print(str(C50_percent) + " correctly classified for C50 dataset")


