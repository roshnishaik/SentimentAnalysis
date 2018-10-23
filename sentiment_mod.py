import nltk
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from pprint import pprint
import sys

import re

#Get English language snowball stemmer and set of stopwords
stemmer = SnowballStemmer("english")
stop_words_set = set(stopwords.words('english'))

#Defines voted classifier
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        weighted_votes = [1,1,1,1,1,1,1,1,1]
        sum_votes = 0
        j = 0
        for c in self._classifiers:
            v = c.classify(features)
            if v == 'pos':
                sum_votes += weighted_votes[j] * 1
            else:
                sum_votes += weighted_votes[j] * -1
            votes.append(v)
            j += 1
        if sum_votes >= 0:
            return 'pos'
        else:
            return 'neg'

    def confidence(self, features):
        votes = []
        weighted_votes = [1,1,1,1,1,1,1,1,1]
        sum_votes = 0
        j = 0
        for c in self._classifiers:
            v = c.classify(features)
            if v == 'pos':
                sum_votes += weighted_votes[j] * 1
            else:
                sum_votes += weighted_votes[j] * -1
            votes.append(v)
            j += 1
        return (sum_votes / 10) * 100


#Define negative list
def negated_words(word_tokens):
    negate_list = []
    modifier = None
    negative_territory = 0
    for j in range(len(word_tokens)):
        word = word_tokens[j]
        neg_verbs = ["n't"]
        for i in neg_verbs:
            if i in word:
                modifier = "vrbAdj"
                negative_territory = 4
        neg_verbs = ["not", "hardly"]
        if word in neg_verbs:
            modifier = "vrbAdj"
            negative_territory = 4
        neg_nouns = ["no", "none"]
        if word in neg_nouns:
            modifier = "nouns"
            negative_territory = 4
        if negative_territory > 0:
            pos = nltk.pos_tag([word])
            pos = pos[0][1]
            if (re.match('VB[G,P,D]*', pos) or re.match(('JJ|RB'), pos)) and modifier == "vrbAdj":
                if word not in stop_words_set: negate_list.append(j)
            elif re.match('NN.*', pos) and modifier == "nouns":
                if word not in stop_words_set: negate_list.append(j)
            negative_territory -= 1
    return negate_list

#Define features in the dataset
def find_features(document):
    words = word_tokenize(document)
    words_set = set()
    negate_list = set(negated_words(words))
    j = 0
    for word in words:
        if word not in stop_words_set:
            if j in negate_list:
                words_set.add("not_" + (stemmer.stem(word.lower())))
            else:
                words_set.add(stemmer.stem(word.lower()))
        j += 1
    features = {}
    for w in word_features:
        features[w] = (w in words_set)
    return features

# Define sentiment of the given text
def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)

#Retrieve pickled algos
documents_f = open("pickled_algos/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()


word_features5k_f = open("pickled_algos/word_features6k.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()


open_file = open("pickled_algos/originalnaivebayes5k.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()


open_file = open("pickled_algos/MNB_classifier5k.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()



open_file = open("pickled_algos/BernoulliNB_classifier5k.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()


open_file = open("pickled_algos/LogisticRegression_classifier5k.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()


open_file = open("pickled_algos/LinearSVC_classifier5k.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_algos/NuSVC_classifier5k.pickle", "rb")
NuSVC_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_algos/SGDC_classifier5k.pickle", "rb")
SGDC_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_algos/RF_classifier5k.pickle", "rb")
RF_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_algos/AB_classifier5k.pickle", "rb")
AB_classifier = pickle.load(open_file)
open_file.close()

#Invoke VoteClassifier
voted_classifier = VoteClassifier(classifier,
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  SGDC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,RF_classifier, AB_classifier)


#Print the predicted polarity of the sentence by Voted classifier
#if len(sys.argv) == 2:
print(sentiment('THIS MOVIE IS GOOD'))
