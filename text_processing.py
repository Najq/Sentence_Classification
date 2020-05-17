#author - Najeeb Qazi

import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

#I used SVM for classification, but model tuning can be used to arrive at better
#hyperparameters. Other models such as naive bayes and logistic regression can be also be compared against it.
#Also, the number of progressive sentences are far less as compared to the non-progressive ones, which could affect the model
#performance, hence we also need to take care of class imbalance


def parse_data(text):

	#replacing prefixes in the entire document, since they do not hold any value also can affect sentence splitting
	text = re.sub('(Mr|Mrs|Ms)[.]', "",text)

	#basic splitting based on period, full stops and exclamation mark, based on zero or more spaces, we could make the splitting more
	#robust, by handling cases such as U.S.A which would fail in this regex
	sentences = re.split(r' *[\.\?!][\'"\)\]]* *', text)

	return sentences


def get_POS_tags(sentences):

	pos_tagged_sentences =[]
	for sent in sentences:
		#tokenizing words using NLTK
		tokens = nltk.word_tokenize(sent)
		#using NLTK POS tagger to tag parts of speech
		pos_tagged_sentences.append(nltk.pos_tag(tokens))

	return pos_tagged_sentences

def get_progressive_sentences(pos_tagged_sentences,original_sentences):

	progressive_sentences = []
	non_progressive_sentences = []
	#storing all the preceding verbs against which we would be comparing our values
	preceding_verb = ["am", "is", "are", "be", "been", "being", "was", "were", "'s","'re","'m"]
	#looping through pos tagged sentences and adding sentences to progressive if they are preceded by "to be" type verb (win_size =2)
	for index,sent in enumerate(pos_tagged_sentences):
		for wordIndex,word in enumerate(sent):
			#flag to check if the sentence is progressive or not
			prog_sentence_found = False
			if(word[1].startswith('VB') and word[0].endswith("ing")):
				#checking for two preceding words
				preceding_word = sent[wordIndex - 1][0]
				preceding_word_2 = sent[wordIndex - 2][0]
				if(preceding_word in preceding_verb or preceding_word_2 in preceding_verb ):
					prog_sentence_found = True
					break
		if(prog_sentence_found):
			progressive_sentences.append(original_sentences[index])
		else:
			non_progressive_sentences.append(original_sentences[index])


	print("Number of progressive sentences",len(progressive_sentences))
	print("Number of non-progressive sentences",len(non_progressive_sentences))
	return (progressive_sentences,non_progressive_sentences)

def model_train(progressive_sentences,non_progressive_sentences):

	#for model, I used SVM since it is widely used for text classification. I've also used a simple Bag of words model, which could be
	#replaced by word embedding techniques such as word2vec and glove embeddings to get a better representation of data and context

	#concatenating data to make training and testing data
	X = progressive_sentences + non_progressive_sentences

	#progressive sentences are denoted by '1' and non-progressive as '0'
	y = np.concatenate((np.ones(len(progressive_sentences)),np.zeros(len(non_progressive_sentences))),axis=0)

	#splitting into train:test using a 80:20 split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

	#TF-idf to identify important words in the document other than just basing it on the frequency of words
	text_clf = Pipeline([
		('vect', CountVectorizer()),
		('tfidf', TfidfTransformer()),
		('clf', SVC(kernel="sigmoid",gamma='auto',probability=True)),
	])

	text_clf.fit(X_train, y_train)

	#evaluation
	predicted = text_clf.predict(X_test)
	predicted_proba = text_clf.predict_proba(X_test)
	print("Average Accuracy ", np.mean(predicted == y_test))

	#checking ROC AUC score to check how our classifier is separating classes
	print("ROC AUC Score for the classifier ",roc_auc_score(y_test, predicted_proba[:,1]))

	return text_clf

def predict_sentence(test_data,model):

	y_test = model.predict(test_data)
	y_test_prob = model.predict_proba(test_data)
	print("Probability score", y_test_prob)

	print(y_test)
	if(y_test[0] == "1.0"):
		print("It is a Progressive Sentence")
	else:
		print("It is not a Progressive Sentence")

def main():
	#reading two files to increase training data
	file1 = open("2nd_Gore-Bush.txt", "r")
	file2 = open("3rd_Bush-Kerry.txt", "r")
	textdata = file1.read()+file2.read()

	#getting parsed sentence data
	sentences = parse_data(textdata)
	#getting pos_tagged sentences
	pos_tagged_sentences = get_POS_tags(sentences)
	#getting progressive and non-progressive sentences by pos_tagged sentences
	(progressive_sentences,non_progressive_sentences) = get_progressive_sentences(pos_tagged_sentences,sentences)

	#training a classifier to identify a progressive sentence
	model = model_train(progressive_sentences,non_progressive_sentences)

	#test sentence, we can change it to any other sentence
	test_sentence = "As Jim Lehrer told you before the first one, these debates are sponsored by the Commission on Presidential Debates"

	predict_sentence([test_sentence],model)


if __name__=='__main__':
	main()


