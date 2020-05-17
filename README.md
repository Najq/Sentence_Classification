# Sentence Classification into Progressive and non-Progressive sentences

* Processing ["The Second Gore-Bush Presidential Debate"](http://www.anc.org/MASC/texts/2nd_Gore-Bush.txt) to predict if sentences used were progressive or not
* For training and identifying sentences as progressive, each sentence is checked if it consists of a verb ending with -ing (VBG) is preceded by a form of the verb "to be" (am, is, are, be, been, being, was, were.
* The corpus was split into sentences using a custom sentence splitter
* TF-IDF is used to identify the important words and SVM is used for classification

