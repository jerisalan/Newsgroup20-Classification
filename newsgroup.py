import re
import string
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn import feature_extraction
from sklearn.datasets import load_files
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from matplotlib import pyplot as plt
import numpy as np
import itertools


__stemmer = nltk.stem.SnowballStemmer("english")
__words_only = re.compile("^[A-Za-z]*$")
EMAIL_REGEX = re.compile(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
category = ['science','religion','recreation','politics','computer','misc' ]
other_stopwords = ['article','bit','book','chip','day','doe','ha','hand','line','lot','number','organization', 'person','place','point','problem','question','time','university','version','wa','way', 'week','work','world','year','u']
stopwords_set = set(stopwords.words('english'))
stopwords_set.update(other_stopwords)


def remove_email_address(docs):
    for i in range(0,len(docs.data)):
    	split_words = docs.data[i].split('\n')   
    	docs.data[i] = ""   
    	for j in range(0,len(split_words)):
    		if EMAIL_REGEX.search(split_words[j]) is None:
    			docs.data[i] += split_words[j] +'\n'
    return docs
    

def remove_specific_words(docs):
    for i in range(0,len(docs.data)):
        split_words = word_tokenize(docs.data[i])
        docs.data[i] = ""
        for j in range(0, len(split_words)):
            if split_words[j].lower() not in stopwords_set:
                docs.data[i] += split_words[j] + ' '
    return docs


def remove_punctuation(s):
    if s not in string.punctuation:
        return True
    return False


def remove_stop_word(s):
    if s not in feature_extraction.text.ENGLISH_STOP_WORDS:
        return True
    return False


def clean_word(s):
    result = ""
    if s is not None:
        for w in nltk.tokenize.word_tokenize(s.lower()):
            if w is not None and remove_stop_word(w) and remove_punctuation(w) and regex_filter(w):
                result += " " + __stemmer.stem(w)
    return result

    
def regex_filter(s):
    if __words_only.match(s) is not None:
        return True
    return False


def setup_pipeline(learning_algo, tf_ind):
    if(tf_ind == 1):
        tf_idf = TfidfVectorizer(preprocessor=clean_word, ngram_range=(1, 3), use_idf=True, min_df=3, max_df=0.5)
    else:
        tf_idf = CountVectorizer(preprocessor=clean_word, ngram_range=(1, 3), min_df=3, max_df=0.5)
    lsa = TruncatedSVD(n_components=250, n_iter=5, random_state=25)
    pipeline_list = [('tf_idf', tf_idf), ('svd', lsa), ('learning_algo', learning_algo)]
    pipeline = Pipeline(pipeline_list)
    return pipeline

    
def print_stats(expected, predicted, learning_algo):
    print('Confusion matrix for: %s'%learning_algo)
    print(metrics.confusion_matrix(expected, predicted))
    print('Actual Recall')
    print(metrics.recall_score(expected, predicted, average='macro'))
    print('Actual Accuracy Score')
    print(metrics.accuracy_score(expected, predicted))
    print('Actual Precision Score')
    print(metrics.precision_score(expected, predicted, average='macro'))
    print('Actual F1 Score')
    print(metrics.f1_score(expected, predicted, average='macro'))
    print '\n'

#Pretty print confusion matrix  
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
 

docs_train = load_files('data/train',categories = category, encoding='latin-1')
#Remove email addresses and specific set of stop words adding bias
docs_train = remove_email_address(docs_train)
docs_train = remove_specific_words(docs_train)
docs_test = load_files('data/test',categories = category, encoding='latin-1')
#Remove email addresses and specific set of stop words adding bias
docs_test = remove_email_address(docs_test)
docs_test = remove_specific_words(docs_test)

            
print('Creating MLP Object')
perceptron = MLPClassifier(verbose=True, solver='adam', alpha=0.001, hidden_layer_sizes=(20,), random_state=1)
print('Created MLP Object')
print('Creating MLP Pipeline')
perceptron_pp = setup_pipeline(perceptron, 1)
print('Created MLP Pipeline')
print('Creating MLP Model')
perceptron_fit = perceptron_pp.fit(docs_train.data, docs_train.target)
print('Created MLP Model')
print('Predicting Values')
perceptron_pred = perceptron_fit.predict(docs_test.data)
print('Printing results')
print_stats(docs_test.target,perceptron_pred,'Multi Layer Perceptron')

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(metrics.confusion_matrix(docs_test.target, perceptron_pred), classes=category, title='Confusion matrix')
plt.show()

#Classifaction using Multiclass SVM
print('Creating SVM Object')
svm_basic = SVC(kernel='linear', class_weight='balanced', probability=True, random_state=40)
print('Created SVM Object')
print('Creating SVM OVR Classifier')
svm_onerest = OneVsRestClassifier(svm_basic)
print('Created SVM OVR Classifier')
print('Creating SVM pipeline')
pipeline_svm_onerest = pipeline_setup(svm_onerest)
print('Created SVM pipeline')
print('Fitting model with training data')
pipeline_svm_fitted = pipeline_svm_onerest.fit(docs_train.data, docs_train.target)
print('Fitted model and predicting now...')
svm_predict = pipeline_svm_fitted.predict(docs_test.data)
print_stats(docs_test.target, svm_predict, 'SVM OneVSRest')

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(metrics.confusion_matrix(docs_test.target, svm_predict), classes=category, title='Confusion matrix')
plt.show()
