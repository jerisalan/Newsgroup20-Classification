import re
import string
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn import feature_extraction
from sklearn.datasets import load_files
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords


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
    print('Confusion matrix for: %s\n'%learning_algo)
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
   
    
docs_train = load_files('train',categories = category, encoding='latin-1') 
docs_train = remove_email_address(docs_train)
docs_train = remove_specific_words(docs_train)
docs_test = load_files('test',categories = category, encoding='latin-1')
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