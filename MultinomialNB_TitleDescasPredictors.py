import pandas as pd
import numpy as np
import pylab as pl
import nltk
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk import WordNetLemmatizer

stemmer = LancasterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()

training_data = pd.read_csv('oneyearcategorized.csv', encoding = "latin1")
training_data['Title'] = training_data['Title'].astype(str)
training_data['Description'] = training_data['Description'].astype(str)
training_data['title_and_description'] = training_data[['Title', 'Description']].apply(tuple, axis=1)
training_data = training_data.astype(str)

training_data['title_and_description'] = training_data['title_and_description'].astype(str)

train_data = training_data.to_dict('records')
train_data

corpus_words ={}
class_words = {}
unnecessary_words = ['please','it','we','hi','is',"'s",'?',',',':','..','.','|','#','-','<','>','(',')','{','}']

classes =  list(training_data['Cat'].unique()) # -- for dataframe
#classes = list(set([a['category'] for a in train_data])) -- for lists
for c in classes:
    class_words[c] = []

for data in train_data:
    for word in nltk.word_tokenize(data['title_and_description']):
    
        if word not in unnecessary_words:
            if word not in stopwords.words('english'):
                
                stemmed_word = stemmer.stem(word.lower())
                if stemmed_word not in corpus_words:
                    corpus_words[stemmed_word] = 1
                else:
                    corpus_words[stemmed_word] += 1
                    
                class_words[data['Cat']].extend([stemmed_word])
                
print("Corpus words and counts: %s \n" % corpus_words)

print("Class words: %s" % class_words)

title = "Probe is down"

def calculate_class_score(title, class_name, show_details=True):
    score = 0
    
    for word in nltk.word_tokenize(title):
        if stemmer.stem(word.lower()) in class_words[class_name]:
            score += 1
            
            if show_details:
                print("  match: %s" % stemmer.stem(word.lower()))
    return score


for c in class_words.keys():
    print ("Class: %s   Score: %s \n" % (c, calculate_class_score(title, c)))

    def calculate_class_score_commonality(title, class_name, show_details=True):
    score = 0
    
    for word in nltk.word_tokenize(title):
        
        if stemmer.stem(word.lower()) in class_words[class_name]:
            score += (1 / corpus_words[stemmer.stem(word.lower())])
            
            if show_details:
                print("   match: %s (%s)" % (stemmer.stem(word.lower()), 1/corpus_words[stemmer.stem(word.lower())]))
    return score

    for c in class_words.keys():
    print("Class: %s   Score: %s \n" % (c, calculate_class_score_commonality(title, c)))

    def classify (title):
    high_class = None
    high_score = 0
    
    for c in class_words.keys():
        
        score = calculate_class_score_commonality(title, c, show_details=False)
        
        if score > high_score:
            high_class = c
            high_score = score
            
    return high_class, high_score

    classify("netprobe always down")