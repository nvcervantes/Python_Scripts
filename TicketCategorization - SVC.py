#==========================================
# Title:  Ticket Categorization for Citi
# Author: Shierene Cervantes
# Created:   7 Jan 2019
# Last Modified: 22 Feb 2019
#==========================================
# python script


import pandas as  pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from datetime import datetime
import sys
reload(sys)
sys.setdefaultencoding('utf8')

df = pd.read_csv('training_data.csv', encoding="latin1")
cols = ['CATEGORY', 'title_and_description']
df = df[cols]
df = df[pd.notnull(df['title_and_description'])]
df.columns = ['CATEGORY', 'title_and_description']
df['Number'] = df['CATEGORY'].factorize()[0]
category_id_df = df[['CATEGORY','Number']].drop_duplicates().sort_values('Number')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['Number','CATEGORY']].values)
df.head()

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=1, norm='l2', encoding='latin-1', ngram_range=(1,2), stop_words='english')
features = tfidf.fit_transform(df.title_and_description).toarray()
labels = df.Number
features.shape

X_train, X_test, y_train, y_test = train_test_split(df['title_and_description'],df['CATEGORY'],random_state=0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
svc = LinearSVC().fit(X_train_tfidf, y_train)

tf  = pd.read_csv('Source.csv', encoding="latin1")
tf['Linux_Predicted'] = svc.predict(count_vect.transform(tf['title_and_description'].values.astype('U')))

now = datetime.now()
def _getNow():
	return now.strftime("%Y%m%d_%H%M%S")
filename = "%s_%s.%s" % ("MLLinux", _getNow(), "csv")
path = '/scervantes/files'
tf.to_csv(path+filename, index = False)
