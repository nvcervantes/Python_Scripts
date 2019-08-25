import pandas as pd
tr_data = pd.read_csv('categorization_data.csv', encoding="latin1")

tr_data['title_and_description'] = tr_data[['Title', 'Description']].apply(tuple, axis=1)

tr_data.to_csv('categorizationdata_withtad.csv')

from io import StringIO
col = ['CATEGORY', 'title_and_description']
tr_data = tr_data[col]
tr_data = tr_data[pd.notnull(tr_data['title_and_description'])]
tr_data.columns = ['CATEGORY', 'title_and_description']
tr_data['Number'] = tr_data['CATEGORY'].factorize()[0]
category_id_df = tr_data[['CATEGORY', 'Number']].drop_duplicates().sort_values('Number')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['Number', 'CATEGORY']].values)
tr_data.head()

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
tr_data.groupby('CATEGORY').title_and_description.count().plot.bar(ylim=0)
plt.show()

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(tr_data.title_and_description).toarray()
labels = tr_data.Number
features.shape

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

X_train, X_test, y_train, y_test = train_test_split(tr_data['title_and_description'], tr_data['CATEGORY'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)
svc = LinearSVC().fit(X_train_tfidf, y_train)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 2
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
import seaborn as sns
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()

cv_df.groupby('model_name').accuracy.mean()

model = LinearSVC()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, tr_data.index, test_size=0.4, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.CATEGORY.values, yticklabels=category_id_df.CATEGORY.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

from sklearn import metrics
print(metrics.classification_report(y_test, y_pred, target_names=tr_data['CATEGORY'].unique()))

test_set = pd.read_csv('categorizationdata_withtad.csv', encoding="latin1")
test_set

test_set['Combined MNB New Predicted Values'] = clf.predict(count_vect.transform(test_set['title_and_description'].values.astype('U')))

test_set['Combined SVC New Predicted Values'] = svc.predict(count_vect.transform(test_set['title_and_description'].values.astype('U')))

test_set['Title MNB New Predicted Values'] = clf.predict(count_vect.transform(test_set['Title'].values.astype('U')))

test_set['Title SVC New Predicted Values'] = svc.predict(count_vect.transform(test_set['Title'].values.astype('U')))

test_set['comparetitlesvc'] = test_set['Combined SVC New Predicted Values'] == test_set['CATEGORY']

test_set['comparetitlesvc'].value_counts(normalize=True)*100

test_set.to_csv('ModelComparison_SVC_MNB.csv')