# Practical Machine learning
# Regression Analysis - Logistic Regression example 
# Chapter 10
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, hamming_loss, jaccard_similarity_score

# Read the data and see what is in there
df = pd.read_csv('data/SMSSpamCollection', delimiter='\t', header=None)
print df.head()
print 'Number of spam messages:', df[df[0] == 'spam'][0].count()
print 'Number of ham messages:', df[df[0] == 'ham'][0].count()

# Using Logistic Regression function, predict for the data in sms.csv
df = pd.read_csv('data/sms.csv')
X_train_raw, X_test_raw, y_train, y_test = train_test_split(df['message'], df['label'])
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

for i, prediction in enumerate(predictions[:5]):
    print X_test_raw[i], 'prediction:', prediction


# Print the confusion matrix for the given data
y_test = []
y_pred = []
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
plt.matshow(confusion_matrix)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#print the accuracy score
y_pred, y_true = [0, 1, 1, 0], [1, 1, 1, 1]
print accuracy_score(y_true, y_pred)


# Evaluate the SMS Classifier 
df = pd.read_csv('data/sms.csv')
X_train_raw, X_test_raw, y_train, y_test = train_test_split(df['message'], df['label'])
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
scores = cross_val_score(classifier, X_train, y_train, cv=5)
print np.mean(scores), scores
#print precision and recall values
precisions = cross_val_score(classifier, X_train, y_train, cv=5, scoring='precision')
print 'Precision', np.mean(precisions), precisions
recalls = cross_val_score(classifier, X_train, y_train, cv=5, scoring='recall')
print 'Recall', np.mean(recalls), recalls
#print F1 score
f1s = cross_val_score(classifier, X_train, y_train, cv=5, scoring='f1')
print 'F1', np.mean(f1s), f1s

#map the predictions
X_train_raw, X_test_raw, y_train, y_test = train_test_split(df['message'], df['label'])
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
predictions = classifier.predict_proba(X_test)
false_positive_rate, recall, thresholds = roc_curve(y_test, predictions[:, 1])
roc_auc = auc(false_positive_rate, recall)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.show()
#use the train data
df = pd.read_csv('data/train.tsv', header=0, delimiter='\t')
print df.count()
print df.head(10)
print df['Phrase'].head()
print df['Sentiment'].describe()
print df['Sentiment'].value_counts()
print df['Sentiment'].value_counts()/df['Sentiment'].count()

# Multi-Class Classification Performance Metrics
predictions = grid_search.predict(X_test)
print 'Accuracy:', accuracy_score(y_test, predictions)
print 'Confusion Matrix:', confusion_matrix(y_test, predictions)
print 'Classification Report:', classification_report(y_test, predictions)

print hamming_loss(np.array([[0.0, 1.0], [1.0, 1.0]]), np.array([[0.0, 1.0], [1.0, 1.0]]))
print hamming_loss(np.array([[0.0, 1.0], [1.0, 1.0]]), np.array([[1.0, 1.0], [1.0, 1.0]]))
print hamming_loss(np.array([[0.0, 1.0], [1.0, 1.0]]), np.array([[1.0, 1.0], [0.0, 1.0]]))
print jaccard_similarity_score(np.array([[0.0, 1.0], [1.0, 1.0]]), np.array([[0.0, 1.0], [1.0, 1.0]]))
print jaccard_similarity_score(np.array([[0.0, 1.0], [1.0, 1.0]]), np.array([[1.0, 1.0], [1.0, 1.0]]))
print jaccard_similarity_score(np.array([[0.0, 1.0], [1.0, 1.0]]), np.array([[1.0, 1.0], [0.0, 1.0]]))