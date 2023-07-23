from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import hstack

# Load data
data = pd.read_csv('enron_spam_data.csv')

data.drop(columns=['Message ID'], inplace=True)
data.dropna(inplace=True)

# Vectorize the Subject column and the Message column
vectorizer = TfidfVectorizer(stop_words='english')
X_subject = vectorizer.fit_transform(data['Subject'])
X_message = vectorizer.transform(data['Message'])

# Combine the sparse matrices
X = hstack([X_subject, X_message])

# Label the spam column
y = data['Spam/Ham']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a multinomial Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Evaluate the classifier on the testing set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)