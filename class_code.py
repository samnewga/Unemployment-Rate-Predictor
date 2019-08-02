import pandas as pd
# import warnings filter
# Not needed for Pandas
from warnings import simplefilter
# Not needed for Pandas
# warnings filter
simplefilter(action='ignore', category=FutureWarning)
print()
# intake data
train = pd.read_csv("phoenixazunemployment.csv")
train.head()

# Create list of
# Features our model will learn from
print("_" * 175)

print("By using the following data in columns")
print("'Total Trip Minutes', 'PickUpHour'")
print("We were able to predict correctly by a certain percentage if the pick up would be at the airport or in the city.")
feature_cols = ['year', 'unemployment_rate']
# feature_cols = ['PickUpHour']
# feature_cols = ['TotalTripMinutes']

# Features X
X = train.loc[:, feature_cols]

# SciKit Learn understand the shape as long as they are fully numeric
# and the right shape.

# Print out the shape of the data
# This returns the # of rows by # of columns
print()
# print("Verify the shape of the data")
# print("This returns the # of rows by # of columns")
# print(X.shape)
# print(X.reindex)

# Label for prediction
# using y for labels
y = train.year
# print(y.shape)
# print(y.reindex)

# Build the scikit Model
# LogisticRegression Model
# Build the classification model
# Import the lib LogisticRegression which is a classification model
# USE from sklearn.linear_model import LogisticRegression
# Create an instance of our model
# USE logreg = LogisticRegression()
# Fit model to training data
# USE print(logreg.fit(X, y))

# Split up the data into training set and a testing set
# Critical to set data from pandas to data frame which
# SciKit can work with.
# df = pd.DataFrame(train)
# print(df)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

# Start classifier specific code
from sklearn import tree

dt_classifier = tree.DecisionTreeClassifier()

from sklearn.neighbors import KNeighborsClassifier

kn_classifier = KNeighborsClassifier()

from sklearn.linear_model import LogisticRegression

logreg_classifier = LogisticRegression()

from sklearn.neural_network import MLPClassifier

ANN_classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

# End classifier specific code

# Train the classifier using training data.
dt_classifier.fit(X_train, y_train)
kn_classifier.fit(X_train, y_train)
logreg_classifier.fit(X_train, y_train)
ANN_classifier.fit(X_train, y_train)
# Use it to classify our test data.
dt_predictions = dt_classifier.predict(X_test)
kn_predictions = kn_classifier.predict(X_test)
logreg_predictions = logreg_classifier.predict(X_test)
ANN_predictions = ANN_classifier.predict(X_test)

# Actual predictions on our test data
# 1 = Student Enrolled Positive Outcome is a yes
# 0 = Student NOT Enrolled Positive Outcome is a no
# Too many to print out, so it prints abbreviated
# print(dt_predictions)
# Determine here how accurate our classifier was on the testing set
# This compares the predictive labels to the true labels aka y_test
# Then calculate the accuracy score
from sklearn.metrics import accuracy_score

print("The accuracy score for the DT Classifier is:")
print(accuracy_score(y_test, dt_predictions))
print("The accuracy score for the KN Classifier is:")
print(accuracy_score(y_test, kn_predictions))
print("The accuracy score for the LR Classifier is:")
print(accuracy_score(y_test, logreg_predictions))
print("The accuracy score for the ANN Classifier is:")
print(accuracy_score(y_test, ANN_predictions))

print()
print("*" * 25, " End of Line ", "*" * 25)
print()
