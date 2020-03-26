#%%
# Preprocessing dataset
from classification_preprocessor import preprocessor
X_train, X_test, y_train, y_test = preprocessor('Social_Network_Ads.csv', [2, 3], 4)
#%%
# Fitting classifier to the Training set
from sklearn.svm import SVC
# kernel = 'linear' means the model uses linear kenel machine, other options are 'sigmoid', 'poly', 'rbf' etc
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
#%%

# Predicting the Test set results
y_pred = classifier.predict(X_test)
#%%

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#%%

# Visualizing results
from classification_visualizer import visualizer
visualizer('SVM Classification', classifier, X_train, y_train, 'training')
#%%
visualizer('SVM Classification', classifier, X_test, y_test, 'test')
#%%

# %%
