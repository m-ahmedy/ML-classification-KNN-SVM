#%%
# Preprocessing dataset
from classification_preprocessor import preprocessor
X_train, X_test, y_train, y_test = preprocessor('Social_Network_Ads.csv', [2, 3], 4)
#%%
# Fitting classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
# p = 2 means the model uses euclidean distance, p = 1 uses manhattan distance
# n_neighbors: The number of neighboring nodes taken into account 
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
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
visualizer('KNN Classification', classifier, X_train, y_train, 'training')
#%%
visualizer('KNN Classification', classifier, X_test, y_test, 'test')
#%%