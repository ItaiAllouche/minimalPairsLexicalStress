import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# with open('/app/datasets/balanced_tagged_embeddings.pkl', 'rb') as file:
#     balanced_data = pickle.load(file)

with open('/app/datasets/balanced_tagged_fixed_embeddings.pkl', 'rb') as file:
    balanced_fixed_data = pickle.load(file)

# extract features and labels
# X = np.array([item[0] for item in balanced_data])  # Features
# y = np.array([item[1] for item in balanced_data])  # Labels
X_fixed = np.array([item[0].numpy().flatten() for item in balanced_fixed_data])  # Features
y_fixed = np.array([item[1] for item in balanced_fixed_data])  # Labels

# reshape the features if necessary
# X = X.reshape(X.shape[0], -1)
X_fixed = X_fixed.reshape(X_fixed.shape[0], -1)

# split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_fixed_train, X_fixed_test, y_fixed_train, y_fixed_test = train_test_split(X_fixed, y_fixed, test_size=0.2, random_state=42)

# create and train the SVM model
# model = SVC()
model_fixed = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42)
# model.fit(X_train, y_train)
model_fixed.fit(X_fixed_train, y_fixed_train)

# make predictions on the test set
# y_pred = model.predict(X_test)
y_fixed_pred = model_fixed.predict(X_fixed_test)

# evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
accuracy_fixed = accuracy_score(y_fixed_test, y_fixed_pred)
# print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Accuracy fixed: {accuracy_fixed * 100:.2f}%")


# Finding optimal hyper-param for SVM
# param_grid = {
#     'C': [0.1, 1, 10, 100],
#     'gamma': ['scale', 'auto'],
#     'kernel': ['linear', 'rbf', 'poly'],
#     'class_weight': [None, 'balanced']
# }

# # Initialize the GridSearchCV object
# grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')

# # Fit the grid search to the data
# grid_search.fit(X_train, y_train)

# # Get the best parameters
# best_params = grid_search.best_params_
# print(f"Best parameters: {best_params}")

# # Train the final model with the best parameters
# best_model = grid_search.best_estimator_

# # Make predictions on the test set
# y_pred = best_model.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy * 100:.2f}%")
