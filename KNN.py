import pandas as pd
import matplotlib.pyplot as plt

# Load the Iris flower dataset from a CSV file
data = pd.read_csv("iris.csv")

# Separate the data into features and labels
features = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
labels = data['species']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# Train a K-Nearest Neighbors (KNN) classifier
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn.predict(X_test)

# Evaluate the model's accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Visualize the Iris flower dataset
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_transformed = pca.fit_transform(features)

plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=labels)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Iris Flower Dataset')
plt.show()
