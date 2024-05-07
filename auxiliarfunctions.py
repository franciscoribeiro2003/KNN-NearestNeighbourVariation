import pandas as pd
from sklearn.base import clone
from sklearn import neighbors
from sklearn.calibration import LabelEncoder
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np
import logging


def preprocess_data(X, y):
    # Convert categorical variables to dummy variables
    X = pd.get_dummies(X, drop_first=True)
    
    # Convert bool columns to int
    X = X.astype({col: 'int' for col in X.select_dtypes(['bool']).columns})
    
    # Convert y to numeric, turning non-numeric values into NaN

     # Map categorical labels to numerical values based on their categories
    label_map = {label: idx for idx, label in enumerate(y.cat.categories)}
    y_numeric = y.map(label_map)

    
    return X, y_numeric



logging.getLogger('matplotlib').setLevel(logging.WARNING)

def fetch_and_prepare_dataset(dataset_id):
    dataset = fetch_openml(data_id=dataset_id)
    X = dataset.data
    y = dataset.target
    X, y = preprocess_data(X, y)
    return X, y


# Function to apply PCA and plot the result
def apply_pca_and_plot_with_encoding(X, y):
    # Encoding categorical target variables
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Standardizing the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Applying PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], c=y_encoded, cmap='viridis', alpha=0.5)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('2 Component PCA')
    plt.colorbar(scatter)
    plt.show()



def draw_data(X, y, xn, xy):
    # Convert y to numeric values if it's not already
    y = pd.to_numeric(y, errors='coerce')
    X = X.dropna(subset=[X.columns[xn], X.columns[xy]])

    # Create a 2D histogram
    plt.hist2d(X.iloc[:, xn], X.iloc[:, xy], bins=12, cmap='viridis')
    plt.title('Dataset Visualization')
    plt.xlabel(X.columns[xn])
    plt.ylabel(X.columns[xy])
    plt.colorbar(label='Count')
    plt.show()



def decision_plot_function(X, y, model1, xn=0, xy=1):
    # Ensure X is a NumPy array if it isn't already (in case it's a DataFrame)
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    # Fit the model on the selected features
    X_selected = X[:, [xn, xy]]
    
    model = clone(model1)
    model.fit(X_selected, y)
    
    # Create a mesh grid on which we will use model to predict class labels
    x_min, x_max = X_selected[:, 0].min() - 1, X_selected[:, 0].max() + 1
    y_min, y_max = X_selected[:, 1].min() - 1, X_selected[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Predict class labels for each point on the grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training points
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X_selected[:, 0], X_selected[:, 1], c=y, edgecolors='k', marker='o', s=50)
    plt.xlabel(f'Feature {xn}')
    plt.ylabel(f'Feature {xy}')
    plt.title('Decision Boundary Plot')
    plt.show()

from sklearn.inspection import permutation_importance

def features_importance_plot(model, X, y, feature_names=None):
    # If feature names are not provided, use indices
    if feature_names is None:
        feature_names = np.arange(X.shape[1])

    # Convert X to DataFrame if it's a NumPy array
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=feature_names)

    # Get the feature importances from the model
    results = permutation_importance(model, X, y, scoring='accuracy')
    importances = results.importances_mean

    # Sort the feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Create a plot with the feature importances
    plt.figure(figsize=(8, 6))
    plt.bar(feature_names[indices], importances[indices], align='center')
    plt.xlabel('Feature')
    plt.ylabel('Feature Importance')
    plt.title('Feature Importances')
    plt.show()