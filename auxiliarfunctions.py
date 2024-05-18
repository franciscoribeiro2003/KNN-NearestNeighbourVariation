from sklearn.calibration import LabelEncoder
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from IPython.display import display, HTML

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def display_accuracy_dataset(accuracy, models):
    ''' 

        Parameters
        ----------
        - accuracy: 
        - models: 

        Return
        ----------
        display 
    '''
    plt.plot(accuracy)
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.title("Comparison of results")
    plt.xticks(range(len(accuracy)), models, rotation='horizontal')
    plt.show()

def display_graphic_for_k(Ks, accuracy, classifiers_names_k):
    ''' 

        Parameters
        ----------
        - Ks: Tested values of k
        - accuracy: 
        - models: 

        Return
        ----------
        display 
    '''
    _, ax = plt.subplots()
    for name in classifiers_names_k:
        result = np.mean(accuracy[name][0:9]), np.mean(accuracy[name][10:19]) ,np.mean(accuracy[name][20:29])
        ax.plot(Ks, result, label=name)

    ax.set_xlabel('K')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy of Classifiers for Different K Values')
    ax.set_xticks(Ks)
    ax.legend()
    plt.show()

def display_fetch_data(name, i):
    ''' 
        Fetch the data from the dataset with id i from openml, preprocess 
        the data and display the data before and after processing

        Parameters
        ----------
        - name: name of the dataset from openml
        - i: id of dataset from openml

        Return
        ----------
        return the data (X, y) from dataset i
    '''
    
    dataset = fetch_openml(data_id = i)

    X = dataset.data
    y = dataset.target

    data = X, y

    print(f"Dataset: {name}")
    
    print(f"Before processing the dataset:")
    table_data(data)
    
    X,y = preprocess_data(X, y)
    data = X, y

    print(f"After processing the dataset:")
    table_data(data)
    
    return data


def table_data(data):
    '''
        Display the first 5 rows of the dataset
        
        Parameters
        ----------
        - data: tuple (X, y) or DataFrame

        Return
        ----------
        None
    '''

    # Verifica se data é uma tupla (X, y)
    if isinstance(data, tuple) and len(data) == 2:
        X, y = data
        # Verifica se X e y são DataFrames ou Series e os transforma em DataFrame se necessário
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)
        # Concatena X e y
        df = pd.concat([X, y], axis=1)
    else:
        # Se data não é uma tupla, assume que é um DataFrame único
        df = pd.DataFrame(data)
    # Exibe o nome do dataset
    
    display(HTML(df.head(5).to_html()))
    print("\n")


def is_continuous(series):
    '''
        Check if the series is continuous or categorical

        Parameters
        ----------
        - series: pd.Series
            
        Return  
        ----------
        - True if the series is continuous, False otherwise
    '''

    unique_values = series.nunique()
    total_values = len(series)
    
    # We assume that if there are many unique values, the data is continuous
    # Here, 10% is an arbitrary threshold that can be adjustedtínuo
    if unique_values / total_values > 0.1:
        return True
    else:
        return False


def preprocess_data(X, y):
    '''
        Preprocess the data by converting values to numeric and encoding categorical variables

        Parameters
        ----------
        - X: pd.DataFrame
        - y: pd.Series

        Return
        ----------
        - X: pd.DataFrame
        - y_numeric: pd.Series
    '''

    # Convert categorical variables to dummy variables
    X = pd.get_dummies(X, drop_first=True)
    
    # Convert bool columns to int
    X = X.astype({col: 'int' for col in X.select_dtypes(['bool']).columns})

    # Check if y is of type 'numeric'
    if pd.api.types.is_numeric_dtype(y):
        if is_continuous(y):
            y = y.astype('category')
            label_map = {label: idx for idx, label in enumerate(y.cat.categories)}
            y_numeric = y.map(label_map)
        else:
            y_numeric = y
            
    # Type categorical    
    else:
        try: 
            y_numeric =pd.to_numeric(y)
        except ValueError:
            label_map = {label: idx for idx, label in enumerate(y.cat.categories)}
            y_numeric = y.map(label_map)
    
    return X, y_numeric


def apply_pca_and_plot_with_encoding(X, y,name):
    '''
        Apply PCA to the data and plot the result 

        Parameters
        ----------
        - X: pd.DataFrame
        - y: pd.Series
        - name: str

        Return
        ----------
        None
    '''

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
    plt.title(f'Dataset: {name}')
    plt.show()


def plot_accuracies(datasets_names,list_accuracies):
    '''
        Showcase the accuracies for different datasets and KNN variants in a plot

        Parameters
        ----------
        - datasets_names: list of str
        - knn_accuracies: list of float
        - knn_isoforest_accuracies: list of float
        - knn_lof_accuracies: list of float
        - knn_isoforest_imp_accuracies: list of float
        - knn_lof_imp_accuracies: list of float

        Return
        ----------
        None
    '''

    x = np.arange(len(datasets_names))  # the label locations

    fig, ax = plt.subplots(figsize=(15, 8))

    # Plotting the lines for each classifier
    ax.plot(datasets_names, list_accuracies["KNN"], marker='o', label='KNN')
    ax.plot(datasets_names, list_accuracies["KNN Modified with Isolation Forest without importance"], marker='o', label='KNN w/ Isolation Forest w/o importance')
    ax.plot(datasets_names, list_accuracies["KNN Modified with Local Outlier Factor without importance"], marker='o', label='KNN w/ Local Outlier Factor w/o importance')
    ax.plot(datasets_names, list_accuracies["KNN Modified with Isolation Forest"], marker='o', label='KNN w/ Isolation Forest')
    ax.plot(datasets_names, list_accuracies["KNN Modified with Local Outlier Factor"], marker='o', label='KNN w/ Local Outlier Factor')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Datasets')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy for Different Datasets and KNN Variants')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets_names, rotation=45, ha="right")
    ax.legend()

    fig.tight_layout()

    plt.show()
