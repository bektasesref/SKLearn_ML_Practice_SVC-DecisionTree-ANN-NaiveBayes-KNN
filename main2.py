import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='micro')
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return y_pred, accuracy, f1, report, cm

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def plot_actual_vs_predicted(y_test, y_pred, model_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted ({model_name})')
    plt.show()

def plot_actual_vs_predicted_line(y_test, y_pred, model_name):
    plt.figure(figsize=(8, 6))
    plt.plot(y_test, label='Actual', linestyle='-')
    plt.plot(y_pred, label='Predicted', linestyle='--')
    plt.xlabel('Sample Index')
    plt.ylabel('Values')
    plt.title(f'Actual vs Predicted Line Plot ({model_name})')
    plt.legend()
    plt.show()

def plot_actual_vs_predicted_bar(y_test, y_pred, model_name):
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(y_test)), y_test, label='Actual', alpha=0.6)
    plt.bar(range(len(y_pred)), y_pred, label='Predicted', alpha=0.6)
    plt.xlabel('Sample Index')
    plt.ylabel('Values')
    plt.title(f'Actual vs Predicted Bar Plot ({model_name})')
    plt.legend()
    plt.show()

def ReadExcel(excelFileName, yColumn, xColumns):
    dataset = pd.read_excel(r'' + excelFileName + '')

    X = dataset.iloc[:, xColumns].values
    y = dataset.iloc[:, yColumn].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    models = {
        'SVC': SVC(probability=True),
        'DecisionTree': DecisionTreeClassifier(),
        'ANN': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000),
        'NaiveBayes': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=3)
    }

    results = {}

    for model_name, model in models.items():
        y_pred, accuracy, f1, report, cm = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
        results[model_name] = {
            'y_pred': y_pred,
            'accuracy': accuracy,
            'f1': f1,
            'report': report,
            'cm': cm
        }

        # Plot actual vs predicted values for each model
        plot_actual_vs_predicted(y_test, y_pred, model_name)
        plot_actual_vs_predicted_line(y_test, y_pred, model_name)
        plot_actual_vs_predicted_bar(y_test, y_pred, model_name)

    return results

results = ReadExcel("normalized.xlsx", 2, [0, 1])

class_names = sorted(results['SVC']['y_pred'].unique())

for model_name, result in results.items():
    print(f"Model: {model_name}")
    print(f"Accuracy: {result['accuracy']}")
    print(f"F1 Score: {result['f1']}")
    print("Classification Report:")
    print(result['report'])
    print("\n")
    plot_confusion_matrix(result['cm'], class_names)
