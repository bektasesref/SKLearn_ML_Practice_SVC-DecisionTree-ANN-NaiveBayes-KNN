import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, classification_report,confusion_matrix

y_pred = 0

# region Models
def modelDecisionTree(X_train, y_train, X_test):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    global y_pred
    y_pred = model.predict(X_test)
    return model

def modelANN(X_train, y_train, X_test): # Artificial Neural Network
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
    model.fit(X_train, y_train)
    global y_pred
    y_pred = model.predict(X_test)
    return model

def modelNaiveBayes(X_train, y_train,X_test):
    model = GaussianNB()
    model.fit(X_train, y_train)
    global y_pred
    y_pred = model.predict(X_test)
    return model

def modelSVC(X_train, y_train, X_test): #Support Vector Machines
    model = SVC(probability=True)
    model.fit(X_train, y_train)
    global y_pred
    y_pred = model.predict(X_test)
    return model


def modelKNN(X_train, y_train,X_test):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    global y_pred
    y_pred = model.predict(X_test)
    return model

# endregion

def PrintAccuracy(y_test,y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

def GetF1Score(y_test, y_pred):
    f1 = f1_score(y_test, y_pred,average='micro')
    print("F1 Score:", f1)

def PrintReport(y_test, y_pred):
    print(classification_report(y_test,y_pred,zero_division=0))

def GetRocCurve(X_test, y_test, model,y_pred):
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])

    auc_score = auc(fpr, tpr)
    print("AUC:", auc_score)

    plt.plot(fpr, tpr, label='ROC Curve (AUC = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

def PlotConfusionMatrix(y_test,y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - SVC')
    plt.show()

def ReadExcel(excelFileName, yColumn,xColumns):
    dataset = pd.read_excel(r'' +excelFileName+'')

    X = dataset.iloc[:, xColumns].values
    y = dataset.iloc[:, yColumn].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    #model = modelSVC(X_train, y_train, X_test)
    #model = modelDecisionTree(X_train,y_train,X_test)
    model = modelANN(X_train,y_train,X_test)
    #model = modelNaiveBayes(X_train,y_train,X_test)

    PrintAccuracy(y_test, y_pred)
    GetF1Score(y_test, y_pred)
    PrintReport(y_test, y_pred)
    PlotConfusionMatrix(y_test,y_pred)
    #GetRocCurve(X_test, y_test, model, y_pred)

ReadExcel("new_features.xlsx",5,[0, 1, 2, 3, 4])
#ReadExcel("normalized.xlsx",2,[0, 1])
