# SKLearn_ML_Practice_SVC-DecisionTree-ANN-NaiveBayes-KNN
 Machine Learning practice by using SKLearn, that used couple of models such as Decision Tree, ANN, NaiveBayes, SVC, KNN. Also prints accuracy, F1 Score, gets ROC curve and plots confusion matrix
 
Contains 2 different approach, main2.py is more modular and multiple & burst at once

"new_features.xlsx" and "normalized.xlsx" files are test purpose only

# Which framework/libraries used?
scikit-learn, pandas, matplotlib


# What classification models does this practice include?
DecisionTree, Artificial Neural Network (ANN), NaiveBayes, Support Vector Machines (SVC), K-Nearest Neighbors (KNN)

Generates graphs for dotted, line and bar versions for each model that entered in main2.py's;
   models = {
        'SVC': SVC(probability=True),
        'DecisionTree': DecisionTreeClassifier(),
        'ANN': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000),
        'NaiveBayes': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=3)
    }
	
line. This is the kinda entry point that which model's are gonna main2 is gonna use the predict, gets F1, Accuracy, Classification Report and lastly confusion matrix at the end of the phase