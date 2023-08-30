import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

#FUNCTION TO VISUALIZE THE NEURAL NETWORK
def visualise(mlp, plot, label):
    #Get the number of neurons in each layer
    n_neurons = [len(layer) for layer in mlp.coefs_]
    n_neurons.append(mlp.n_outputs_)

    #Calculate the coordinates for each neuron on the graph
    y_range = [0, max(n_neurons)]
    x_range = [0, len(n_neurons)]
    loc_neurons = [[[l, (n+1)*(y_range[1]/(layer+1))] for n in range(layer)] for l,layer in enumerate(n_neurons)]
    x_neurons = [x for layer in loc_neurons for x,y in layer]
    y_neurons = [y for layer in loc_neurons for x,y in layer]

    #Identify the range of weights
    weight_range = [min([layer.min() for layer in mlp.coefs_]), max([layer.max() for layer in mlp.coefs_])]

    #Prepare the subplot
    ax = fig.add_subplot(2, 1, plot)
    ax.title.set_text(label)
    
    #Draw the neurons
    ax.scatter(x_neurons, y_neurons, s=100, zorder=5)
    
    #Draw the connections between neurons with line widths corresponding to the weight of the connection
    for l,layer in enumerate(mlp.coefs_):
        for i,neuron in enumerate(layer):
            for j,w in enumerate(neuron):
                ax.plot([loc_neurons[l][i][0], loc_neurons[l+1][j][0]], [loc_neurons[l][i][1], loc_neurons[l+1][j][1]], 'white', linewidth=((w-weight_range[0])/(weight_range[1]-weight_range[0])*5+0.2)*1.2)
                ax.plot([loc_neurons[l][i][0], loc_neurons[l+1][j][0]], [loc_neurons[l][i][1], loc_neurons[l+1][j][1]], 'grey', linewidth=(w-weight_range[0])/(weight_range[1]-weight_range[0])*5+0.2)

#CREATE THE FIGURE FOR RESULTS VISUALIZATION
fig = plt.figure(figsize=(8,8))

#LOAD DATA TO BE LEARNED
data = pd.read_csv('heart.csv')

#SCALE NUMERIC DATA EXCLUDING THE TARGET VARIABLE
sc = StandardScaler()
num_d = data.select_dtypes(exclude=['object'])
num_d = num_d.iloc[:,:-1]
data[num_d.columns] = sc.fit_transform(num_d)

#CONVERT CATEGORICAL TEXTUAL DATA INTO NUMERICAL CATEGORIES
le = LabelEncoder()
objList = data.select_dtypes(include="object").columns
for feat in objList:
    data[feat] = le.fit_transform(data[feat].astype(str))

#GET STATISTICAL SUMMARY OF THE DATASET
print("\n================ STATISTICAL SUMMARY OF THE DATASET ================")
print(data.describe().transpose())

#SPLIT DATA INTO TRAINING AND TESTING SETS (80% training, 20% testing)
features = data.iloc[:,:-1]
target = data.iloc[:,-1:]
X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=0.8)

#EXAMINE THE TRAINING INPUT DATA
print("\n================ STATISTICAL SUMMARY OF TRAINING INPUT DATA ================")
print(X_train.describe().transpose())

#CONSTRUCT THE ANN MODEL WITH 3 HIDDEN LAYERS OF 4 NEURONS, IDENTITY ACTIVATION FUNCTION, & MAXIMUM ITERATION OF 1000
mlp = MLPClassifier(hidden_layer_sizes=(4,4,4), activation='identity', max_iter=1000)

#VISUALIZE THE NEURAL NETWORK BEFORE TRAINING
mlp.partial_fit(X_train, y_train, np.unique(target))
visualise(mlp, 1, "Neural Network Before Training")

#TRAIN THE MODEL USING THE 'FIT' METHOD OF THE CLASSIFIER
mlp.fit(X_train, y_train)

#USE THE FITTED (TRAINED) MODEL TO PREDICT THE OUTPUTS FOR THE TESTING DATA
predictions = mlp.predict(X_test)

#EVALUATION OF THE NEURAL NETWORK
#Generate and display the contingency matrix
print("\n================ CONTINGENCY MATRIX ================")
cm = confusion_matrix(y_test, predictions)
tp = cm[0][0]
fp = cm[0][1]
tn = cm[1][1]
fn = cm[1][0]
print(cm)

#Calculate the performance metrics based on the contingency matrix
print("\nTotal Number of Testing Data = " + str(tp+fp+tn+fn))
print("True Positives = " + str(tp))
print("False Positives = " + str(fp))
print("True Negatives = " + str(tn))
print("False Negatives = " + str(fn))
print("\nOverall Performance Statistics")
print("Accuracy = " + str((tp+tn)/(tp+tn+fp+fn)))
print("True Positive Ratio (Sensitivity) = " + str(tp/(tp+fn)))
print("False Positive Ratio (False Alarm Rate) = " + str(fp/(fp+tn)))
print("True Negative Ratio (Specificity) = " + str(tn/(fp+tn)))

#Generate and display the classification report
print("\n================ CLASSIFICATION REPORT ================")
print("Performance Statistics by Class")
print(classification_report(y_test, predictions))

#VISUALIZE THE NEURAL NETWORK AFTER TRAINING
visualise(mlp, 2, "Neural Network After Training")
