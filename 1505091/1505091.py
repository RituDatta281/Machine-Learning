from math import log
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.preprocessing import binarize
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer 

def binarize_2(data,num_cols,target):
    for i in range(len(num_cols)):  
    #     print(data[num_cols[i]])
        sorted_array = sorted(data[num_cols[i]].unique())
        mid_point = np.empty((0, len(sorted_array)))
        for k in range(len(sorted_array)-1):
            mid_point = np.append(mid_point,((sorted_array[k] + sorted_array[k+1])/2))

    #     print(mid_point)

        information_gain = np.empty((0, len(mid_point)))

        for j in mid_point:
            temp = binarize(data[num_cols[i]].values.reshape(1,-1),threshold = j).reshape(-1,1)
            information_gain = np.append(information_gain,InfoGain(data,temp,target))
        #     print(j, information_gain)

        correct_value = mid_point[np.argmax(information_gain)]
    #     print(correct_value, 'correct val')

        data[num_cols[i]]=binarize(data[num_cols[i]].values.reshape(1,-1),threshold = correct_value).reshape(-1,1)
#         print(data[num_cols[i]])

def binarize_(data, num_cols):
    for i in range(len(num_cols)):
        binarizer= Binarizer(data[num_cols[i]].mean())
        data[num_cols[i]] = binarizer.fit_transform(data[num_cols[i]].values.reshape(1,-1)).reshape(-1,1)

def Telco_preprocessing(data):
    data.drop('customerID',axis=1,inplace=True)
    data= pd.DataFrame(data)
    # print(data)
    data['TotalCharges'] = data["TotalCharges"].replace(" ",0)
    data['TotalCharges']= pd.to_numeric(data['TotalCharges'])
    data['TotalCharges'] = data["TotalCharges"].replace(0,data['TotalCharges'].mean())
    data['TotalCharges']
    target = data["Churn"]
#     print ("\nMissing values :  ", data.isnull().sum().values.sum())
#     print ("\nUnique values :  \n",data.nunique())

    cat = data.nunique()[data.nunique() < 6].keys().tolist()
    target_col = ["Churn"]
    cat = [x for x in cat if x not in target_col]
    num_cols   = [x for x in data.columns if x not in cat + target_col ]
    bin_cols = data.nunique()[data.nunique() == 2].keys().tolist()
    bin_cols = [x for x in bin_cols if x not in target_col]
    multi_cols = [x for x in cat if x not in bin_cols]

    #binary encoding
    le = preprocessing.LabelEncoder()
    for i in bin_cols:
        data[i] = le.fit_transform(data[i])
    #     print(data[i])

    #string -> encoding
    le = preprocessing.LabelEncoder()
    for i in multi_cols:
        data[i] = le.fit_transform(data[i])
    #     print(data[i])
    
#     binarize_2( data, num_cols,target )
    binarize_(data,num_cols)
    return data

def entropy(column):
    elements,counts = np.unique(column,return_counts = True)
    entropy = 0
    for i in range(elements.size):
        entropy += counts[i]/np.sum(counts)*log(counts[i]/np.sum(counts),2)
    entropy= -entropy
    return entropy

def InfoGain(data,attribute,target):
#     attribute= data.get(attribute_name)
    total_entropy = entropy(data.get(target))
    elements,counts = np.unique(attribute,return_counts = True)
    split_columns={}
    for i in range(elements.size):
        split_columns[i] = data[attribute == elements[i]]  
    weighted_Entropy = 0
    for i in range(elements.size):
        weighted_Entropy += (counts[i]/np.sum(counts))*entropy(split_columns[i].get(target))
    Info_gain = total_entropy - weighted_Entropy 
    return Info_gain

def PLURALITY_VALUE(example,target):
    elements,count = np.unique(example[target],return_counts=True)
    selected_index =  np.argmax(count)
#     print(np.argmax(count))
#     print(np.unique(example[target])[selected_index])
    return np.unique(example[target])[selected_index]

def decision_tree_learning(examples, attributes,target, parent_examples, depth):
    
# if examples is empty then return PLURALITY_VALUE(parent_examples)

    if depth == 0:    
        return PLURALITY_VALUE(examples,target)
    
    if examples.size==0 :
#         tree.value = PLURALITY_VALUE(parent_examples)
        return PLURALITY_VALUE(parent_examples,target)
    
# else if all examples have same classification then return the classification
    elif (np.unique(examples[target]).size) < 2:
        classification = np.unique(examples[target])[0]
        return classification
    
# else if attributes is empty then return PLURALITY_VALUE(examples)
    elif attributes.size == 0:
#         tree.value = PLURALITY_VALUE(examples)
        return PLURALITY_VALUE(examples,target)
    
    else:      
        InfoGain_values = np.empty((0, attributes.size))
        parent_examples = PLURALITY_VALUE(examples,target)
        for feature in attributes:
            InfoGain_values = np.append(InfoGain_values,InfoGain(examples,examples.get(feature), target))
        
        selected_column = examples.get(attributes[np.argmax(InfoGain_values)])
        best_attribute = attributes[np.argmax(InfoGain_values)]
        
        tree = {best_attribute:{}}
       
        featuresNew = np.empty((0, attributes.size-1))
        
        for i in range(attributes.size):
            if(attributes[i]!=best_attribute):
                featuresNew = np.append(featuresNew,attributes[i])
                
        elements,counts=np.unique(selected_column,return_counts=True)
        
        for i in range((np.unique(selected_column)).size):
            value = np.unique(selected_column)[i]
            tree[best_attribute][value]  = decision_tree_learning(examples.get(examples[best_attribute] == value),featuresNew,target,parent_examples, depth-1)
    return tree

def prediction_itr(data_row, features, decision_tree):
    for i in range(len(features)):
        if features[i] in list(decision_tree.keys()):
            try:
                decision_tree[features[i]][data_row[i]]
            except:
#                 print("Feature ", features[i], "value", data_row[i], "\n")
                return "Yes"
            if isinstance(decision_tree[features[i]][data_row[i]],dict):
                return prediction_itr(data_row,features,decision_tree[features[i]][data_row[i]])
            else:
                return decision_tree[features[i]][data_row[i]]
    

def predict(data, features, decision_tree, target):
    predicted_result = []

    for i in range(len(data)):
        row = data.iloc[i,:-1].values
        predicted_result.append(prediction_itr(row, features, decision_tree))
    return predicted_result
    

def train_test_split(data):
    training_data = data.iloc[:int(len(data)*.8)]
    testing_data = data.iloc[int(len(data)*.8):]
    return training_data,testing_data

def Normalize(w):
    return [float(i) / sum(w) for i in w]

def AdaBoost(examples, decision_tree_learning, K, attributes, target):
    
    w = [] #w, a vector of N example weights, initially 1/N
    h = [] #h, a vector of K hypotheses
    z = [] #z, a vector of K hypothesis weights
    
    for i in range(len(examples)):
        w.append(1/len(examples))
        
    actual = examples[target].values.reshape(-1,1)
    for k in range(K):
#         print("vector\n",w)
        temp = examples.copy()
        dataSampled = temp.sample(frac = 1, weights = w, replace=True )
        tree={}
        h.append(decision_tree_learning(dataSampled,attributes,target, None , 1))
        predicted = predict(examples,attributes, h[k],target)
        error = 0.0
        for j in range(len(examples)):
            if actual[j]!= predicted[j]:
                error = error + w[j]
#         print("error ", error)    
        if error > 0.5 :
            continue
            
        for j in range(len(examples)):
            if actual[j] == predicted[j] :
                w[j] = w[j] * error/(1-error)
                
        w = Normalize(w)
#         print(w)
        z.append(log((1-error)/error , 2))
#         print(z)
        
    return h,z   
    

def WeightedMajority(data,h,z,k, features, target):
    pred = []
    actual_result = data.get(target).values
    for i in range(len(actual_result)):
        if actual_result[i] == "Yes":
            actual_result[i] = 1
        else :
            actual_result[i] = -1
    
    
    for i in range(k):
        predictedValue = predict( data,features,h[i],target)
        for j in range(len(predictedValue)):
            if predictedValue[j] == "Yes":
                predictedValue[j] = z[i] 
            else:
                predictedValue[j] = -z[i]
            
        pred.append(predictedValue)
        
    final_pred = []
    count = 0.0
    
    for i in range(len(data)):
        value = 0 
        for j in range(k):
            value += pred[j][i]
        if value >= 0:
            final_pred.append(1)
        else:
            final_pred.append(-1)
        
        if final_pred[i] == actual_result[i]:
            count+=1
            
    accuracy = (count/len(data))*100
    print ("Accuracy using Adaboost: ",accuracy,'%\n Where k =', k)
        
    

def decisionTreeResult(training_data,features,target,testing_data):
    DT = decision_tree_learning(training_data,features,target, None, np.inf )  

    Actual_result = testing_data[target].values.reshape(-1,1)
    predicted_result = predict(testing_data,features,DT,target)
#     print(len(Actual_result),len(predicted_result))
    correct_prediction=0.0
    false_positive = 0.0
    false_negative = 0.0
    true_positive = 0.0
    true_negative = 0.0
    
    
    for i in range(len(testing_data)):
        if Actual_result[i] == predicted_result[i]:
            correct_prediction +=1
            if Actual_result[i] == "Yes":
                true_positive += 1
            else:
                true_negative += 1
        else:
            
            if predicted_result[i] == "Yes":
                false_positive += 1
            else:
                false_negative += 1
    
    precision = true_positive/(true_positive+false_positive)
    
    recall = true_positive/(true_positive+ false_negative)

    print("Accuracy (Using decision tree) : ", (correct_prediction/float(len(testing_data))) * 100, "%\n" )
    
    print("True positive (Using decision tree) : ", (true_positive/(true_positive+false_negative)) * 100, "%\n" )
    
    print("True negative (Using decision tree) : ", (true_negative/(true_negative+false_positive)) * 100, "%\n" )
    
    print("Positive predictive value :" , (true_positive/(false_positive+true_positive)) * 100, "%\n" )
    
    print("False Discovery rate :" , (1-(true_positive/(false_positive+true_positive))) * 100, "%\n" )
    
    print("F1 rate: ", 2.0*(precision*recall)/(precision+recall)*100, "%\n")    

def ADABOOST(training_data,features,target,k, testing_data):
    DT = decision_tree_learning(training_data,features,target, None,1 )  
    h,z= AdaBoost(training_data,decision_tree_learning,k,features,target)
    WeightedMajority(testing_data,h,z,k, features, target)

def option_(option,k):
    
    if option == 1:
        data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
        data = Telco_preprocessing(data)
        features= data.columns[:-1]
        target = data.columns[features.size]
        training_data,testing_data = train_test_split(data) 

        #---------------- DECISION TREE PREDICTION----------------
        decisionTreeResult(training_data.copy(),features,target,testing_data.copy())

        # ---------------------ADABOOST-----------------
        ADABOOST(training_data.copy(),features,target,k,testing_data.copy())
        
    elif option == 2:
        features= ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','salary']
        data = pd.read_csv('adult_data.csv', delimiter=",",names=features)
        Testdata = pd.read_csv('adult_test.csv', delimiter=",",names=features)
#         print ("\nUnique values :  \n",data.nunique())
        target = features.pop()
        data[target]= data[target].replace(' <=50K', 'No')
        data[target]= data[target].replace(' >50K', 'Yes')
        # --------------------- DELETING MISSING VALUES ----------------------------------------
        for feature in features:
            data[feature] = data[feature].replace(' ?', np.nan)
        
        data = data.dropna(axis = 0).reset_index(drop=True)
        
        num_cols = ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
        binarize_(data, num_cols)
#         binarize_2( data, num_cols, target )

        Testdata[target]= Testdata[target].replace(' <=50K.', 'No')
        Testdata[target]= Testdata[target].replace(' >50K.', 'Yes')

        for feature in features:
            Testdata[feature] = Testdata[feature].replace(' ?', np.nan)
        
        Testdata = Testdata.dropna(axis = 0).reset_index(drop=True)
#         print(Testdata)
        binarize_( Testdata, num_cols )
        
        training_data = data.copy()
        testing_data = Testdata.copy()
        features = data.columns[:-1]
        target = data.columns[features.size]
        
         #---------------- DECISION TREE PREDICTION----------------
        decisionTreeResult(training_data,features,target,testing_data)
        
         # ---------------------ADABOOST-----------------
        ADABOOST(training_data,features,target,k,testing_data)
        
    else:
        data3 = pd.read_csv("creditcard.csv")
        features3 = data3.columns[:-1]
        target3 = data3.columns[features3.size]
        data3[target3]= data3[target3].replace(0, 'No')
        data3[target3]= data3[target3].replace(1, 'Yes')
        
        num_cols3 = data3.nunique()[data3.nunique() > 2].keys().tolist()
#         print(features3)
        binarize_(data3,num_cols3)
#         print(data3)
        data3 = data3.sample(n=20000)
        training_data3,testing_data3 = train_test_split(data3)
        
#          #---------------- DECISION TREE PREDICTION----------------
        decisionTreeResult(training_data3.copy(),features3,target3,testing_data3.copy())
        
        # ---------------------ADABOOST-----------------
        ADABOOST(training_data3,features3,target3,k,testing_data3)
        

k = 5
option_(2,k)


