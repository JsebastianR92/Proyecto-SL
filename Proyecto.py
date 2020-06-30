import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as mt
from sklearn.externals import joblib
import tensorflow as tf


Data_Titanic = pd.read_csv("data_titanic_proyecto.csv")
Data_Titanic.head(3)

Data_Titanic.describe()

Data_Titanic["Embarked"].value_counts()

Data_Titanic["passenger_class"].value_counts()


Data_Titanic["Age"] = Data_Titanic["Age"].fillna(Data_Titanic["Age"].median());
Data_Titanic["Age"].isnull().sum()

Data_Titanic["Female"] = (Data_Titanic["passenger_sex"] == 'F').astype(np.float)

Data_Titanic["Embarked"] = Data_Titanic["Embarked"].fillna('X')
Data_Titanic["Embarked_S"] = (Data_Titanic["Embarked"] == 'S').astype(np.float32)
Data_Titanic["Embarked_C"] = (Data_Titanic["Embarked"] == 'C').astype(np.float32)
Data_Titanic["Embarked_Q"] = (Data_Titanic["Embarked"] == 'Q').astype(np.float32)

Data_Titanic["Class_Lower"] = (Data_Titanic["passenger_class"] == 'Lower').astype(np.float32)
Data_Titanic["Class_Middle"] = (Data_Titanic["passenger_class"] == 'Middle').astype(np.float32)
Data_Titanic["Class_Upper"] = (Data_Titanic["passenger_class"] == 'Upper').astype(np.float32)

Data_Titanic["Survived"] = (Data_Titanic["passenger_survived"].values == 'Y').astype(np.float32)
Data_Titanic.head(3)




Campos = ["Age", "SibSp", "Parch", "Fare", "Female", 
             "Embarked_S", "Embarked_C", "Embarked_Q", 
             "Class_Lower", "Class_Middle", "Class_Upper"]

y = Data_Titanic["Survived"].values
X = Data_Titanic[Campos].values

X.shape, y.shape



Data_Titanic[Campos].head(20)





Campos_Data = ["Age", "SibSp", "Parch", "Fare", "Female", "Embarked_S", "Embarked_C", "Embarked_Q", "Class_Lower", "Class_Middle", "Class_Upper", "Survived"]

Data_Titanic[Campos_Data].to_csv('TitanicProcessed.csv')




from sklearn.model_selection import train_test_split



X_train_cv, X_test, y_train_cv, y_test = train_test_split(X, y, test_size = 0.4, shuffle = True,random_state = 314)
X_train, X_val, y_train, y_val = train_test_split(X_train_cv, y_train_cv, test_size = 0.2, random_state = 42)
X_train.shape, X_val.shape, X_test.shape




from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


X_train.shape




SVM_M = Pipeline([("scaler", StandardScaler()),("svc", SVC(C = 10., kernel='rbf', gamma = 0.01, tol = 0.001, max_iter = 5000))])
SVM_M.fit(X_train, y_train.reshape(len(y_train), ));
SVM_M.score(X_train, y_train), SVM_M.score(X_val, y_val)



y_pred = SVM_M.predict(X_val)
(mt.accuracy_score(y_val, y_pred, normalize=True), mt.f1_score(y_val, y_pred), mt.precision_score(y_val, y_pred), mt.recall_score(y_val, y_pred, average='weighted'))



def entrenaMet_Lis_New(X, y, C_param): 
    SVM_M = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(C = C_param, kernel='rbf', tol = 0.001, max_iter = 5000))])
    SVM_M.fit(X, y.reshape(len(y), ))
    return SVM_M

def Metricas_Acq(modelo, X_train, y_train, X_val, y_val):
    y_pred = modelo.predict(X_train)
    Metrica_Ent = [mt.accuracy_score(y_train, y_pred, normalize=True), 
                    mt.f1_score(y_train, y_pred), 
                    mt.precision_score(y_train, y_pred), 
                    mt.recall_score(y_train, y_pred, average='weighted')]
    y_pred = modelo.predict(X_val)
    Metricas = [mt.accuracy_score(y_val, y_pred, normalize=True), 
                    mt.f1_score(y_val, y_pred), 
                    mt.precision_score(y_val, y_pred), 
                    mt.recall_score(y_val, y_pred, average='weighted')]
    
    return Metrica_Ent, Metricas





ParametroC = [0.1, 0.5, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 15., 20., 25.]
Met_Lis_New = []
Met_List = []
for c in ParametroC:
    svm = entrenaMet_Lis_New(X_train, y_train, C_param = c)
    Metrica_Ent, valMetrics = Metricas_Acq(svm, X_train, y_train, X_val, y_val)
    Met_Lis_New.append(Metrica_Ent.copy())
    Met_List.append(valMetrics.copy())




SVM_Res = pd.DataFrame(np.column_stack((ParametroC, np.array(Met_Lis_New), np.array(Met_List))), 
                             columns = ['C', 'AccuracyTrain', 'F1Train', 'PrecisionTrain', 'RecallTrain', 
                                       'AccuracyVal', 'F1Val', 'PrecisionVal', 'RecallVal'])
SVM_Res






fig = plt.figure(figsize=(14, 8), dpi= 80, facecolor='w', edgecolor='k')
plt.plot(SVM_Res.C, SVM_Res.AccuracyTrain, 'b-', 
         SVM_Res.C, SVM_Res.AccuracyVal, 'r-');
plt.legend(("Train Accuracy", "Validation Accuracy"));






SVM_final = entrenaMet_Lis_New(X_train, y_train, C_param = 9)






from sklearn.model_selection import GridSearchCV

parameters = {'svc__C' : [0.1, 0.5, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 15., 20., 25.], 
              'svc__gamma' : [0.1, 0.5, 1., 2., 3., 5., 7.5, 10., 15., 20., 25.]}

svc = SVC(gamma="scale")
SVM_M = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel = 'rbf', tol = 0.001, max_iter = 5000))])

new_pamcf = GridSearchCV(SVM_M, parameters, cv=5, iid=False, verbose=True, n_jobs=4)
new_pamcf.fit(X_train_cv, y_train_cv) 



new_pamcf.best_params_




SVM_M = Pipeline([("scaler", StandardScaler()),("svc", SVC(C = 3.0, gamma = 0.1, kernel='rbf', tol = 0.001, max_iter = 5000))])
SVM_M.fit(X_train_cv, y_train_cv.reshape(len(y_train_cv), ))
Metricas_Acq(SVM_M, X_train, y_train, X_val, y_val)



import tensorflow as tf




def entrenar_reg_logistica(Xtrain, Ytrain, lr, lambda_val, epochs):
    import time
    m, k = Xtrain.shape
    
    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():
        X = tf.placeholder(tf.float32, shape = (None, k), name = "X")
        Ylabels = tf.placeholder(tf.float32, name = "Ylabels")
        lr_param = tf.placeholder(tf.float32, name = "lr")
        lambda_param = tf.placeholder(tf.float32, name = "lambda")
        
        W = tf.Variable(tf.truncated_normal(shape = [k, 1]), name = "W")
        b = tf.Variable(tf.truncated_normal(shape = (1, 1)), name = "b")
        
        with tf.name_scope("Logits"):
            Logits = tf.add(tf.matmul(X, W), b, name = "Logits")
            YlabelsHat = tf.nn.sigmoid(Logits)

        with tf.name_scope("FuncionCosto"):
            w_norm = tf.divide(tf.multiply(tf.multiply(tf.constant(0.5), lambda_param), 
                                 tf.reduce_sum(tf.square(W))), tf.cast(m, tf.float32), name = "W_norm")

            classif_term = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels = Ylabels, logits = Logits), name = "CostoClasif") 

            cost = tf.add(classif_term, w_norm, name="Costo")

        with tf.name_scope("GradientDes.Optimizer"):
            optimizer = tf.train.GradientDescentOptimizer(lr_param).minimize(cost) 
 
        init = tf.global_variables_initializer() 

    start = time.time()
    with tf.Session(graph = g) as sess: 

        sess.run(init)
        for epoch in range(epochs):
            _, c_ = sess.run([optimizer, cost], 
                             feed_dict = {X : Xtrain, Ylabels : Ytrain.reshape((m, 1)), 
                                          lr_param : lr, lambda_param : lambda_val})
            if (epoch + 1) % round(epochs*0.1) == 0:
                print("Epoch: %d, \t costo = %0.4f" % (epoch+1, c_))
        w_, b_ = sess.run([W, b])
        
    end = time.time()
    print("Tiempo transcurrido: %0.2f segundos" % (end-start))
    return w_, b_






w_, b_ = entrenar_reg_logistica(X_train, y_train, lr = 0.001, epochs = 1000, lambda_val = 8.)





def Logistic_Pred(x, weights, b):
    def sigmoid(x):
        return (1 / (1 + np.exp(-x)))
    l = np.matmul(x, weights) + b
    y_hat = 1.0*(sigmoid(l) > 0.5)
    return y_hat





(mt.accuracy_score(y_train, Logistic_Pred(X_train, w_, b_)), 
mt.accuracy_score(y_val, Logistic_Pred(X_val, w_, b_)) )






def Metric_Logit(w, b, X_train, y_train, X_val, y_val):
    y_pred = Logistic_Pred(X_train, w, b)
    Metrica_Ent = [mt.accuracy_score(y_train, y_pred, normalize=True), 
                    mt.f1_score(y_train, y_pred), 
                    mt.precision_score(y_train, y_pred), 
                    mt.recall_score(y_train, y_pred, average='weighted')]

    y_pred = Logistic_Pred(X_val, w, b)
    Metricas = [mt.accuracy_score(y_val, y_pred, normalize=True), 
                    mt.f1_score(y_val, y_pred), 
                    mt.precision_score(y_val, y_pred), 
                    mt.recall_score(y_val, y_pred, average='weighted')]
    
    return Metrica_Ent, Metricas





Data_Titanic[Campos].head(2)





varFilter = [0,3,4,5,6,7,8,9,10]

lambda_param_list = [0.1, 0.5, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 15., 20., 25.]
Met_Lis_New = []
Met_List = []
for lam in lambda_param_list:
    w_, b_ = entrenar_reg_logistica(X_train[:, varFilter], y_train, 
                                    lr = 0.001, epochs = 1000, 
                                    lambda_val = lam)
    Metrica_Ent, valMetrics = Metric_Logit(w_, b_, 
                                             X_train[:, varFilter], y_train, 
                                             X_val[:, varFilter], y_val)
    Met_Lis_New.append(Metrica_Ent.copy())
    Met_List.append(valMetrics.copy())




Log_Reg = pd.DataFrame(np.column_stack((lambda_param_list, np.array(Met_Lis_New), np.array(Met_List))), 
                             columns = ['Lambda', 'AccuracyTrain', 'F1Train', 'PrecisionTrain', 'RecallTrain', 
                                       'AccuracyVal', 'F1Val', 'PrecisionVal', 'RecallVal'])
Log_Reg





fig = plt.figure(figsize=(14, 8), dpi= 80, facecolor='w', edgecolor='k')
plt.plot(Log_Reg.Lambda, Log_Reg.AccuracyTrain, 'b-', 
         Log_Reg.Lambda, Log_Reg.AccuracyVal, 'r-');
plt.legend(("Train Accuracy", "Validation Accuracy"));






varFilter = [0,3,4,8,9,10]

lambda_param_list = [0.1, 0.5, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 15., 20., 25.]
Met_Lis_New = []
Met_List = []
for lam in lambda_param_list:
    w_, b_ = entrenar_reg_logistica(X_train[:, varFilter], y_train, 
                                    lr = 0.001, epochs = 1000, 
                                    lambda_val = lam)

    Metrica_Ent, valMetrics = Metric_Logit(w_, b_, 
                                             X_train[:, varFilter], y_train, 
                                             X_val[:, varFilter], y_val)
    Met_Lis_New.append(Metrica_Ent.copy())
    Met_List.append(valMetrics.copy())





Log_Reg = pd.DataFrame(np.column_stack((lambda_param_list, np.array(Met_Lis_New), np.array(Met_List))), 
                             columns = ['Lambda', 'AccuracyTrain', 'F1Train', 'PrecisionTrain', 'RecallTrain', 
                                       'AccuracyVal', 'F1Val', 'PrecisionVal', 'RecallVal'])
Log_Reg



fig = plt.figure(figsize=(14, 8), dpi= 80, facecolor='w', edgecolor='k')
plt.plot(Log_Reg.Lambda, Log_Reg.AccuracyTrain, 'b-', 
         Log_Reg.Lambda, Log_Reg.AccuracyVal, 'r-');
plt.legend(("Train Accuracy", "Validation Accuracy"));




varFilter = [0,3,4,8,9,10]

w_, b_ = entrenar_reg_logistica(X_train[:, varFilter], y_train, 
                                lr = 0.001, epochs = 3000, lambda_val = 6)
(mt.accuracy_score(y_train, Logistic_Pred(X_train[:, varFilter], w_, b_)), 
mt.accuracy_score(y_val, Logistic_Pred(X_val[:, varFilter], w_, b_)) )




from sklearn.naive_bayes import GaussianNB

def Naive_Bayes(X, y, varFilter):
    model = GaussianNB()
    model.fit(X[:, varFilter], y.reshape(len(y),))
    return model



naive1 = Naive_Bayes(X_train, y_train, [0,1,2,3,4,5,6,7,8,9,10])
Metricas_Acq(naive1, X_train, y_train, X_val, y_val)





combinaciones = [[0,1,2,3,4,5,6,7,8,9,10], [0,3,4,8,9,10], [0,3,4,8,9,10], [0,1,2,3,4,5,6,7], [0,3,4,5,6,7,8,9,10], [0,1,3,5,6,7], [0,3,8,9,10], [0,4,5,6,7,8,9,10]]
Met_Lis_New = []
Met_List = []

for combFilter in combinaciones: 
    mNB = Naive_Bayes(X_train, y_train, combFilter)
    Metrica_Ent, valMetrics = Metricas_Acq(mNB, X_train[:, combFilter], y_train, 
                                          X_val[:, combFilter], y_val)
    Met_Lis_New.append(Metrica_Ent.copy())
    Met_List.append(valMetrics.copy())





Naive_Bayes_Result = pd.DataFrame(np.column_stack((combinaciones, np.array(Met_Lis_New), np.array(Met_List))), 
                             columns = ['Variables', 'AccuracyTrain', 'F1Train', 'PrecisionTrain', 'RecallTrain', 
                                       'AccuracyVal', 'F1Val', 'PrecisionVal', 'RecallVal'])
Naive_Bayes_Result




fig = plt.figure(figsize=(14, 8), dpi= 80, facecolor='w', edgecolor='k')
combinaciones_str = [str(comb) for comb in combinaciones]
plt.plot(combinaciones_str, Naive_Bayes_Result.AccuracyTrain, 'b*',
         combinaciones_str, Naive_Bayes_Result.AccuracyVal, 'r+');
plt.legend(("Train Accuracy", "Validation Accuracy"));





Filtro_Tit = [0,4,5,6,7,8,9,10]
mNB_final = Naive_Bayes(X_train_cv, y_train_cv, Filtro_Tit)

(mt.accuracy_score(y_train, mNB_final.predict(X_train[:, Filtro_Tit])), 
mt.accuracy_score(y_val, mNB_final.predict(X_val[:, Filtro_Tit])), 
mt.accuracy_score(y_train_cv, mNB_final.predict(X_train_cv[:, Filtro_Tit])))








def Features_prob(pdDataFrame, featureName, x, qsize):
    f, bins_f = pd.qcut(pdDataFrame[featureName], qsize, retbins=True, duplicates='drop')
    rango = pd.cut(x, bins = bins_f)

    if rango.isnull().any():
        return 0, bins_f
    return (f.value_counts()[rango] / len(f)).values, bins_f

def Lab_Prob(pdDataFrame, featureName, x, className, label, bins_f):
    # Obtener las categorías de pdFeature y los bins 
    f = pd.cut(pdDataFrame.loc[pdDataFrame[className] == label , featureName], 
               bins = bins_f, include_lowest=True)
    rango = pd.cut(x, bins = bins_f, include_lowest=True)
    
    if rango.isnull().any():
        return np.zeros(len(x))
    
    return (f.value_counts()[rango] / len(f)).values

def Lb_Probab(pdDataFrame, className, label):
    p = pdDataFrame.loc[pdDataFrame[className] == label, className].count() / len(pdDataFrame)
    return p

def Lab_Prob_seg(pdDataFrame, featureName, x, className, label):
    pClass = pdDataFrame.loc[(pdDataFrame[className] == label), featureName].count()
    pInt = pdDataFrame.loc[(pdDataFrame[className] == label)&(pdDataFrame[featureName] == 1), featureName].count()
    
    pLabel = pInt/pClass
    pNotLabel = 1. - pLabel
    prob = np.array([pNotLabel, pLabel])
    return prob[x.astype(int)]

def Pred_Tit(Titanic, TitanicEval):
    age = TitanicEval.Age.values
    fare = TitanicEval.Fare.values
    _, binsAge = Features_prob(Titanic, "Age", age, 5)
    _, binsFare = Features_prob(Titanic, "Fare", fare, 5)
    
    BinaryFields = ["Female"]
    
    survived_list = [0,1]
    Surv_p1 = []
    for survived in survived_list:
        Ft1 = Lab_Prob(Titanic, "Age", age, "Survived", survived, binsAge)
        Ft2 = Lab_Prob(Titanic, "Fare", fare, "Survived", survived, binsFare)
        ones_ft = np.ones(len(TitanicEval.values))
        for binaryFeature in BinaryFields:
            ones_ft = Lab_Prob_seg(Titanic, binaryFeature, 
                                                TitanicEval[binaryFeature].values, 
                                                "Survived", survived)
        Surv = Lb_Probab(Titanic, "Survived", survived)
        Surv_p1.append( (Ft1 * Ft2 * ones_ft) * Surv )

    Surv_p1 = np.array(Surv_p1)
    Surv_p1_yes = Surv_p1[1,:] / np.sum(Surv_p1, axis=0)
    return np.array([p > 0.5 for p in Surv_p1_yes]).astype(np.float)







Field_Train = ["Age", "Fare", "Female", "Embarked_S", "Embarked_C", "Embarked_Q", 
             "Class_Lower", "Class_Middle", "Class_Upper", "Survived"]
NB_freq_varFilter = [0,3,4,5,6,7,8,9,10]
Titanic_train = pd.DataFrame(np.column_stack(
    (X_train[:, NB_freq_varFilter], y_train)), columns=Field_Train)







yhat_NBmanual_train = Pred_Tit(Titanic_train, Titanic_train)
yhat_NBmanual_train

mt.accuracy_score(y_train, yhat_NBmanual_train)






Field_Ev = ["Age", "Fare", "Female", "Embarked_S", "Embarked_C", "Embarked_Q", 
             "Class_Lower", "Class_Middle", "Class_Upper"]
 
Titanic_val = pd.DataFrame(X_val[:, NB_freq_varFilter], columns=Field_Ev)




# Obtener los valores dados por el NB frecuentista
yhat_NBmanual_val = Pred_Tit(Titanic_train, Titanic_val)
yhat_NBmanual_val

mt.accuracy_score(y_val, yhat_NBmanual_val)



from sklearn.tree import DecisionTreeClassifier
from sklearn import tree




def Arbol_Ent(X, y, max_depth): 
    tree_model = tree.DecisionTreeClassifier(max_depth = max_depth)
    tree_model.fit(X, y)
    return tree_model

Arbol_Ent(X_train, y_train, 2)






Met_Lis_New = []
Met_List = []
depth_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
for depth in depth_list: 
    tree_model = Arbol_Ent(X_train, y_train, depth)
    Metrica_Ent, valMetrics = Metricas_Acq(tree_model, X_train, y_train, X_val, y_val)
    Met_Lis_New.append(Metrica_Ent.copy())
    Met_List.append(valMetrics.copy())



Result_Arbol = pd.DataFrame(np.column_stack((depth_list, np.array(Met_Lis_New), np.array(Met_List))), 
                             columns = ['Profundidad', 'AccuracyTrain', 'F1Train', 'PrecisionTrain', 'RecallTrain', 
                                       'AccuracyVal', 'F1Val', 'PrecisionVal', 'RecallVal'])
Result_Arbol



fig = plt.figure(figsize=(14, 8), dpi= 80, facecolor='w', edgecolor='k')
plt.plot(Result_Arbol.Profundidad, Result_Arbol.AccuracyTrain, 'b-',
         Result_Arbol.Profundidad, Result_Arbol.AccuracyVal, 'r-');
plt.legend(("Train Accuracy", "Validation Accuracy"));





Arbol_Fin = Arbol_Ent(X_train_cv, y_train_cv, max_depth = 7)

(mt.accuracy_score(y_train, Arbol_Fin.predict(X_train)), 
mt.accuracy_score(y_val, Arbol_Fin.predict(X_val)), 
mt.accuracy_score(y_train_cv, Arbol_Fin.predict(X_train_cv)))




import graphviz
dot_data = tree.export_graphviz(Arbol_Fin, out_file=None, feature_names=Data_Titanic[Campos].columns.values, class_names=Data_Titanic["passenger_survived"].value_counts().index.values,filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.format = 'png'
graph.render("Imagen/Arbol")
































































































































































