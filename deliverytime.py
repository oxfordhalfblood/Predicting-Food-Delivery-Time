import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from fancyimpute import IterativeImputer
from sklearn import metrics
# from fancyimpute import MICE
# from fancyimpute import KNN

# training_set = pd.read_excel("mainfiles/Data_Train.xlsx")
# test_set = pd.read_excel("mainfiles/Data_Test.xlsx")

# =========  Preprocess Training Set ============

# training_set['Cuisines'] = training_set['Cuisines'].str.split(',', expand= False)
# mlb = MultiLabelBinarizer()

# training_set = training_set.join(pd.DataFrame(mlb.fit_transform(training_set.pop('Cuisines')),
#                           columns=mlb.classes_,
#                           index=training_set.index))

# training_set['Restaurant'] = training_set['Restaurant'].str.replace(r'^ID_', '')
# training_set['Delivery_Time'] = training_set['Delivery_Time'].str.replace(r' minutes$', '')
# training_set['Average_Cost'] = training_set['Average_Cost'].str.replace(r'^₹', '')
# training_set['Minimum_Order'] = training_set['Minimum_Order'].str.replace(r'^₹', '')

# training_set["Average_Cost"].replace('for', np.nan, inplace=True)
# training_set["Rating"].replace(['NEW','-','Opening Soon','Temporarily Closed'], np.nan, inplace=True)
# training_set["Votes"].replace(['-'], np.nan, inplace=True)
# training_set["Reviews"].replace(['-'], np.nan, inplace=True)

# training_set["Average_Cost"] = training_set["Average_Cost"].str.replace(",","").astype(float)
# le = preprocessing.LabelEncoder()
# training_set['Location'] = le.fit_transform(training_set['Location'])
# training_set = training_set.apply(pd.to_numeric)

# mice_imputer = IterativeImputer()
# training_set= pd.DataFrame(mice_imputer.fit_transform(training_set),columns=training_set.columns, index=training_set.index)

# print("\nContains NaN/Empty cells : ", training_set.isnull().any().any())
# print(training_set.head(14))

# training_set.to_csv("train.csv",index=False)

# =========  Preprocess Test Set ============

# test_set['Cuisines'] = test_set['Cuisines'].str.split(',', expand= False)
# mlb = MultiLabelBinarizer()

# test_set = test_set.join(pd.DataFrame(mlb.fit_transform(test_set.pop('Cuisines')),
#                           columns=mlb.classes_,
#                           index=test_set.index))

# test_set['Restaurant'] = test_set['Restaurant'].str.replace(r'^ID_', '')
# # test_set['Delivery_Time'] = test_set['Delivery_Time'].str.replace(r' minutes$', '')
# test_set['Average_Cost'] = test_set['Average_Cost'].str.replace(r'^₹', '')
# test_set['Minimum_Order'] = test_set['Minimum_Order'].str.replace(r'^₹', '')

# # test_set["Average_Cost"].replace('for', np.nan, inplace=True)
# test_set["Rating"].replace(['NEW','-','Opening Soon','Temporarily Closed'], np.nan, inplace=True)
# test_set["Votes"].replace(['-'], np.nan, inplace=True)
# test_set["Reviews"].replace(['-'], np.nan, inplace=True)

# test_set["Average_Cost"] = test_set["Average_Cost"].str.replace(",","").astype(float)
# le = preprocessing.LabelEncoder()
# test_set['Location'] = le.fit_transform(test_set['Location'])
# test_set = test_set.apply(pd.to_numeric)

# mice_imputer = IterativeImputer()
# test_set= pd.DataFrame(mice_imputer.fit_transform(test_set),columns=test_set.columns, index=test_set.index)

# print("\nContains NaN/Empty cells : ", test_set.isnull().any().any())
# print(test_set.head(14))

# test_set.to_csv("test.csv",index=False)

# ===== Train-Test col discrepancy solve ======

# training_set = pd.read_csv("train.csv") 
# test_set = pd.read_csv("test.csv") 

# Get missing columns in the training test
# missing_cols = set( training_set.columns ) - set( test_set.columns )
# # Add a missing column in test set with default value equal to 0
# for c in missing_cols:
#     test_set[c] = 0
# # Ensure the order of column in the test set is in the same order than in train set
# test_set = test_set[training_set.columns]

# total_cols=len(training_set.axes[1])
# print(total_cols)

#test_set.to_csv("test_new.csv",index=False)   #Save the New Test set with fixed col no.

# ==================================== Extra

# # X_df = pd.DataFrame(training_set[training_set['Rating']== 4])   # Just training_set['Rating']== 4 gives true/false
# print(test_set.head())


# ====================================

test_set = pd.read_csv("test_new.csv") 
test_set.drop(['Delivery_Time'], axis=1, inplace=True)
# Independent Variables
# print(training_set['Delivery_Time'])

# cols_at_end = ['Delivery_Time']    # push this col at the end of train

# training_set = training_set[[c for c in training_set if c not in cols_at_end] 
#         + [c for c in cols_at_end if c in training_set]]

# print(list(training_set.columns))    
#training_set.to_csv("train_new.csv",index=False)   #Save the New Test set with fixed col no.

# ========== On Validation Set ===========

training_set = pd.read_csv("train_new.csv") 
valid_train, valid_test = train_test_split(training_set, test_size=0.2)
# print(len(valid_train))   #8875
# print(len(valid_test))    #2219

valid_trainX = valid_train.iloc[:,0 : -1]
valid_trainY = valid_train.iloc[:, -1]  
valid_testX = valid_test.iloc[:,0:-1]   
valid_testY = valid_test.iloc[:, -1]
valid_testY.to_csv("valid_testY.csv",index=False) 

# ========== Validation Set Testing =======

# clf = SVC(kernel='rbf', C=1, gamma='auto')
clf = SVC(gamma='auto')
clf.fit(valid_trainX,valid_trainY)

pred_clf = clf.predict(valid_testX)
pred_clf = pd.DataFrame(pred_clf, columns = ['Delivery_Time']) # Converting to dataframe
# print(pred_clf)

pred_clf.to_excel("predic_valid.xlsx", index = False ) # Saving the output in to an excel

print("Accuracy:",metrics.accuracy_score(valid_testY, pred_clf))

#/*
# ========== On MAIN Test Set ===========

# X_train = training_set.iloc[:,0 : -1]   # all rows except last col

# # Dependent Variables
# Y_train = training_set.iloc[:, -1]  # all rows in last col
# # print(Y_train)

# # Independent Variables for Test Set
X_test = test_set.iloc[:,:]      # all rows and cols
# # print(X_test)

# # ============= MAIN Classify ==========

# clf = SVC(kernel='rbf', C=1, gamma='auto')
# clf.fit(X_train,Y_train)

pred_clf = clf.predict(X_test)
pred_clf = pd.DataFrame(pred_clf, columns = ['Delivery_Time']) # Converting to dataframe
# print(pred_clf)
pred_clf = pred_clf.astype(int)
# pd.options.display.float_format = '{:,.0f}'.format
# pred_clf = pred_clf.astype(str)

pred_clf['Delivery_Time'] = pred_clf['Delivery_Time'].astype(str) + ' minutes'
pred_clf.to_excel("predic2.xlsx", index = False ) # Saving the output in to an excel
# */
# ===================================

# print("\nEDA on Training Set\n")
# print("#"*30)
# print("\nFeatures/Columns : \n", training_set.columns)
# print("\n\nNumber of Features/Columns : ", len(training_set.columns))
# print("\nNumber of Rows : ",len(training_set))
# print("\n\nData Types :\n", training_set.dtypes)
# print("\nContains NaN/Empty cells : ", training_set.isnul.values.any())
# print("\nTotal empty cells by column :\n", training_set.isnull().sum(), "\n\n")

# training_set.fillna(0, inplace = True)

# print(len(training_set['Restaurant'].unique().tolist()))
# print(len(training_set['Location'].unique().tolist()))
# training_set['Cuisines'].unique()
# training_set['Delivery_Time'].unique()  # strip minutes from delivery time


# #convert the categorical columns into numeric

# print(training_set['Delivery_Time'].tail())
# print(training_set.dtypes)
# print(training_set.mean())
# print("\nContains NaN/Empty cells : ", training_set.isnull().any().any())
# training_set.fillna(training_set.mean(), inplace=True)

# training_set = pd.DataFrame(data=MICE.complete(training_set), columns=training_set.columns, index=training_set.index)


# imp = IterativeImputer(max_iter=10, random_state=0)
# df is my data frame with the missings. I keep only floats
# df_numeric = df.select_dtypes(include=[np.float]).as_matrix()

# I now run fancyimpute KNN, 
# it returns a np.array which I store as a pandas dataframe
# df_filled = pd.DataFrame(KNN(3).complete(df_numeric))

# imp.fit(training_set)



## -- use labelencoder ---

# create the Labelencoder object

# le = preprocessing.LabelEncoder()
# #convert the categorical columns into numeric
# training_set['Restaurant'] = le.fit_transform(training_set['Restaurant'])
# training_set['Location'] = le.fit_transform(training_set['Location'])
# training_set['Cuisines'] = le.fit_transform(training_set['Cuisines'])




###### ----- classifier  ----

# gbr=GradientBoostingRegressor()   # Leaderboard SCORE :  0.8364249755816828 @ RS =126 ,n_estimators=350, max_depth=6


# gbr = SVC(gamma='auto')
# gbr.fit(X_train,Y_train)

# y_pred_gbr = gbr.predict(X_test)
# y_pred_gbr = pd.DataFrame(y_pred_gbr, columns = ['Delivery Time']) # Converting to dataframe
# print(y_pred_gbr)

# y_pred_gbr.to_excel("gbr.xlsx", index = False ) # Saving the output in to an excel


# ========= alternative for train-test col discrepancy: But not a good practice

# for column in X.columns:
#     if column not in X_test.columns:
#         X_test[column] = 0

# for column in X_test.columns:
#     if column not in X.columns:
#         X_test.drop([column], axis=1, inplace=True)


#_, train_acc = model.evaluate(trainX, trainy, verbose=0)
# _, test_acc = model.evaluate(testX, testy, verbose=0)
