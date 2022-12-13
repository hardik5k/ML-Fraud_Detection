import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn import model_selection
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_auc_score, f1_score,plot_confusion_matrix,plot_roc_curve


train = pd.read_csv("../datasets/preprocessed/train_preprocessed.csv")
test = pd.read_csv("../datasets/preprocessed/test_preprocessed.csv")

def submit(y_pred):
    sample = pd.read_csv('../mock_submission_real.csv')
    sample.isFraud = y_pred
    sub_name = '../submission.csv'
    sample.to_csv(sub_name, index=False)

y = train[['isFraud']]

X = train.drop(['isFraud'], axis = 1)
del train


"""## Hyperparamter Tuning"""

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

# y_train.values.ravel().shape

# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_jobs= -1,class_weight='balanced')
# estimators = [5, 10, 50, 100]
# max_depth = [1, 5, 10, 15, 20]
# param = {'n_estimators': estimators ,'max_depth':max_depth}
# model = RandomizedSearchCV(estimator=clf,  param_distributions=param, cv=3, n_iter=6, scoring='roc_auc')
# model.fit(X_train,y_train.values.ravel())

# model.best_params_

pred_valid = np.zeros(X.shape[0])
pred_test = np.zeros(test.shape[0])
cmx = []
roc_auc_scores = []


n_fold = 5
folds = StratifiedKFold(n_splits=n_fold, shuffle = True)
for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
    
    clf = RandomForestClassifier(max_depth=15,n_estimators=10)
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    clf.fit(X_train,y_train.values.ravel())
    
    pred_valid[valid_index] = clf.predict_proba(X_valid)[:, 1]
    pred_test += clf.predict_proba(test)[:, 1] / folds.n_splits
    roc_auc_scores.append(roc_auc_score(y_valid, pred_valid[valid_index]))
    print('Fold %2d AUC : %.6f' % (fold_n + 1, roc_auc_score(y_valid, pred_valid[valid_index]))) 
    
    # Confusion matrix
    cmx.append(confusion_matrix(y_valid, pred_valid[valid_index].round()))

    


print('Full AUC score %.6f' % roc_auc_score(y, pred_valid))

conf_matrix = np.average(cmx, axis= 0)
sns.heatmap(conf_matrix, annot=True, fmt='g')
plt.show()

submit(pred_test)