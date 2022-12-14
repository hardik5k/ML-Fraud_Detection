import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn import model_selection
from sklearn.model_selection import train_test_split, StratifiedKFold
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
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

"""## Random Over/Under Sampling"""

# over = RandomOverSampler(sampling_strategy=0.25)
# under = RandomUnderSampler(sampling_strategy=0.25)
# X_over, y_over = over.fit_resample(X, y)

# X_f, y_f = under.fit_resample(X_over, y_over)

"""## Hyperparamter Tuning"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

y_train.values.ravel().shape

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
clf = XGBClassifier(
        learning_rate=0.02,
        subsample=0.8,
        tree_method='gpu_hist',
        colsample_bytree=0.85,
        scale_pos_weight = 10,
        missing=-1,
        reg_alpha=0.15,
        reg_lambda =0.85
    )
    
estimators = [100,500,1000,2000]
max_depth = [9, 10, 12, 15]
param = {'n_estimators': estimators ,'max_depth':max_depth}
model = RandomizedSearchCV(estimator=clf,  param_distributions=param, verbose=1,cv=3, n_iter=6, scoring='roc_auc')
model.fit(X_train,y_train.values.ravel())

model.best_params_

"""## XGBOOST"""

pred_valid = np.zeros(X.shape[0])
pred_test = np.zeros(test.shape[0])
feature_importance_df = pd.DataFrame()
cmx = []
main_features = []
roc_auc_scores = []

n_fold = 5
folds = StratifiedKFold(n_splits=n_fold, shuffle = True)
for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
    clf = XGBClassifier(
        n_estimators=1000,
        max_depth=15,
        learning_rate=0.03,
        subsample=0.8,
        tree_method='gpu_hist',
        colsample_bytree=0.85,
        scale_pos_weight = 10,
        missing=-1,
        reg_alpha=0.15,
        reg_lambda =0.85
    )
    
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    clf.fit(X_train,y_train)
    pred_valid[valid_index] = clf.predict_proba(X_valid)[:, 1]
    pred_test += clf.predict_proba(test)[:, 1] / folds.n_splits
    roc_auc_scores.append(roc_auc_score(y_valid, pred_valid[valid_index]))
    print('Fold %2d AUC : %.6f' % (fold_n + 1, roc_auc_score(y_valid, pred_valid[valid_index]))) 
    
    # Confusion matrix
    cmx.append(confusion_matrix(y_valid, pred_valid[valid_index].round()))
    
    # Features importance
    feature_imp = clf.get_booster().get_score(importance_type='gain')
    main_features.append(feature_imp)
    
print('Full AUC score %.6f' % roc_auc_score(y, pred_valid))

conf_matrix = np.average(cmx, axis= 0)
sns.heatmap(conf_matrix, annot=True, fmt='g')
plt.show()

submit(pred_test)
