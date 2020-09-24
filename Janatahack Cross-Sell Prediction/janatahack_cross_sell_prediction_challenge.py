import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score




df = pd.read_csv('../input/janatahack-crosssell-prediction/train.csv')
df_test = pd.read_csv('../input/janatahack-crosssell-prediction/test.csv')
df.info()


df["Gender","Vehicle_Age","Vehicle_Damage"] = df["Gender","Vehicle_Age","Vehicle_Damage"].astype('category')



df["Gender"] = df["Gender"].replace({'Male':1, 'Female':0})
df["Vehicle_Damage"] = df["Vehicle_Damage"].replace({'Yes':1, 'No':0})

df["Vehicle_Age"] = df["Vehicle_Age"].replace({'1-2 Year':2, '< 1 Year':1,'> 2 Years':3})


df_test["Gender"] = df_test["Gender"].replace({'Male':1, 'Female':0})
df_test["Vehicle_Damage"] = df_test["Vehicle_Damage"].replace({'Yes':1, 'No':0})

df_test["Vehicle_Age"] = df_test["Vehicle_Age"].replace({'1-2 Year':2, '< 1 Year':1,'> 2 Years':3})



X = df.drop('Response',axis=1)
y = df['Response'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)




from xgboost import XGBClassifier




model = XGBClassifier(silent=True,
                      booster = 'gbtree',
                      scale_pos_weight=7,
                      learning_rate=0.01,  
                      colsample_bytree = 0.7,
                      subsample = 0.5,
                      max_delta_step = 3,
                      reg_lambda = 2,
                     objective='binary:logistic',
                      
                      n_estimators=818, 
                      max_depth=8,
                     )

eval_set = [(X_test, y_test)]
eval_metric = ["logloss"]
%time model.fit(X_train, y_train,early_stopping_rounds=50, eval_metric=eval_metric, eval_set=eval_set)


predictions = model.predict_proba(X_test)[:,-1]

roc_auc_score(y_test, predictions)


p1 = model.predict_proba(df_test)[:,-1]


submission_df = pd.read_csv("../input/janatahack-crosssell-prediction/sample_submission.csv")




submission_df['Response']=p1
submission_df.to_csv('xgb python first try.csv',index=False)
submission_df.head(5)




Y = np.round(p1)

submission = pd.DataFrame(Y)

submission = submission.rename(columns={0: "Response"})
submission.index = submission['Response']
submission.drop('Response',axis=1,inplace=True,index = False)

p1>0.5

id = df_test['id']

temp = df_test['id']
temp['Response'] = submission
temp.to_csv("../working/submission3.csv", index = False)



submission = pd.DataFrame(p1)
submission = submission.rename(columns={0: "Class"})
submission.index = submission['Class']
submission.drop('Class',axis=1,inplace=True)
submission.to_csv('submissiom.csv',header=True, index=True)

