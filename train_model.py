import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
df=pd.read_csv("gender_classification_v7.csv")
x = df.drop("gender", axis=1)
y = df["gender"] 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=RandomForestClassifier(n_estimators=200,random_state=42)
model.fit(x_train,y_train)
prediction=model.predict(x_test)
joblib.dump(model,"model.pkl")