import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle as pkl

def clean_data(_):
    data = pd.read_csv(r'C:\PROJECTS FOR GITHUB\applied_Machine_Learning\Cancer_prediction\data.csv')
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

def create_model(data):
    X = data.drop(['diagnosis'], axis=1)
    Y = data['diagnosis']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(Y_test, Y_pred))
    print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))
    print("Classification Report:\n", classification_report(Y_test, Y_pred))
    return model, scaler

def main():
   data = clean_data(None)
   model, scaler = create_model(data)

   model_path = r'C:\PROJECTS FOR GITHUB\applied_Machine_Learning\Cancer_prediction\model.pkl'
   scaler_path = r'C:\PROJECTS FOR GITHUB\applied_Machine_Learning\Cancer_prediction\scaler.pkl'

   
   import os
   os.makedirs(os.path.dirname(model_path), exist_ok=True)

   with open(model_path, 'wb') as f:
      pkl.dump(model, f)
   

   with open(scaler_path, 'wb') as f:
      pkl.dump(scaler, f)
   


if __name__ == "__main__":
    main()
