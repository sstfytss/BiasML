import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_model(file_name):
  with open(file_name, 'r') as f:
    loaded_params = json.load(f)
  
  return loaded_params

def save_model(best_params, file_name):
  with open(file_name, 'w') as f:
    json.dump(best_params, f)

def train_test_model(X, y, model_type, file_path=False):
    """
    Train either a logistic regression, decision tree, or random forest model on the provided data.

    Parameters:
    X (pd.DataFrame): Features.
    y (pd.Series): Target variable.
    model_type (str): Type of model to train.

    Returns:
    None
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train and test
    if model_type == "logistic":
      # scale
      scaler = StandardScaler()
      X_train_scaled = scaler.fit_transform(X_train)
      X_test_scaled = scaler.transform(X_test)

      if not file_path:
        model = LogisticRegression(max_iter=1000)
      else:
        loaded_params = load_model(file_path)
        model = LogisticRegression(**loaded_params)
      print(model)
      model.fit(X_train_scaled, y_train)
      y_pred = model.predict(X_test_scaled)

    elif model_type == "random_forest":
      if not file_path:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
      else:
        loaded_params = load_model(file_path)
        model = RandomForestClassifier(**loaded_params)
      print(model)
      model.fit(X_train, y_train)
      y_pred = model.predict(X_test)

    elif model_type == "decision_tree":
      if not file_path:
        model = DecisionTreeClassifier(random_state=42)
      else:
        model = DecisionTreeClassifier(**loaded_params)
      print(model)
      model.fit(X_train, y_train)
      y_pred = model.predict(X_test)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("\n")

    # make sure y_pred is a dataframe
    y_pred = pd.DataFrame(y_pred, columns=["target"], index=X_test.index)

    return X_train, X_test, y_train, y_test, y_pred, model