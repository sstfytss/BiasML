import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_test_model(X, y, model_type):
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

      model = LogisticRegression(max_iter=1000)
      print(model)
      model.fit(X_train_scaled, y_train)
      y_pred = model.predict(X_test_scaled)

    elif model_type == "random_forest":
      model = RandomForestClassifier(n_estimators=100, random_state=42)
      print(model)
      model.fit(X_train, y_train)
      y_pred = model.predict(X_test)

    elif model_type == "decision_tree":
      model = DecisionTreeClassifier(random_state=42)
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