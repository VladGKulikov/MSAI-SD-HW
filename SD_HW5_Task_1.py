import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression

class Builder:
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray):
        # 1. Copy train dataset
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
    
    def get_subsample(self, df_share: int): 
        # 2. Shuffle data (don't miss the connection between X_train and y_train)
        X, y = shuffle(self.X_train, self.y_train, random_state=42)

        # 3. Return subsample of X_train and y_train
        part = int(len(y) * df_share / 100.0)
        return X[:part, :], y[:part]

if __name__ == "__main__":
    # 1. Load iris dataset    
    X, y = load_iris(return_X_y=True)

    # 2. Shuffle data
    X, y = shuffle(X, y, random_state=42)

    # 2.2. Divide into train / test.
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state=42)    
 
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    
    sample = Builder(X_train, y_train)

    for df_share in range(10, 101, 10):
        
        curr_X_train, curr_y_train =  sample.get_subsample(df_share)
        """
        1. Preprocess curr_X_train, curr_y_train in the way you want
        2. Train Linear Regression on the subsample
        3. Save or print the score to check how df_share affects the quality
        """
        clf.fit(curr_X_train, curr_y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Value of df_share = {df_share:.2f}%, accuracy = {accuracy:.2f}")

# Ouput
# Value of df_share = 10.00%, accuracy = 0.70
# Value of df_share = 20.00%, accuracy = 0.73
# Value of df_share = 30.00%, accuracy = 0.87
# Value of df_share = 40.00%, accuracy = 0.93
# Value of df_share = 50.00%, accuracy = 0.97
# Value of df_share = 60.00%, accuracy = 1.00
# Value of df_share = 70.00%, accuracy = 1.00
# Value of df_share = 80.00%, accuracy = 1.00
# Value of df_share = 90.00%, accuracy = 1.00
# Value of df_share = 100.00%, accuracy = 1.00