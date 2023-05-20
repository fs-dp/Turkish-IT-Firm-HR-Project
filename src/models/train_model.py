import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score ,confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# Load the preprocessed dataset
df_merged = pd.read_pickle("../models/df_merged.pkl")

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Define the target variable
y = df_merged['Attrition']

# Drop the target variable from the input features
df_merged.drop("Attrition", axis=1, inplace=True)

# Apply the scaler to the dataset
X = scaler.fit_transform(df_merged)

#Logistic regression model function
def logistic_regression(X, y, test_size):
    # Split the dataset into train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Initialize the Logistic Regression model
    model = LogisticRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Visualize the confusion matrix
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Compute the accuracy score
    acc_score = accuracy_score(y_test, y_pred)

    print(f"Accuracy is {round(100*acc_score,2)}%")

    # Print the classification report
    print(classification_report(y_test, y_pred))

#Random forest model function
def random_forest(X, y, test_size):
    # Split the dataset into train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Initialize Random Forest model
    model = RandomForestClassifier()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Visualize the confusion matrix
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Compute the accuracy score
    acc_score = accuracy_score(y_test, y_pred)

    print(f"Accuracy is {round(100*acc_score,2)}%")

    # Print the classification report
    print(classification_report(y_test, y_pred))

#Deep learning model function
def deep_learning(X, y, test_size):
    # Split the dataset into train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Initialize the Deep Learning model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=500, activation='relu', input_shape=(50,), kernel_regularizer=l2(0.01)))
    model.add(tf.keras.layers.Dense(units=500, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(tf.keras.layers.Dense(units=500, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) # Changed to sigmoid activation and 1 unit
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=["accuracy"])

    # Implement early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

    # Train the Deep Learning model
    epoch_hist = model.fit(X_train, y_train, epochs=100, batch_size=50, validation_split=0.2, callbacks=[es])

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    y_pred = (y_pred>0.5)

    plt.plot(epoch_hist.history['loss'])
    plt.plot(epoch_hist.history['val_loss']) # Added validation loss
    plt.title('Model Loss Progress During Training/Validation')
    plt.ylabel('Training and Validation Losses')
    plt.xlabel('Epoch Number')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.show()

    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Visualize the confusion matrix
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Compute the accuracy score
    acc_score = accuracy_score(y_test, y_pred)

    print(f"Accuracy is {round(100*acc_score,2)}%")

    # Print the classification report
    print(classification_report(y_test, y_pred))

    # Print the classification report
    print(classification_report(y_test, y_pred))

# Call the functions
logistic_regression(X=X, y=y, test_size=0.2)
random_forest(X=X, y=y, test_size=0.2)
deep_learning(X=X, y=y, test_size=0.25)