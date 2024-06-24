import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import seaborn as sns
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
dataset = pd.read_csv("./train.csv")

# Drop rows with missing values
dataset = dataset.dropna()

# Replace categorical values with numerical labels
dataset.replace({"Loan_Status": {'Y': 1, 'N': 0},
                 "Dependents": {'3+': 4},
                 "Married": {'Yes': 1, 'No': 0},
                 "Gender": {'Male': 1, 'Female': 0},
                 "Self_Employed": {'Yes': 1, 'No': 0},
                 "Property_Area": {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
                 "Education": {'Graduate': 1, 'Not Graduate': 0}
                 }, inplace=True)

# Visualize categorical variables
sns.countplot(x='Education', hue='Loan_Status', data=dataset)
sns.countplot(x='Married', hue='Loan_Status', data=dataset)

# Prepare features and target variable
X = dataset.drop(columns=['Loan_Status', 'Loan_ID'], axis=1)
Y = dataset["Loan_Status"]

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2, stratify=Y)

# Initialize SVM classifier with linear kernel
classifier = SVC(kernel='rbf')

# Train the model
classifier.fit(X_train, Y_train)

# Evaluate training data accuracy
train_data_prediction = classifier.predict(X_train)
train_data_accuracy = accuracy_score(train_data_prediction, Y_train)
print(f"Training Accuracy: {train_data_accuracy * 100:.2f}%")

# Evaluate test data accuracy
test_data_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(test_data_prediction, Y_test)
print(f"Test Accuracy: {test_data_accuracy * 100:.2f}%")

# Save the model to disk
with open("model.pkl", "wb") as file:
    pickle.dump(classifier, file)
