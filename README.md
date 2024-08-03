

# Assignment---Data-1202<BR>
Nana Esi Hinson 
Student Number - 100957828

<B>Project Title</B> <BR>
In this study, we used the drugdataset.csv to evaluate the effectiveness of the Support Vector Machine (SVM) and Naive Bayes (NB) models.

--------------

<B>Prerequisites</B><BR>
Jupyter Notebook via Anaconda to run Python

---------------

<B>Installing </B><BR>

import pandas 
import pandas as pd and load the dataset<BR>
Run the codes in Jupyter environment<BR>
 Python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
file_path = '/mnt/data/drugdataset.csv'
data = pd.read_csv(file_path)


------------

<B>Code for using SVM and NB models</B>

# Preprocess the data
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Split the data into features and target
X = data.drop('Drug', axis=1)
y = data['Drug']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train an SVM model
svm_model = SVC()
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Train a Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)


-----------------

<B>Running the Tests</B><BR>
Test the coding done

# Evaluate the SVM model
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_classification_report = classification_report(y_test, y_pred_svm)


# Evaluate the Naive Bayes model
nb_accuracy = accuracy_score(y_test, y_pred_nb)
nb_classification_report = classification_report(y_test, y_pred_nb)



Results expected

Results
SVM Model
Accuracy: 0.975

              precision    recall  f1-score   support

           0       1.00      1.00      1.00         6
           1       1.00      1.00      1.00         3
           2       1.00      0.80      0.89         5
           3       1.00      1.00      1.00        11
           4       0.94      1.00      0.97        15

    accuracy                           0.97        40
   macro avg       0.99      0.96      0.97        40
weighted avg       0.98      0.97      0.97        40

Naive Bayes Model
Accuracy: 0.9

              precision    recall  f1-score   support

           0       0.75      1.00      0.86         6
           1       0.75      1.00      0.86         3
           2       0.83      1.00      0.91         5
           3       1.00      1.00      1.00        11
           4       1.00      0.73      0.85        15

    accuracy                           0.90        40 
   macro avg       0.87      0.95      0.89        40 
weighted avg       0.92      0.90      0.90        40

---------------------------

<B>Break down into end to end tests</B><BR>
Divide the tests into end-to-end scenarios.
1. Accuracy: The ratio of correctly predicted outcomes to all forecasts made is known as accuracy. 97% and 90%, respectively, in this instance.<BR>
2. Precision: The degree of accuracy of your model is what defines precision. Put another way, you may ask how often a model predicts the future accurately. In this instance, SVM outperforms Naive Bayes in terms of precision across all classes. This suggests that SVM performs better in reducing false positives<BR>.
3. Recall: The percentage of accurately anticipated positive observations to all observations made during the actual class is known as recall. SVM is more effective in identifying true positives in this instance than Naive Bayes, as evidenced by its greater recall scores.<BR>
4. F1 score: SVM performs better overall than Naive Bayes<BR> because it has a higher F1-score across all classes, which balances precision and recall.

-----------

<B>Built With</B> 

Jupyter from Anaconda<BR>
Excel Dataset

---------
Comparison and Understanding

Model Performance: Precision, recall, and F1-score are the three main measures where SVM beats Naive Bayes. This implies that SVM performs better on this specific classification challenge. <BR>
Use Cases: If achieving high accuracy and minimizing false positives and false negatives is the aim, SVM is the recommended choice. Despite being a little less accurate, Naive Bayes can still be helpful because of its ease of use and speedier computation, especially for very big datasets or situations where quick, approximative answers are sufficient. <BR>
In general, SVM offers superior classification for the data.
