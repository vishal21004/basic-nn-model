#  DEVELOPING A NEURAL NETWORK REGRESSION MODEL
## AIM:
To develop a neural network regression model for the given dataset.

## THEORY:
Neural network regression models learn complex relationships between input variables and continuous outputs through interconnected layers of neurons. By iteratively adjusting parameters via forward and backpropagation, they minimize prediction errors. Their effectiveness hinges on architecture design, regularization, and hyperparameter tuning to prevent overfitting and optimize performance.
### Architecture:
  This neural network architecture comprises two hidden layers with ReLU activation functions, each having 5 and 3 neurons respectively, followed by a linear output layer with 1 neuron. The input shape is a single variable, and the network aims to learn and predict continuous outputs.

## NEURAL NETWORK MODEL:
![image](https://github.com/Rithigasri/basic-nn-model/assets/93427256/de6017e4-fcd7-4a31-abbd-9a36bd7ae689)

## DESIGN STEPS:
### STEP 1:
Loading the dataset.
### STEP 2:
Split the dataset into training and testing.
### STEP 3:
Create MinMaxScalar objects ,fit the model and transform the data.
### STEP 4:
Build the Neural Network Model and compile the model.
### STEP 5:
Train the model with the training data.
### STEP 6:
Plot the performance plot
### STEP 7:
Evaluate the model with the testing data.
## PROGRAM:

```python
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('DL').sheet1

rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df=df.astype({'INPUT':'float'})
df=df.astype({'OUTPUT':'float'})
df.head()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = df[['INPUT']].values
y = df[['OUTPUT']].values
X

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)

model=Sequential([
    #Hidden ReLU Layers
    Dense(units=5,activation='relu',input_shape=[1]),
    Dense(units=3,activation='relu'),
    #Linear Output Layer
    Dense(units=1)
])

model.compile(optimizer='rmsprop',loss='mse')
model.fit(X_train1,y_train,epochs=3000)

loss= pd.DataFrame(model.history.history)
loss.plot()

X_test1 =Scaler.transform(X_test)
model.evaluate(X_test1,y_test)

X_n1=[[4]]
X_n1_1=Scaler.transform(X_n1)
model.predict(X_n1_1)

```
## DATASET INFORMATION:
![image](https://github.com/Rithigasri/basic-nn-model/assets/93427256/cdef71ea-4774-4bf2-baf2-9d7dae7d9592)


## OUTPUT:
### Training Loss Vs Iteration Plot:
![image](https://github.com/Rithigasri/basic-nn-model/assets/93427256/a7b48087-1179-4781-8786-e3d160344202)
### Epoch Training:
![image](https://github.com/Rithigasri/basic-nn-model/assets/93427256/1247ecf7-80e4-4443-ab84-c09d0cd4d541)
### Test Data Root Mean Squared Error:
![image](https://github.com/Rithigasri/basic-nn-model/assets/93427256/0114d30a-8081-4205-a158-95efe5450804)
### New Sample Data Prediction:
![image](https://github.com/Rithigasri/basic-nn-model/assets/93427256/ea52cc7b-b09f-400c-90e8-a8170793c2ef)


## RESULT:
Thus a basic neural network regression model for the given dataset is written and executed successfully.
