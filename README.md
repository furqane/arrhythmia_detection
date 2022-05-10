# Arrhythmia Detection (Classification)

This project served as my capstone project for my Data Science bootcamp offered by Lighthouse Labs.

This project is a combination of two of my interests: biomedical engineering and deep learning. Data science plays a crucial role in cutting edge research in the field of healhcare and biomedical engineering, from understanding disease mechanisms to improving medical imaging.

For this project, I have decided to explore a classification problem in regards to heart arrhythmias. By being able to develop a high accuracy machine/deep learning model that can analyze electrocardiogram (ECG/EKG) data and classify not only whether the ECG reflects an arrythmia but also which type, we can improve medical devices and wearable technologies.

The dataset used originated from Kaggle. This [link](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) will redirect you to the dataset page. The samples in the dataset have been cropped, downsampled, and padded with zeros to the fixed dimension of 188. 

## Performing Exploratory Data Analysis (EDA)

As noted in the description of the dataset, the dataset contains 109446 samples, and 5 different classes. The classes are as follows:
* N/0: Non-ecotic beats (normal beat) 
* S/1: Supraventricular ectopic beats 
* V/2: Ventricular ectopic beats 
* F/3: Fusion Beats 
* Q/4: Unknown Beats

The image below is an example of a S/1 sample ECG:

![F1](https://user-images.githubusercontent.com/90627794/167512661-0fe95230-65fc-4e29-acde-de4258de3714.png)

Following [basic_EDA.ipynb](basic_EDA.ipynb) in this projects' repository, the distribution of the classes is found to be as follows:

```python
0.0    72471
4.0     6431
2.0     5788
1.0     2223
3.0      641
Name: 187, dtype: int64
```
The classes are not well balanced. In theory, implimenting an upsampling technique to equalize the size of the classes will help the classification results. For personal validation on the benefits of upsampling, I first chose to focus on developing the best architecture for my Neural Netowrk. Then I upsampled the data to observe any performance changes in its classification. 

## Building a base Dense Neural Network (DNN)

Following [base_DNN.ipynb](base_DNN.ipynb), I developed a baseline DNN to compare iterative models to. The base model consists of following layers: input layer -> 64 nodes hidden layer -> 32 nodes hidden layer -> 5 output nodes. A softmax activation function was used in the output layer to normalize the output of the model to a probability distribution of which the sum is 1. In this case, 5 output nodes are specified to reflect the 5 classes in the dataset.

To measure the performances of these models, a confusion matrix was utilized. This allows me to see how exactly the model misclassifies the samples. 

![Unknown](https://user-images.githubusercontent.com/90627794/167513519-f52e5d3a-01dc-463f-9f89-be9aa59eae23.png)

## Building Long Short-Term Memory NNs (LSTMs)

In theory, an LSTM is the perfect NN architecture to utlitize for this problem. LSTMs are great at processing sequential/time series data. They look not only at a single data point but rather, take into consideration the whole sequence. What is an ECG, but a sequence?

Following [base_LSTM.ipynb](base_LSTM.ipynb), the resulting confusion matrix is as displayed:

![Unknown-2](https://user-images.githubusercontent.com/90627794/167514899-afaf6f08-3832-4957-ba97-8928b59475d8.png)

This performs slightly better than the base model. Thus began the iterative process of trying different stackings of layers and messing around with hyperparamters. One of the iterations have been pushed to this repository: [itr1_LSTM.ipynb](itr1_LSTM.ipynb).

## Final LSTM Model

This following section contains two parts, one without upsampling and one with.

### Without Upsampling

Following [itr2_LSTM.ipynb](itr2_LSTM.ipynb), the resulting confusion matrix is shown below:

![Unknown-3](https://user-images.githubusercontent.com/90627794/167515363-722ecdbf-1915-44f8-8614-659e6128ba85.png)

This model has already shown a substantial increase in performance when compared to the baseline. For example, there is a 23% in the F/3 class. As hypothesized, LSTMS are powerful for time series problems.

### With Upsampling

To confirm/validate the benefit of upsampling, the dataset was modified to contain 20,000 samples in each class. This change is reflected in [final_LSTM.ipynb](final_LSTM.ipynb). The resulting confusion matrix can be seen:

![Unknown-4](https://user-images.githubusercontent.com/90627794/167515948-05f02210-753f-48d7-b1b3-8a9bb9f751b5.png)

There is an overall increase in classification accuracy by upsampling the dataset, with as much as a 10% increase for the S/1 class. This suggests the benefits of having a balanced dataset for machine/deep learning problems are significant (as is confirmed in wide-spread practice). 

## Moving Forward

In the future, the next iteration of this project I would like to explore is predicting cardiac arrest prior to its' occurance. This could have profound live-saving implications in healthcare. This has been attempted as well in academia, in particular by Anna Goldenberg and her research team. You can see her presentation in [this Youtube video](https://www.youtube.com/watch?v=jNrTRs0lqWo). 

## TensorFlow Code for Final LSTM Model

```python
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam

in_nn = Input(shape=(X_train.shape[1],X_train.shape[2]), name='in_nn')

lstm1 = LSTM(units=128, name='lstm1', return_sequences = True)(in_nn)   #(takes in for shape (batch_size, size1, size2))
lstm2 = LSTM(units=64, name='lstm2', return_sequences = True)(lstm1)
lstm3 = LSTM(units=64, name='lstm3', return_sequences = True)(lstm2)
flatten = Flatten()(lstm3)
dense1 = Dense(units=64, activation='relu', name='dense1')(flatten)
dense2 = Dense(units=32, activation='relu', name='dense2')(dense1)
dense3 = Dense(units=5, activation='softmax', name='dense3')(flatten)

model = Model(inputs=in_nn, outputs=dense3)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=25, batch_size=16)
```





