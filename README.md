# Arrhythmia Detection (Classification)

This project served as my final project for my Data Science bootcamp offered by Lighthouse Labs.

This project is a combination of two of my interests: biomedical engineering and deep learning. Data science plays a crucial role in new and upcoming research in the field of healhcare, from understanding disease mechanisms to improving medical imaging.

For this project, I have decided to explore a classification problem in regards to heart arrhythmias. By being able to develop a machine/deep learning model that can analyze electrocardiogram (ECG/EKG) data and classify not only whether the ECG reflects an arrythmia but also which type, can improve medical devices and wearable technologies.

The dataset used originated from Kaggle. This [link](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) will redirect you to the dataset page.

## Performing Exploratory Data Analysis (EDA)

As noted in the description of the dataset, the dataset contains 109446 samples, and 5 different classes. The classes are as follows:
N: Non-ecotic beats (normal beat) 
S: Supraventricular ectopic beats 
V: Ventricular ectopic beats 
F: Fusion Beats 
Q: Unknown Beats

Following the file labeled as 'basic_EDA.ipynb' in this projects' repository, the distribution of the classes is found to be as follows: 

```python
0.0    72471
4.0     6431
2.0     5788
1.0     2223
3.0      641
Name: 187, dtype: int64
```
