**Titanic Survival Prediction Models**<br>

This repository contains three different models built to predict the survivors of the Titanic disaster, as part of the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic/overview). The three models are:
- Hybrid Model of XGBoost and Random Forest
- TensorFlow Keras Sequential Model
- Scikit-learn Logistic Regression Model

Hybrid Model of XGBoost and Random Forest<br>
The hybrid model is a combination of a Random Forest model and an XGBoost model. The XGBoost model requires no filling in of null values and only needs one-hot encoding for categorical features. On the other hand, the Random Forest model requires imputation of missing values by replacing them with the respective median values, in addition to one-hot encoding. We conduct grid searches with cross-validation splitting to find the optimal hyperparameters for each model that would minimize the log loss. Both models are given equal weight in the hybrid model.<br>
With minimal feature engineering, the hybrid model achieves an accuracy of approximately 0.77 on the competition test dataset.

TensorFlow Keras Sequential Model<br>
The TensorFlow Keras Sequential model is a neural network model built using the Keras API. The model consists of multiple layers of densely connected neurons, with four features ('Sex', 'Age', 'Pclass', and 'Fare')being selected for the input layer. We use the Adam optimizer and binary cross-entropy loss function to train the model. With minimal feature engineering, the model achieves an accuracy of approximately 0.75 on the competition test dataset.

Scikit-learn Logistic Regression Model<br>
The Scikit-learn Logistic Regression model is a linear model that uses logistic regression to predict the probability of survival. We dropped null values from the feature columns 'Sex', 'Age', 'Pclass', 'Fare', and 'Parch'. With minimal feature engineering, the model achieves an accuracy of approximately 0.58 on the competition test dataset.

Conclusion<br>
In this repository, we have built and compared three different models for predicting the survivors of the Titanic disaster. The hybrid model achieved the highest accuracy, followed by the TensorFlow Keras Sequential model and the Scikit-learn Logistic Regression model. We hope that these models and the accompanying code will be helpful to others who are interested in learning about and building machine learning models.
