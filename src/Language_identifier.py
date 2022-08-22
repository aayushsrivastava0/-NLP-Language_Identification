import numpy as np  # linear algerbra
import pandas as pd  # data processing
import matplotlib.pyplot as plt  # plotting
import seaborn as sns  # plotting
import json  # data processing(.json)

from sklearn.preprocessing import LabelEncoder  # Creates placeholders for categorical variables
from sklearn.feature_extraction.text import CountVectorizer  # converts text into vector matrix
from sklearn.model_selection import train_test_split  # split data into training and testing sets
from sklearn.naive_bayes import MultinomialNB  # ML model for naive bayes
from sklearn.metrics import accuracy_score, confusion_matrix  # measure the accuracy of the model
from sklearn.metrics import classification_report  # classification report of the model

import re  # NLP
import nltk  # natural langauge processing
from nltk.tokenize import word_tokenize  # tokenizer1
from nltk.stem import PorterStemmer  # stemmer
from nltk.corpus import stopwords  # stopwords
###python -m nltk.downloader punkt


import tensorflow as tf  # create neural networks
from tensorflow.keras import Sequential  # create squential NN model
from tensorflow.keras.layers import Dense  # implements the operation: output = activation(dot(input, kernel) + bias)
from tensorflow.keras.utils import plot_model  # plot model architecture
from tensorflow.keras.callbacks import EarlyStopping  # early stopping of training
from tensorflow.keras.models import load_model  # load saved model

from sklearn.model_selection import GridSearchCV  # hyperparameter optimization
from sklearn.model_selection import RandomizedSearchCV  # hyperparameter optimization
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier  # linking keras model to sklearn


data = pd.read_csv('dataset.csv', encoding='utf-8').copy()
                                                           # utf-8 encoding use to be able to read text in other langauge
#print(data.shape)  # shape of the dataset

#print(data.head())

# Preprocessing the data

data = data.drop_duplicates(subset='Text')
data = data.reset_index(drop=True)

# adding nonalphanumeric char to stopwords
nonalphanumeric = ['\'', '.', ',', '\"', ':', ';', '!', '@', '#', '$', '%', '^', '&',
                 '*', '(', ')', '-', '_', '+', '=', '[', ']', '{', '}', '\\', '?',
                 '/','>', '<', '|', ' ']

stopwords = nonalphanumeric
def clean_text(text):
    """
    takes text as input and returns cleaned text after tokenization,
    stopwords removal and stemming
    """
    tokens = word_tokenize(text) # creates text into list of words
    words = [word.lower() for word in tokens if word not in stopwords] # creates a list with words which are not stopwords
    words = [PorterStemmer().stem(word) for word in words] # stems(remove suffixes and prefixes)  words
    return " ".join(words) # joins the list of cleaned words into a sentence string

# applying clean_text function to all rows in 'Text' column

data['clean_text'] = data['Text'].apply(clean_text)


#using LabelEncoder to get placeholder number values for categorical variabel 'language'
le = LabelEncoder()
data['language_encoded'] = le.fit_transform(data['language'])
print(data.head())

lang_list = [i for i in range(22)]
lang_list = le.inverse_transform(lang_list)
lang_list = lang_list.tolist()
print(lang_list)

# plotting a language-wise freqeuncy distribtion for number of samples in each language

# plt.figure(figsize=(10,10))
# plt.title('Language Counts')
# ax = sns.countplot(y=data['language'], data=data)
# plt.show()


# Preprocessing data
# We see that there are english words present in the chinsese text and hence we will remove them.
def remove_english(text):
    """
    function that takes text as input and returns text without english words
    """
    pat = "[a-zA-Z]+"
    text = re.sub(pat, "", text)
    return text


data_Chinese = data[data['language']=='Chinese'] # Chinese data in dataset
clean_text = data.loc[data.language=='Chinese']['clean_text']
clean_text = clean_text.apply(remove_english) # removing english words

data_Chinese.loc[:,'clean_text'] = clean_text

# removing old chinese text and appending new cleaned chinese text

data.drop(data[data['language']=='Chinese'].index, inplace=True, axis=0)
data = data.append(data_Chinese)

# shuffling dataframe and resetting index

data =data.sample(frac=1).reset_index(drop=True)


# Splitting into inputs and targets

# defining input variable
# vectorizing input varible 'clean_text' into a matrix

x = data['clean_text']

cv = CountVectorizer() # ngram_range=(1,2)
x = cv.fit_transform(x)

# changing the datatype of the number into uint8 to consume less memory
x = x.astype('uint8')

# defining target variable

y = data['language_encoded']
# splitting data into training and testing datasets

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

"""
Creating a Multilayer Perceptron model
"""
# converting csr matrix into np.ndarray supported by tensorflow

x_train = x_train.toarray()
x_test = x_test.toarray()
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

INPUT_SIZE = x_train.shape[1]
print(INPUT_SIZE)
OUTPUT_SIZE = len(data['language_encoded'].unique())
print(OUTPUT_SIZE)


# epochs and batch_size hyperparameters

EPOCHS = 10
BATCH_SIZE = 128

# configuring early stopping

es = EarlyStopping(monitor='accuracy', patience=1)

# creating the MLP model

model = Sequential([
    Dense(100, activation='softsign', kernel_initializer='glorot_uniform', input_shape=(INPUT_SIZE,)),
    Dense(80, activation='softsign', kernel_initializer='glorot_uniform'),
    Dense(50, activation='softsign', kernel_initializer='glorot_uniform'),
    Dense(OUTPUT_SIZE, activation='softmax')
])
# compiling the MLP model

model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# fitting the model with earlystopping callback to avoid overfitting

hist = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.3, callbacks=[es], verbose=2)

# summary of the MLP model

model.summary()

# # architetcure of the MLP model
# plot_model(model, show_shapes=True)

# evaluating the loss and accuracy of the model

loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
print('Accuracy %.3f'%accuracy)

# creating loss vs epochs plot

# plt.title('Learning Curve')
# plt.xlabel('Epochs')
# plt.ylabel('Categorical Crossentropy')
# plt.plot(hist.history['loss'], label='train')
# plt.plot(hist.history['val_loss'], label='val')
# plt.legend()
# plt.show()


# saving the model

model.save('language_identifcation_model.h5')

#################################################################
# PERFORMING PREDICTIONS
#################################################################

model = load_model('language_identifcation_model.h5')

# using the model for prediction

sent = """आप कितना सोचते हो
अगर आप ठिठुरती रातों को गिनें
अरे क्या आप मिल सकते हैं (अरे, क्या आप मिल सकते हैं?)
क्या तुम मिलोगे (क्या तुम मिलोगे?)
सर्दियों का अंत बताओ
एक कोमल वसंत के दिन तक
मैं चाहता हूं कि तुम तब तक रहो जब तक फूल खिल न जाएं
ज्यों का त्यों"""


sent = cv.transform([sent])
ans = model.predict(sent)
ans = np.argmax(ans)
print(le.inverse_transform([ans]))