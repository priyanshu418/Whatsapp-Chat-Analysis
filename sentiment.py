import pickle

def fff(x): # converting .txt to list
  with open(x) as file:
    line = []
    for lines in file.readlines():
      line.append(lines)
    return line
line = fff('train.txt')

import pandas as pd
import re

def csv(line): # converting list into data frame
  list1,list2 = [],[]
  for lines in line:
    x,y = lines.split(';')
    y = y.replace('\n','')
    list1.append(x)
    list2.append(y)
  df = pd.DataFrame(list(list1),columns=['sentence'])
  df['emotion'] = list2
  return df

df = csv(line)

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

wn = WordNetLemmatizer()

def lem(x): # lemmating the text
  print("Lemmatizing...")
  corpus = []
  i=1
  for words in x:
    words = words.split()
    y = [wn.lemmatize(word) for word in words if not word in stopwords.words('english')]
    y =  ' '.join(y)
    corpus.append(y)
  return corpus
x = lem(df['sentence']) # list

test_line = fff('test.txt')
test_df = csv(test_line)
x_test = lem(test_df['sentence']) # lemmatizing

all = x + x_test # list
y = df.iloc[:,1].values # training data as array
y_test = test_df.iloc[:,1].values  # testing data ar array

from tensorflow.keras.layers import Embedding,LSTM,Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential

# y_train = pd.DataFrame(y) # data frame

# creating a internal dictionary where we are indexing each word
tokenizer = Tokenizer(num_words=10000, split=' ')
tokenizer.fit_on_texts(all)

def conv(all):
    # replacing each word with it's corresponding value acc. to dict
    X1 = tokenizer.texts_to_sequences(all)
    X1 = pad_sequences(X1, maxlen=20, padding='post', truncating='post')
    return X1

# all is list type
# X1 = conv(all)
# X_train = X1[:16000]
# X_test = X1[16000:]


# Y_train = pd.get_dummies(y_train).values # creating one hot vector for all unique values of y
Y_test = pd.get_dummies(y_test).values


model = Sequential()
model.add(Embedding(input_dim=10000,output_dim = 64,input_length=20)) # dim = dimension, number of dimensions of vector
model.add(LSTM(64)) # 64 is number of neurons in the LSTM
model.add(Dense(6,activation='softmax'))

model.compile(optimizer='rmsprop',loss='mse',metrics=['accuracy'])
# model.fit(X_train,Y_train,batch_size=32,epochs=10,verbose=2,validation_split=0.2)


# saving the model to disk
filename = 'finalized_lstm_model.sav'
# pickle.dump(model, open(filename, 'wb'))
# print("saved the model in disk")


# loading the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

loss,acc = model.evaluate(X_test,Y_test)
print("loss : ", loss)
print("accuracy : ", acc)

preds = model.predict(X_test)

# test box in stremlit app
sentence = st.text_input('Input your sentence here:')
if sentence:
    st.write(my_model.predict(sentence))