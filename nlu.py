import numpy as np
import json
# Loading json data
with open('data_full.json') as file:
  data = json.loads(file.read())

#Convert data into Numpy array
data = np.array(data)
#Transpose data array
data = data.T

#Read data as text and label
text = data[0]
labels = data[1]

#Split data into test set and train set
from sklearn.model_selection import train_test_split
train_txt,test_txt,train_label,test_labels = train_test_split(text,labels,test_size = 0.3)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
max_num_words = 40000
#Set intents as unique labels
classes = np.unique(labels)

#Create token dictionary
tokenizer = Tokenizer(num_words=max_num_words)
#Assign unique integer to each text based on frequency
tokenizer.fit_on_texts(train_txt)
#The token dictionary itself
word_index = tokenizer.word_index

ls=[]
for c in train_txt:
    ls.append(len(c.split()))
#Sort the ls array and get the length of 98% of the sentences    
maxLen=int(np.percentile(ls, 98))
#Converting train text into integer sequence
train_sequences = tokenizer.texts_to_sequences(train_txt)
#Making all these sequences the same length (maxLen) by adding padding or truncating them
train_sequences = pad_sequences(train_sequences, maxlen=maxLen, padding='post')
test_sequences = tokenizer.texts_to_sequences(test_txt)
test_sequences = pad_sequences(test_sequences, maxlen=maxLen, padding='post')

from sklearn.preprocessing import OneHotEncoder,LabelEncoder

label_encoder = LabelEncoder()
#By assigning integer label to each intent
integer_encoded = label_encoder.fit_transform(classes)
#Can set sparse_output to true if you have a lot of classes, because there will be a lot of zeroes in your set
onehot_encoder = OneHotEncoder(sparse_output=False)
#Changing the shape of the encoder into no. Intents row with 1 column (Transpose)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
#Convert integer 1,2,3 into one hot encoder form (001), (010), (100)
onehot_encoder.fit(integer_encoded)
train_label_encoded = label_encoder.transform(train_label)
train_label_encoded = train_label_encoded.reshape(len(train_label_encoded), 1)
train_label = onehot_encoder.transform(train_label_encoded)
test_labels_encoded = label_encoder.transform(test_labels)
test_labels_encoded = test_labels_encoded.reshape(len(test_labels_encoded), 1)
test_labels = onehot_encoder.transform(test_labels_encoded)

import wget
url ='https://www.dropbox.com/s/a247ju2qsczh0be/glove.6B.100d.txt?dl=1'
wget.download(url)

#Creating embeddings for NLU
embeddings_index={}
with open('glove.6B.100d.txt', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

#Convert dictionary to list
all_embs = np.stack(list(embeddings_index.values()))
emb_mean,emb_std = all_embs.mean(), all_embs.std()
num_words = min(max_num_words, len(word_index))+1
embedding_dim=len(embeddings_index['the'])
#Drawing random samples from a normal (Gaussian) distribution
embedding_matrix = np.random.normal(emb_mean, emb_std, (num_words, embedding_dim))
for word, i in word_index.items():
    if i >= max_num_words:
        break
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Activation, Bidirectional,Embedding
#Create sequence of layers
model = Sequential()
#Add word embeddings
model.add(Embedding(num_words, 100, trainable=False,input_length=train_sequences.shape[1], weights=[embedding_matrix]))
#Processing sequential data in both forward and backward directions within a recurrent neural network (RNN) layer
model.add(Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.1, dropout=0.1), 'concat'))
#To counter overfitting
model.add(Dropout(0.3))
#Handling sequential data and overcome the vanishing gradient problem commonly encountered in traditional RNNs
model.add(LSTM(256, return_sequences=False, recurrent_dropout=0.1, dropout=0.1))
model.add(Dropout(0.3))
#Creating a fully connected layer with relu activation function
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.3))
#Creating the last layer with softmax function for classification
model.add(Dense(classes.shape[0], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(train_sequences, train_label, epochs = 20,
          batch_size = 64, shuffle=True,
          validation_data=[test_sequences, test_labels])

import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

import pickle
import json
model.save('models/intents.h5')

with open('utils/classes.pkl','wb') as file:
   pickle.dump(classes,file)

with open('utils/tokenizer.pkl','wb') as file:
   pickle.dump(tokenizer,file)

with open('utils/label_encoder.pkl','wb') as file:
   pickle.dump(label_encoder,file)

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
class IntentClassifier:
    def __init__(self,classes,model,tokenizer,label_encoder):
        self.classes = classes
        self.classifier = model
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder

    def get_intent(self,text):
        self.text = [text]
        self.test_keras = self.tokenizer.texts_to_sequences(self.text)
        self.test_keras_sequence = pad_sequences(self.test_keras, maxlen=16, padding='post')
        self.pred = self.classifier.predict(self.test_keras_sequence)
        return self.label_encoder.inverse_transform(np.argmax(self.pred,1))[0]
    
import pickle

from tensorflow.keras.models import load_model
model = load_model('models/intents.h5')

with open('utils/classes.pkl','rb') as file:
  classes = pickle.load(file)

with open('utils/tokenizer.pkl','rb') as file:
  tokenizer = pickle.load(file)

with open('utils/label_encoder.pkl','rb') as file:
  label_encoder = pickle.load(file)

nlu = IntentClassifier(classes,model,tokenizer,label_encoder)
print(nlu.get_intent("How do I pay less tax in a legal way?"))