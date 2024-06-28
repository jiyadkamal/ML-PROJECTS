import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import nltk

data = pd.read_csv('train.csv')
data = data.dropna()
data['content'] = data['title']+' '+data['author']

portstem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [portstem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

data['content'] = data['content'].apply(stemming)



x = data['content'].values
y = data['label'].values

vectorizor = TfidfVectorizer()
vectorizor.fit(x)
x = vectorizor.transform(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)

model = LogisticRegression()
model.fit(x_train,y_train)

prediction = model.predict(x_test[0])
print(prediction)