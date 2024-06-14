import re
from nltk.corpus import stopwords
from gensim.models.fasttext import load_facebook_model
import numpy as np
from gensim.test.utils import datapath
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
import joblib

model = load_facebook_model(datapath("cc.id.300.bin"))
listStopword = set(stopwords.words('indonesian'))
def cleansing(teks):
    if(type(teks) != str):
            teks = str(teks)
    cleaned_text = re.sub(r'[^\w\s]', '', teks)
    cleaned_text = re.sub(r'\d+', '', cleaned_text)
    return cleaned_text

def casefolding(teks):
    lower_case = teks.lower()
    return lower_case

def tokenization (teks):
        pisah = teks.split()
        return pisah

def stopwordremoval (teks):
    removed = [token for token in teks if token not in listStopword]
    return removed

def preprocessing(kalimat):
    cleaned_text = cleansing(teks=kalimat)
    lower_case = casefolding(teks=cleaned_text)
    pisah = tokenization(teks=lower_case)
    stopword = stopwordremoval(teks=pisah)
    return stopword

def tokens_to_vectors(texts):
    vectors = []
    embedding_dim = 300
    for text in texts:
        tokens = text
        text_vectors = [model.wv[token] if token in model.wv else np.zeros(embedding_dim) for token in tokens]
        if(text_vectors):
            text_vectors = np.vstack(text_vectors)
            vectors.append(np.mean(text_vectors, axis=0)) 
        else:
            vectors.append(np.zeros(embedding_dim))
    return np.array(vectors)

def train() :
    df = pd.read_csv('deepfake_database.csv', sep=';')
    texts = df['Tweet']
    label = df['Sentimen']
    text_embeddings = tokens_to_vectors([preprocessing(kalimat=text) for text in texts]) 
    train_texts, test_texts, train_labels, test_labels = train_test_split(text_embeddings, label, test_size=0.2)
    scaler = MinMaxScaler()
    train_texts = scaler.fit_transform(train_texts)
    test_texts = scaler.transform(test_texts)
    rf = RandomForestClassifier(random_state=42, class_weight='balanced_subsample', criterion="entropy")
    rf.fit(train_texts, train_labels)
    rf_test = rf.predict(test_texts)
    cnb = ComplementNB()
    cnb.fit(train_texts, train_labels)
    cnb_test = cnb.predict(test_texts)
    print(classification_report(test_labels, rf_test))
    print(classification_report(test_labels, cnb_test))
    eclf = VotingClassifier(estimators=[('rf', rf), ('cnb', cnb)], voting='soft')
    eclf.fit(train_texts, train_labels)
    eclf_test = eclf.predict(test_texts)
    print(classification_report(test_labels, eclf_test))

def main ():
    train()

if __name__ == "__main__":
    main()