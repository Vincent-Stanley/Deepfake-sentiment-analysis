import re
from nltk.corpus import stopwords
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from mpstemmer import MPStemmer
from gensim.models import FastText

stemmer = MPStemmer()
listStopword = set(stopwords.words('indonesian'))
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
slangs = 'colloquial-indonesian-lexicon.csv'
df1 = pd.read_csv(slangs, sep=',')
formal = df1['formal']
slang = df1['slang']
slang_dict = dict(zip(slang, formal))

def cleansing(teks):
    if(type(teks) != str):
            teks = str(teks)
    cleaned_text = re.sub(r'@\w+', '', teks) #Menghapus semua @nama dalam teks
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text) #Menghapus semua tanda baca dan karakter khusus, kecuali karakter kata dan spasi
    cleaned_text = re.sub(r'\d+', '', cleaned_text) #Menghapus semua angka dari teks.
    return cleaned_text

def casefolding(teks):
    lower_case = teks.lower() #Mengubah huruf dalam teks menjadi huruf kecil.
    return lower_case

def remove_elongation(text):
    pattern = re.compile(r"(.)\1{2,}") #Karakter yang diulang tiga atau lebih kali berturut-turut.
    return pattern.sub(r"\1", text) #Mengganti karakter yang berulang menjadi satu karakter.

def tokenization (teks):
    pisah = teks.split() #Memisahkan teks menjadi satuan kata
    return pisah

def stopwordremoval (teks):
    removed = [token for token in teks if token not in listStopword] # Menghapus kata-kata dalam daftar stopword dari teks.
    return removed

def removeSlang(teks):
    new_words = [slang_dict[word] if word in slang_dict else word for word in teks] #Mengganti kata slang dengan bentuk formalnya menggunakan kamus slang 'colloquial-indonesian-lexicon.csv'.
    return new_words

def stemming (teks):
     return [stemmer.stem(token) for token in teks] #Mengubah kata berimbuhan menjadi kata dasar dari MPStemmer()

def preprocessing(kalimat):
    cleaned_text = cleansing(teks=kalimat)
    lower_case = casefolding(teks=cleaned_text)
    rem_elong = remove_elongation(text=lower_case)
    pisah = tokenization(teks=rem_elong)
    stopword = stopwordremoval(teks=pisah)
    rem_slang = removeSlang(teks=stopword)
    stem = stemming(teks=rem_slang)
    return stem

def tokens_to_vectors(texts, model):
    vectors = []
    for text in texts:
        tokens = text
        text_vectors = [model.wv[token] if token in model.wv else np.zeros(model.get_dimension()) for token in tokens] # Mengubah setiap token menjadi vektor dari kamus FastText. Jika token tidak ditemukan dalam kamus, diganti dengan vektor nol.
        if(text_vectors):
            text_vectors = np.vstack(text_vectors) # Menggabungkan semua vektor token menjadi satu matriks.
            vectors.append(np.mean(text_vectors, axis=0)) # Menghitung rata-rata dari semua vektor token untuk menghasilkan satu vektor per teks
        else:
            vectors.append(np.zeros(model.get_dimension()))
    return np.array(vectors)

def train() :
    df = pd.read_csv('New_Deepfake_DatabaseNew2024.csv', sep=';')
    texts = df['Tweet']
    label = df['Sentiment']
    text_embeddings = [preprocessing(kalimat=text) for text in texts] #Melakukan proses preprocessing dan hasil disimpan di dalam list
    train_texts, test_texts, train_labels, test_labels = train_test_split(text_embeddings, label, stratify=label, test_size=0.2) # memisahkan data latih : data uji 80 : 20
    model = FastText(vector_size=300, window=5, min_count=1) 
    model.build_vocab(corpus_iterable=train_texts)
    model.train(corpus_iterable=train_texts, total_examples=len(train_texts), epochs=75) #Melatih model FastText dengan corpus dari teks latih
    train_texts = tokens_to_vectors(texts = train_texts, model = model) 
    test_texts = tokens_to_vectors(texts = test_texts, model = model) #Melakukan vektorisasi satuan kata dalam data latih dan data uji
    scaler = MinMaxScaler()
    train_texts = scaler.fit_transform(train_texts)
    test_texts = scaler.transform(test_texts) #mengubah hasil vektorisasi data sehingga setiap vektor di antara 0 sampai 1 sehingga CNB dapat menerima
    rf = RandomForestClassifier(random_state=42, max_depth= 8, min_samples_leaf= 50, min_samples_split= 100, class_weight = None, criterion='entropy', n_estimators= 500)
    rf.fit(train_texts, train_labels)
    rf_test = rf.predict(test_texts)
    rf_cm = confusion_matrix(test_labels, rf_test)
    print(classification_report(test_labels, rf_test))
    sns.heatmap(rf_cm, annot=True,fmt='d', cmap='YlGnBu', xticklabels=rf.classes_, yticklabels=rf.classes_) #Menunjukkan evaluasi model dalam bentuk confusion matrix
    plt.xlabel('Predicted Sentiment',fontsize=12)
    plt.ylabel('Labelled Sentiment',fontsize=12)
    plt.title('RF Confusion Matrix',fontsize=16)
    plt.show()
    print('\n')
    cnb = ComplementNB()
    cnb.fit(train_texts, train_labels)
    cnb_test = cnb.predict(test_texts)
    cnb_cm = confusion_matrix(test_labels, cnb_test)
    print(classification_report(test_labels, cnb_test))
    sns.heatmap(cnb_cm, annot=True,fmt='d', cmap='YlGnBu', xticklabels=cnb.classes_, yticklabels=cnb.classes_)
    plt.xlabel('Predicted Sentiment',fontsize=12)
    plt.ylabel('Labelled Sentiment',fontsize=12)
    plt.title('CNB Confusion Matrix',fontsize=16)
    plt.show()
    eclf = VotingClassifier(estimators=[('rf', rf), ('cnb', cnb)], voting='soft')
    eclf.fit(train_texts, train_labels)
    eclf_test = eclf.predict(test_texts)
    eclf_cm = confusion_matrix(test_labels, eclf_test)
    print(classification_report(test_labels, eclf_test))
    sns.heatmap(eclf_cm, annot=True,fmt='d', cmap='YlGnBu', xticklabels=eclf.classes_, yticklabels=eclf.classes_)
    plt.xlabel('Predicted Sentiment',fontsize=12)
    plt.ylabel('Labelled Sentiment',fontsize=12)
    plt.title('Ensemble Confusion Matrix',fontsize=16)
    plt.show()

def main ():
    train()

if __name__ == "__main__":
    main()