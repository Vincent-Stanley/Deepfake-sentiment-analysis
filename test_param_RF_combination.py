import re
from nltk.corpus import stopwords
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from mpstemmer import MPStemmer
from gensim.models import FastText
from sklearn.model_selection import StratifiedKFold

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
max_depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , None]
min_samples_leafs = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
min_samples_splits = [2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]

def cleansing(teks):
    if(type(teks) != str):
            teks = str(teks)
    cleaned_text = re.sub(r'@\w+', '', teks)
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
    cleaned_text = re.sub(r'\d+', '', cleaned_text)
    return cleaned_text

def casefolding(teks):
    lower_case = teks.lower()
    return lower_case

def remove_elongation(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1", text)

def tokenization (teks):
    pisah = teks.split()
    return pisah

def stopwordremoval (teks):
    removed = [token for token in teks if token not in listStopword]
    return removed

def removeSlang(teks):
    new_words = [slang_dict[word] if word in slang_dict else word for word in teks]
    return new_words

def stemming (teks):
     return [stemmer.stem(token) for token in teks]

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
        text_vectors = [model.wv[token] if token in model.wv else np.zeros(model.get_dimension()) for token in tokens]
        if(text_vectors):
            text_vectors = np.vstack(text_vectors)
            vectors.append(np.mean(text_vectors, axis=0)) 
        else:
            vectors.append(np.zeros(model.get_dimension()))
    return np.array(vectors)

def train() :
    df = pd.read_csv('New_Deepfake_DatabaseNew2024.csv', sep=';')
    texts = df['Tweet']
    label = df['Sentiment']
    text_embeddings = [preprocessing(kalimat=text) for text in texts]
    train_texts, test_texts, train_labels, test_labels = train_test_split(text_embeddings, label, stratify=label, test_size=0.2)
    model = FastText(vector_size=300, window=5, min_count=1) 
    model.build_vocab(corpus_iterable=train_texts)
    model.train(corpus_iterable=train_texts, total_examples=len(train_texts), epochs=75)
    train_texts = tokens_to_vectors(texts = train_texts, model = model)
    test_texts = tokens_to_vectors(texts = test_texts, model = model)
    scaler = MinMaxScaler()
    train_texts = scaler.fit_transform(train_texts)
    test_texts = scaler.transform(test_texts)
    kf = StratifiedKFold(n_splits=5,random_state=42,shuffle=True)
    for depth in max_depths:
        for leaf in min_samples_leafs:
            for split in min_samples_splits:
                all_train_log_losses = []
                all_test_log_losses = []
                for i, (train_index, test_index) in enumerate(kf.split(train_texts, train_labels)):
                    data_train = train_texts[train_index]
                    target_train = train_labels.iloc[train_index]

                    data_validation = train_texts[test_index]
                    target_validation = train_labels.iloc[test_index]

                    clf1 = RandomForestClassifier(random_state=42, max_depth=depth, min_samples_leaf=leaf, min_samples_split=split)
                            
                    clf1.fit(data_train, target_train)

                    train_preds = clf1.predict_proba(data_train)
                    validation_preds = clf1.predict_proba(data_validation)

                    train_log_loss = log_loss(target_train, train_preds)
                    validation_log_loss = log_loss(target_validation, validation_preds)

                    all_train_log_losses.append(train_log_loss)
                    all_test_log_losses.append(validation_log_loss)
                print(f'Max Depth: {depth}, Min Samples Leaf: {leaf}, Min Samples Split: {split}')
                print(f'Training Log Loss: {np.mean(all_train_log_losses)}')
                print(f'Validation Log Loss: {np.mean(all_test_log_losses)}')

def main ():
    train()

if __name__ == "__main__":
    main()

