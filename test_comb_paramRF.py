import re
from nltk.corpus import stopwords
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
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

rf_hyperparams_grid={ #parameter yang akan dilakukan grid search
    "n_estimators":[50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    "criterion": ["gini", "entropy"],
    "class_weight": ['balanced', 'balanced_subsample', None],
}

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
    kf = StratifiedKFold(n_splits=5,random_state=42,shuffle=True)
    rf = RandomForestClassifier(random_state=42, max_depth= 8, min_samples_leaf= 50, min_samples_split= 100)
    rf_grid_search=GridSearchCV(  #melakukan grid search dengan negative log loss sebagai nilai evaluasi model dan stratefied k-fold sebagai pemisahan data dari cross validarion
        estimator=rf,
        param_grid=rf_hyperparams_grid,
        scoring="neg_log_loss",
        refit=True,
        return_train_score=True,
        cv=kf,
        verbose=10,
        n_jobs=-1,
    )
    tuned_grid_model = rf_grid_search.fit(train_texts, train_labels)
    print(tuned_grid_model.best_params_)
    best_params_predict = tuned_grid_model.predict_proba(test_texts)
    print("The log loss of the model with Grid Search is: " + str(log_loss(test_labels, best_params_predict)))
        
def main ():
    train()

if __name__ == "__main__":
    main()