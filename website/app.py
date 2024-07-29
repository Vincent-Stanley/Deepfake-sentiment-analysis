from flask import Flask, request, jsonify, send_file, url_for
import pandas as pd
from flask_cors import CORS
import re
import numpy as np
import mysql.connector
from nltk.corpus import stopwords
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from mpstemmer import MPStemmer
import os
from gensim.models.fasttext import FastText
stemmer = MPStemmer()

app = Flask(__name__)
CORS(app) 
random_forest_classifier = None
cnb_classifier = None
ensemble_classifier = None
metode = None
imputer = SimpleImputer(strategy='mean')
scaler = MinMaxScaler()
model = None
listStopword = set(stopwords.words('indonesian'))
stemmer = MPStemmer()
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


@app.route('/')
def home():
    return "Sentiment Analysis API is running."

@app.route('/preprocess', methods=['POST'])
def preprocess_data():
    data = request.get_json()
    text = data['text']
    step = data['step']

    if step == 'cleansing':
        processed_text = cleansing(text)
    elif step == 'casefolding':
        processed_text = casefolding(text)
    elif step == 'remove_elongation':
        processed_text = remove_elongation(text)
    elif step == 'tokenization':
        processed_text = tokenization(text)
    elif step == 'stopwordremoval':
        processed_text = stopwordremoval(text)
    elif step == 'remove_slang':
        processed_text = removeSlang(text)
    elif step == 'stemming':
        processed_text = stemming(text)
    else:
        return jsonify({'error': 'Invalid step'}), 400

    return jsonify({'text': processed_text})

class ModelContainer:
    def __init__(self):
        self.random_forest_classifier = None
        self.cnb_classifier = None
        self.ensemble_classifier = None
        self.scaler = MinMaxScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.metode = None
        self.model = None

models = ModelContainer()

@app.route('/split', methods=['POST'])
def split_data():
    global random_forest_classifier, cnb_classifier, ensemble_classifier, scaler, imputer
    all = request.get_json() 
    data = all['datas']
    split = all['splice'] / 100.0
    data_with_indices = list(enumerate(data))
    data_train, data_test = train_test_split(data_with_indices, test_size=split)
    train_indices, data_train = zip(*data_train)
    test_indices, data_test = zip(*data_test)
    sorted_data_train = [data[i] for i in sorted(train_indices)]
    sorted_data_test = [data[i] for i in sorted(test_indices)]
    return jsonify({'train_data': sorted_data_train, 'test_data': sorted_data_test})

@app.route('/train', methods=['POST'])
def train_datas():
    global random_forest_classifier, cnb_classifier, ensemble_classifier, scaler, imputer, metode
    all = request.get_json()
    data_train = all['datas']
    method = all['methods']
    models.metode = method   
    train_data = np.vstack([item['vectorized_tweet'] for item in data_train])
    train_label = [item['sentimen'] for item in data_train]
    vectorized_text = models.scaler.fit_transform(train_data)
    if (method == 'Random Forest'):
        models.random_forest_classifier = RandomForestClassifier(random_state=42, max_depth = 8, min_samples_leaf= 50, min_samples_split=100, n_estimators= 500, criterion="entropy", class_weight=None)
        models.random_forest_classifier.fit(vectorized_text, train_label)
    elif(method == 'Complement Naive Bayes'):
        models.cnb_classifier = ComplementNB()
        models.cnb_classifier.fit(vectorized_text, train_label)
    elif(method == 'Ensemble Method'):
        models.random_forest_classifier = RandomForestClassifier(random_state=42, max_depth = 8, min_samples_leaf= 50, min_samples_split=100, n_estimators= 500, criterion="entropy", class_weight=None)
        models.random_forest_classifier.fit(vectorized_text, train_label)
        models.cnb_classifier = ComplementNB()
        models.cnb_classifier.fit(vectorized_text, train_label)
        models.ensemble_classifier = VotingClassifier(estimators=[('rf', models.random_forest_classifier), ('cnb', models.cnb_classifier)], voting='soft')
        models.ensemble_classifier.fit(vectorized_text, train_label)
    return jsonify({'message': 'Model trained successfully'}), 200


@app.route('/trainfasttext', methods=['POST'])
def train_fastText():
    data = request.get_json()
    all = data['datas']
    processed_text = [item['stemming_tweet'] for item in all]
    models.model = FastText(vector_size=300, window=1, min_count=1) 
    models.model.build_vocab(corpus_iterable=processed_text)
    models.model.train(corpus_iterable=processed_text, total_examples=len(data), epochs=75) 
    return jsonify({'message': 'Model trained successfully'}), 200

@app.route('/vectorization', methods=['POST'])
def vectorizing_text():
    data = request.get_json()
    processed_text = data['text']
    vectorize_text = tokens_to_vectors([processed_text], models.model)
    vectorize_text_list = vectorize_text.tolist()
    return jsonify({'vectorize': vectorize_text_list})

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    global random_forest_classifier, cnb_classifier, ensemble_classifier, metode
    data = request.get_json()
    vectorized_text = data['text']
    vectorized_text = models.scaler.transform(vectorized_text)
    if models.metode == 'Random Forest':
        prediction = models.random_forest_classifier.predict(vectorized_text)
    elif models.metode == 'Complement Naive Bayes':
        prediction = models.cnb_classifier.predict(vectorized_text)
    elif models.metode == 'Ensemble Method':
        prediction = models.ensemble_classifier.predict(vectorized_text)
    else:
        return jsonify({'error': 'Invalid method'}), 400
    sentiment = prediction[0]
    if (type(sentiment) != str):
        sentiment = str(sentiment)
    return jsonify({'sentiment': sentiment})

@app.route('/evaluate', methods=['POST'])
def evalute_model() :
    global random_forest_classifier, cnb_classifier, ensemble_classifier, metode
    all = request.get_json()
    data = all['datas']
    label = [item['sentimen'] for item in data]
    predicted_sentiment = [item['predicted_sentiment'] for item in data]
    conf_matrix = confusion_matrix(label, predicted_sentiment)
    conf_matrix = conf_matrix.tolist() 
    if models.metode == 'Random Forest':
        classes = models.random_forest_classifier.classes_
    elif models.metode == 'Complement Naive Bayes':
        classes = models.cnb_classifier.classes_
    elif models.metode == 'Ensemble Method':
        classes = models.ensemble_classifier.classes_
    else:
        return jsonify({'error': 'Invalid method'}), 400
    target_names = ['negatif', 'netral', 'positif']
    class_report = classification_report(label, predicted_sentiment, target_names=target_names, output_dict=True)
    classes = classes.tolist()
    return jsonify({'confusion_matrix': conf_matrix, 'classification_report': class_report, 'classes': classes})

@app.route('/download', methods=['POST'])
def download_excel():
    all = request.get_json()
    data = all['datas']
    columns_order = ['id', 'username', 'url', 'tweet', 'cleansed_tweet', 'casefolded_tweet', 
                     'remove_elongation_tweet', 'tokenized_tweet', 'stopword_removal_tweet', 'remove_slang_tweet',
                     'stemming_tweet', 'vectorized_tweet', 'date', 'sentimen', 'predicted_sentiment'] 
    df = pd.DataFrame(data, columns=columns_order)
    excel_file = 'predictions.xlsx'
    df.to_excel(excel_file, index=False)
    return jsonify({'success': True, 'file_url': url_for('export_excel', filename=excel_file, _external=True)})

@app.route('/export', methods=['GET'])
def export_excel():
    filename = request.args.get('filename')
    if not filename or not os.path.exists(filename):
        return jsonify({'error': 'File not found'}), 404
    return send_file(filename, as_attachment=True)

@app.route('/dataset', methods=['GET'])
def get_dataset():
    conn = mysql.connector.connect(
        host='127.0.0.1',
        user='root',
        password='Gareth55_1818',
        database='deepfake_db'
    )
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, url, tweet, date, sentimen FROM test")
    rows = cursor.fetchall()
    conn.close()

    dataset = []
    for row in rows:
        dataset.append({
            'id': row[0],
            'username': row[1],
            'url': row[2],
            'tweet': row[3],
            'date': row[4],
            'sentimen': row[5],
        })
    
    return jsonify(dataset)

if __name__ == '__main__':
    app.run(debug=True)