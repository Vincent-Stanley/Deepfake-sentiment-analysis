let db;

function openDatabase() {
    return new Promise((resolve, reject) => {
        let request = indexedDB.open('MyDatabase', 1);

        request.onerror = function(event) {
            console.error('Database error:', event.target.errorCode);
            reject(event.target.errorCode);
        };

        request.onsuccess = function(event) {
            db = event.target.result;
            console.log('Database opened successfully');
            resolve(db);
        };

        request.onupgradeneeded = function(event) {
            db = event.target.result;
            if (!db.objectStoreNames.contains('MyObjectStore')) {
                db.createObjectStore('MyObjectStore', { keyPath: 'id' });
            }
        };
    });
}

function addData(storeName, data) {
    return new Promise((resolve, reject) => {
        let transaction = db.transaction([storeName], 'readwrite');
        let objectStore = transaction.objectStore(storeName);
        let request = objectStore.put(data);

        request.onsuccess = function(event) {
            console.log('Data added successfully');
            resolve(event.target.result);
        };

        request.onerror = function(event) {
            console.error('Error adding data:', event.target.errorCode);
            reject(event.target.errorCode);
        };
    });
}

function getData(storeName, id) {
    return new Promise((resolve, reject) => {
        let transaction = db.transaction([storeName]);
        let objectStore = transaction.objectStore(storeName);
        let request = objectStore.get(id);

        request.onsuccess = function(event) {
            resolve(event.target.result);
        };

        request.onerror = function(event) {
            console.error('Error retrieving data:', event.target.errorCode);
            reject(event.target.errorCode);
        };
    });
}



document.getElementById('circle1').classList.add('active');
document.getElementById('substepper3').style.display = 'none';
contents = ['CollectTable', 'PreprocessData', 'WordEmbedding', 'Split', 'SentimentAnalysis', 'Visualization']
subcontents = ['substepContent3-1','substepContent3-2','substepContent3-3','substepContent3-4','substepContent3-5', 'substepContent3-6', 'substepContent3-7', 'substepContent3-8']
function show(step) {
    document.querySelectorAll('.content > div').forEach(div => div.style.display = 'none');
    
    document.getElementById(step).style.display = 'block';

    document.querySelectorAll('.step .circle').forEach(circle => circle.classList.remove('active'));
    const stepIndex = contents.indexOf(step);
    if (stepIndex !== -1) {
        document.getElementById(`circle${stepIndex + 2}`).classList.add('active');
    } else {
        console.error(`Step '${step}' not found in contents array.`);
    }

    document.getElementById('substepper3').style.display = (step === 'PreprocessData') ? 'block' : 'none';
}

function showSubstep(substep) {
    document.querySelectorAll('#PreprocessData > div').forEach(div => div.style.display = 'none');

    document.getElementById(substep).style.display = 'block';

    document.querySelectorAll('.substep .circle').forEach(circle => circle.classList.remove('active'));
    const substepIndex = subcontents.indexOf(substep);
    if (substepIndex !== -1) {
        document.getElementById(`subcircle3-${substepIndex + 1}`).classList.add('active');
    } else {
        console.error(`Step '${step}' not found in contents array.`);
    }
}

document.getElementById('start_cleansing').addEventListener('click', async function() {
    this.style.display = 'none';
    
    await startCleansing();
});
document.getElementById('start_casefolding').addEventListener('click', async function() {
    await startCaseFolding();
});

document.getElementById('start_remove_elongation').addEventListener('click', async function() {
    await startRemoveElongation();
});

document.getElementById('start_tokenization').addEventListener('click', async function() {
    await startTokenization();
});

document.getElementById('start_stopword').addEventListener('click', async function() {
    await startStopword();
});

document.getElementById('start_remove_slang').addEventListener('click', async function() {
    await startRemoveSlang();
});

document.getElementById('start_stemming').addEventListener('click', async function() {
    await startStemming();
});

document.getElementById('final_result').addEventListener('click', async function() {
    await showResult();
});

document.getElementById('train_fast').addEventListener('click', async function() {
    this.style.display = 'none';
    await trainFast();
    const start = document.getElementById('start_FastText');
    start.style.display = 'block';
});

document.getElementById('start_FastText').addEventListener('click', async function() {
    this.style.display = 'none';
    const showing = document.getElementById('vectorization_result');
    showing.style.display = 'block';
    await startWordEmbedding();
});

document.getElementById('trainRFButton').addEventListener('click', function() {
    trainModel('Random Forest');
});

document.getElementById('trainCNBButton').addEventListener('click', function() {
    trainModel('Complement Naive Bayes');
});

document.getElementById('trainELButton').addEventListener('click', function() {
    trainModel('Ensemble Method');
});

document.getElementById('analisis').addEventListener('click', function() {
    AnalisisSentimen();
    this.style.display = 'none';
    const prediksi = document.getElementById('prediksi');
    prediksi.style.display = 'block';
});

document.getElementById('exportExcel').addEventListener('click', function() {
    exportToExcel();
});


document.getElementById('fetchDatasetButton').addEventListener('click', function() {
    fetch(`http://127.0.0.1:5000/dataset?table=test`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        openDatabase().then(() => {
            addData('MyObjectStore', { id: 'dataset', value: data });
        });
        const tableBody = document.getElementById('datasetTableBody');
        tableBody.innerHTML = ''; 
        data.forEach(item => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td style="text-align: center">${item.id}</td>
                <td style="text-align: center">${item.username}</td>
                <td><a href="${item.url}" target="_blank">${item.url}</a></td>
                <td>${item.tweet}</td>
                <td style="text-align: center">${item.date}</td>
                <td style="text-align: center">${item.sentimen}</td>
            `;
            tableBody.appendChild(row);
        });
        document.getElementById('datasetTable').style.display = 'table';
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while fetching the dataset.');
    });
});

document.getElementById('splittingData').addEventListener('click', async function(){
    const spliceValue = document.getElementById('splice').value;
    openDatabase().then(() => {
        addData('MyObjectStore', { id: 'splice', value: spliceValue});
    });
    const data = await splitData(spliceValue);
    const train_data = data.train_data;
    const test_data = data.test_data;
    openDatabase().then(() => {
        addData('MyObjectStore', { id: 'train_data', value: train_data });
        addData('MyObjectStore', { id: 'test_data', value: test_data });
    });
    this.style.display = 'none';
    const table = document.getElementById('table_test');
    table.style.display = 'block';
    const tableBody = document.getElementById('datasetTableBody4');
    const train = document.getElementById('training');
    tableBody.innerHTML = '';
    for (const item of test_data) {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${item.id}</td>
            <td>${item.username}</td>
            <td><a href="${item.url}" target="_blank">${item.url}</a></td>
            <td>${item.tweet}</td>
            <td>${item.cleansed_tweet}</td>
            <td>${item.casefolded_tweet}</td>
            <td>${item.remove_elongation_tweet}</td>
            <td>${item.tokenized_tweet}</td>
            <td>${item.stopword_removal_tweet}</td>
            <td>${item.remove_slang_tweet}</td>
            <td>${item.stemming_tweet}</td>
            <td>${item.vectorized_tweet}</td>
            <td>${item.date}</td>
            <td style="text-align: center">${item.sentimen}</td>
        `;
        tableBody.appendChild(row);
    }
    document.getElementById('datasetTable4').style.display = 'table';
    train.style.display = 'block';
});

async function splitData(split) {
    try {
        const db = await openDatabase();
        const storedData = await getData('MyObjectStore', 'dataset');
        if (storedData) {
            const data = storedData.value
            const response = await fetch('http://127.0.0.1:5000/split', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({datas: data, splice: parseInt(split)})
            });
            const result = await response.json();
            return result;
        } else {
            console.error('Error: Data not found in IndexedDB');
            alert("Error: Data not found in IndexedDB");
        }
    } catch (error) {
        console.error('Error:', error);
        alert("Error processing training data.");
    }
}

async function startCleansing() {
    try {
        const db = await openDatabase();
        const storedData = await getData('MyObjectStore', 'dataset');

        if (storedData) {
            const data = storedData.value;
            const tableBody = document.getElementById('datasetTableBody2-1');
            tableBody.innerHTML = '';

            // Perform cleansing for each item
            for (const item of data) {
                const cleanedText = await fetchProcessedText(item.tweet, 'cleansing');
                item['cleansed_tweet'] = cleanedText;
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${item.id}</td>
                    <td>${item.username}</td>
                    <td><a href="${item.url}" target="_blank">${item.url}</a></td>
                    <td>${cleanedText}</td>
                    <td>${item.date}</td>
                    <td style="text-align: center">${item.sentimen}</td>
                `;
                tableBody.appendChild(row);
            }
            await addData('MyObjectStore', { id: 'dataset', value: data });
            document.getElementById('datasetTable2-1').style.display = 'table';
            document.getElementById('start_casefolding').style.display = 'block';
        } else {
            console.error('Error: Data not found in IndexedDB');
        }
    } catch (error) {
        console.error('Error processing cleansing:', error);
    }
}

async function startCaseFolding() {
    try {
        const db = await openDatabase();
        const storedData = await getData('MyObjectStore', 'dataset');

        if (storedData) {
            const data = storedData.value;
            const tableBody = document.getElementById('datasetTableBody2-2');
            tableBody.innerHTML = '';

            // Perform casefolding using cleansed data
            for (const item of data) {
                const caseFoldedText = await fetchProcessedText(item.cleansed_tweet, 'casefolding');
                item['casefolded_tweet'] = caseFoldedText;
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${item.id}</td>
                    <td>${item.username}</td>
                    <td><a href="${item.url}" target="_blank">${item.url}</a></td>
                    <td>${caseFoldedText}</td>
                    <td>${item.date}</td>
                    <td style="text-align: center">${item.sentimen}</td>
                `;
                tableBody.appendChild(row);
            }
            await addData('MyObjectStore', { id: 'dataset', value: data });
            document.getElementById('datasetTable2-2').style.display = 'table';
            document.getElementById('start_remove_elongation').style.display = 'block';
        } else {
            console.error('Error: Data not found in IndexedDB');
        }
    } catch (error) {
        console.error('Error processing cleansing:', error);
    }
}

async function startRemoveElongation() {
    try {
        const db = await openDatabase();
        const storedData = await getData('MyObjectStore', 'dataset');

        if (storedData) {
            const data = storedData.value;
            const tableBody = document.getElementById('datasetTableBody2-3');
            tableBody.innerHTML = '';

            // Perform casefolding using cleansed data
            for (const item of data) {
                const removeElongationText = await fetchProcessedText(item.casefolded_tweet, 'remove_elongation');
                item['remove_elongation_tweet'] = removeElongationText;
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${item.id}</td>
                    <td>${item.username}</td>
                    <td><a href="${item.url}" target="_blank">${item.url}</a></td>
                    <td>${removeElongationText}</td>
                    <td>${item.date}</td>
                    <td style="text-align: center">${item.sentimen}</td>
                `;
                tableBody.appendChild(row);
            }
            await addData('MyObjectStore', { id: 'dataset', value: data });
            document.getElementById('datasetTable2-3').style.display = 'table';
            document.getElementById('start_tokenization').style.display = 'block';
        } else {
            console.error('Error: Data not found in IndexedDB');
        }
    } catch (error) {
        console.error('Error processing cleansing:', error);
    }
}

async function startTokenization() {
    try {
        const db = await openDatabase();
        const storedData = await getData('MyObjectStore', 'dataset');

        if (storedData) {
            const data = storedData.value;
            const tableBody = document.getElementById('datasetTableBody2-4');
            tableBody.innerHTML = '';
            for (const item of data) {
                const tokenizedText = await fetchProcessedText(item.remove_elongation_tweet, 'tokenization');
                item['tokenized_tweet'] = tokenizedText;
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${item.id}</td>
                    <td>${item.username}</td>
                    <td><a href="${item.url}" target="_blank">${item.url}</a></td>
                    <td>${tokenizedText}</td>
                    <td>${item.date}</td>
                    <td style="text-align: center">${item.sentimen}</td>
                `;
                tableBody.appendChild(row);
            }
            await addData('MyObjectStore', { id: 'dataset', value: data });
            document.getElementById('datasetTable2-4').style.display = 'table';
            document.getElementById('start_stopword').style.display = 'block';
        } else {
            console.error('Error: Data not found in IndexedDB');
        }
    } catch (error) {
        console.error('Error processing cleansing:', error);
    }
}

async function startStopword() {
    try {
        const db = await openDatabase();
        const storedData = await getData('MyObjectStore', 'dataset');
        if (storedData) {
            const data = storedData.value;
            const tableBody = document.getElementById('datasetTableBody2-5');
            tableBody.innerHTML = ''
            for (const item of data) {
                const stopwordText = await fetchProcessedText(item.tokenized_tweet, 'stopwordremoval');
                item['stopword_removal_tweet'] = stopwordText;
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${item.id}</td>
                    <td>${item.username}</td>
                    <td><a href="${item.url}" target="_blank">${item.url}</a></td>
                    <td>${stopwordText}</td>
                    <td>${item.date}</td>
                    <td style="text-align: center">${item.sentimen}</td>
                `;
                tableBody.appendChild(row);
            }
            await addData('MyObjectStore', { id: 'dataset', value: data });
            document.getElementById('datasetTable2-5').style.display = 'table';
            document.getElementById('start_remove_slang').style.display = 'block';
        } else {
            console.error('Error: Data not found in IndexedDB');
        }
    } catch (error) {
        console.error('Error processing cleansing:', error);
    }
}

async function startRemoveSlang() {
    try {
        const db = await openDatabase();
        const storedData = await getData('MyObjectStore', 'dataset');

        if (storedData) {
            const data = storedData.value;
            const tableBody = document.getElementById('datasetTableBody2-6');
            tableBody.innerHTML = ''
            for (const item of data) {
                const removeSlangText = await fetchProcessedText(item.stopword_removal_tweet, 'remove_slang');
                item['remove_slang_tweet'] = removeSlangText;
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${item.id}</td>
                    <td>${item.username}</td>
                    <td><a href="${item.url}" target="_blank">${item.url}</a></td>
                    <td>${removeSlangText}</td>
                    <td>${item.date}</td>
                    <td style="text-align: center">${item.sentimen}</td>
                `;
                tableBody.appendChild(row);
            }
            await addData('MyObjectStore', { id: 'dataset', value: data });
            document.getElementById('datasetTable2-6').style.display = 'table';
            document.getElementById('start_stemming').style.display = 'block';
        } else {
            console.error('Error: Data not found in IndexedDB');
        }
    } catch (error) {
        console.error('Error processing cleansing:', error);
    }
}

async function startStemming() {
    try {
        const db = await openDatabase();
        const storedData = await getData('MyObjectStore', 'dataset');

        if (storedData) {
            const data = storedData.value;
            const tableBody = document.getElementById('datasetTableBody2-7');
            tableBody.innerHTML = ''
            for (const item of data) {
                const stemmingText = await fetchProcessedText(item.remove_slang_tweet, 'stemming');
                item['stemming_tweet'] = stemmingText;
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${item.id}</td>
                    <td>${item.username}</td>
                    <td><a href="${item.url}" target="_blank">${item.url}</a></td>
                    <td>${stemmingText}</td>
                    <td>${item.date}</td>
                    <td style="text-align: center">${item.sentimen}</td>
                `;
                tableBody.appendChild(row);
            }
            await addData('MyObjectStore', { id: 'dataset', value: data });
            document.getElementById('datasetTable2-7').style.display = 'table';
            document.getElementById('final_result').style.display = 'block';
        } else {
            console.error('Error: Data not found in IndexedDB');
        }
    } catch (error) {
        console.error('Error processing cleansing:', error);
    }
}

async function showResult() {
    try {
        const db = await openDatabase();
        const storedData = await getData('MyObjectStore', 'dataset');

        if (storedData) {
            const data = storedData.value;
            const tableBody = document.getElementById('datasetTableBody2-8');
            tableBody.innerHTML = '';
            for (const item of data) {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${item.id}</td>
                    <td>${item.username}</td>
                    <td><a href="${item.url}" target="_blank">${item.url}</a></td>
                    <td>${item.tweet}</td>
                    <td>${item.cleansed_tweet}</td>
                    <td>${item.casefolded_tweet}</td>
                    <td>${item.remove_elongation_tweet}</td>
                    <td>${item.tokenized_tweet}</td>
                    <td>${item.stopword_removal_tweet}</td>
                    <td>${item.remove_slang_tweet}</td>
                    <td>${item.stemming_tweet}</td>
                    <td>${item.date}</td>
                    <td style="text-align: center">${item.sentimen}</td>
                `;
                tableBody.appendChild(row);
            }
            document.getElementById('datasetTable2-8').style.display = 'table';
        } else {
            console.error('Error: Data not found in IndexedDB');
        }
    } catch (error) {
        console.error('Error processing cleansing:', error);
    }
}

async function fetchProcessedText(tweet, step) {
    try {
        const response = await fetch('http://127.0.0.1:5000/preprocess', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: tweet, step: step })
        });
        const result = await response.json();
        return result.text;
    } catch (error) {
        console.error(`Error fetching ${step}:`, error);
        return 'Unknown';
    }
}

async function trainFast() {
    document.getElementById('loadingScreen').style.display = 'flex';
    try {
        const db = await openDatabase();
        const storedData = await getData('MyObjectStore', 'dataset');
        if (storedData) {
            const data = storedData.value;
            const response = await fetch('http://127.0.0.1:5000/trainfasttext', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({datas: data})
            });
            const result = await response.json();
            document.getElementById('loadingScreen').style.display = 'none';
            return result.data;
        } else {
            console.error('Error: Data not found in IndexedDB');
            alert("Error: Data not found in IndexedDB");
        }
    } catch (error) {
        console.error('Error:', error);
        alert("Error processing training model.");
    }
}

async function startWordEmbedding() {
    try {
        const db = await openDatabase();
        const storedData = await getData('MyObjectStore', 'dataset');

        if (storedData) {
            const data = storedData.value;
            const tableBody = document.getElementById('datasetTableBody3');
            tableBody.innerHTML = '';
            for (const item of data) {
                const vectorizedtext = await fetchVectorizedText(item.stemming_tweet);
                item['vectorized_tweet'] = vectorizedtext;
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${item.id}</td>
                    <td>${item.username}</td>
                    <td><a href="${item.url}" target="_blank">${item.url}</a></td>
                    <td>${item.tweet}</td>
                    <td>${item.cleansed_tweet}</td>
                    <td>${item.casefolded_tweet}</td>
                    <td>${item.remove_elongation_tweet}</td>
                    <td>${item.tokenized_tweet}</td>
                    <td>${item.stopword_removal_tweet}</td>
                    <td>${item.remove_slang_tweet}</td>
                    <td>${item.stemming_tweet}</td>
                    <td>${vectorizedtext}</td>
                    <td>${item.date}</td>
                    <td style="text-align: center">${item.sentimen}</td>
                `;
                tableBody.appendChild(row);
            }
            await addData('MyObjectStore', { id: 'dataset', value: data });
            document.getElementById('datasetTable3').style.display = 'table';
            document.getElementById('startSplit').style.display = 'block';
        } else {
            console.error('Error: Data not found in IndexedDB');
        }
    } catch (error) {
        console.error('Error processing cleansing:', error);
    }
}

async function trainModel(method){
    document.getElementById('loadingScreen').style.display = 'flex';
    openDatabase().then(() => {
        addData('MyObjectStore', { id: 'method', value: method});
    });
    const spliceData = await getData('MyObjectStore', 'splice');
    const metodeData = await getData('MyObjectStore', 'method');
    const splice = spliceData.value;
    const metode = metodeData.value;
    document.getElementById('methods').innerHTML += metode + ' dengan pemisahan data latih dan data uji : ' + (100-parseInt(splice)  + ' : ' + splice)
    try {
        const db = await openDatabase();
        const storedData = await getData('MyObjectStore', 'train_data');
        if (storedData) {
            const data = storedData.value;
            const response = await fetch('http://127.0.0.1:5000/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({datas: data, methods : method})
            });
            const result = await response.json();
            document.getElementById('loadingScreen').style.display = 'none';
            return result.data;
        } else {
            console.error('Error: Data not found in IndexedDB');
            alert("Error: Data not found in IndexedDB");
        }
    } catch (error) {
        console.error('Error:', error);
        alert("Error processing training data.");
    }
}

async function AnalisisSentimen(){
    try {
        document.getElementById('loadingScreen').style.display = 'flex';
        const db = await openDatabase();
        const storedData = await getData('MyObjectStore', 'test_data');
        if (storedData) {
            const data = storedData.value;
            const tableBody3 = document.getElementById('datasetTableBody5');
            tableBody3.innerHTML = '';
            for (const item of data) {
                const sentiment = await fetchSentiment(item.vectorized_tweet);
                document.getElementById('loadingScreen').style.display = 'none';
                item['predicted_sentiment'] = sentiment;
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td style="text-align: center">${item.id}</td>
                    <td style="text-align: center">${item.username}</td>
                    <td><a href="${item.url}" target="_blank">${item.url}</a></td>
                    <td>${item.tweet}</td>
                    <td>${item.cleansed_tweet}</td>
                    <td>${item.casefolded_tweet}</td>
                    <td>${item.remove_elongation_tweet}</td>
                    <td>${item.tokenized_tweet}</td>
                    <td>${item.stopword_removal_tweet}</td>
                    <td>${item.remove_slang_tweet}</td>
                    <td>${item.stemming_tweet}</td>
                    <td>${item.vectorized_tweet}</td>
                    <td style="text-align: center">${item.date}</td>
                    <td style="text-align: center">${item.sentimen}</td>
                    <td style="text-align: center">${item.predicted_sentiment}</td>
                `;
                tableBody3.appendChild(row);
            }
            await addData('MyObjectStore', { id: 'test_data', value: data });
            document.getElementById('datasetTable5').style.display = 'table';
            document.getElementById('buttons2').style.display = 'flex';
        } else {
            console.error('Error: Data not found in IndexedDB');
        }
    } catch (error) {
        console.error('Error processing cleansing:', error);
    }
}

async function fetchVectorizedText(tweet) {
    try {
        const response = await fetch(`http://127.0.0.1:5000/vectorization`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: tweet })
        });
        const result = await response.json();
        return result.vectorize;
    } catch (error) {
        console.error('Error fetching sentiment:', error);
        return 'Unknown';
        }
}

async function fetchSentiment(tweet) {
    try {
        const response = await fetch(`http://127.0.0.1:5000/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: tweet })
        });
        const result = await response.json();
        return result.sentiment;
    } catch (error) {
        console.error('Error fetching sentiment:', error);
        return 'Unknown';
        }
}

document.getElementById('showAccuracy').addEventListener('click', async function() {
    fetchEvaluate();
});

async function fetchEvaluate () {
    try {
        const db = await openDatabase();
        const storedData = await getData('MyObjectStore', 'test_data');
        if (storedData) {
            const data = storedData.value;
            const result = await showEvaluation(data);
            const confMatrix = result.confusion_matrix;
            console.log("Confusion matrix received:", confMatrix);
            const classes = result.classes;
            displayConfusionMatrix(confMatrix, classes);
            const classReport = result.classification_report;
            displayClassificationReport(classReport);
        } else {
            console.error('Error: Data not found in IndexedDB');
        }
    } catch (error) {
        console.error('Error processing cleansing:', error);
    }
}
    

document.getElementById('backButton').addEventListener('click', function() {
    showSlide('tableSlide');
});

function showSlide(slideId) {
    const slides = document.querySelectorAll('.slide');
    slides.forEach(slide => {
        slide.classList.remove('active');
    });
    document.getElementById(slideId).classList.add('active');
}

function displayConfusionMatrix(confMatrix, classes) {
    const zmax = Math.max(...confMatrix.map(row => Math.max(...row)));
    let annotations = [];
    var trace = {
        z: confMatrix,
        x: classes,
        y: classes,
        type: 'heatmap',
        colorscale: 'YlGnBu',
        showscale: true,
        zmin: 0, 
        zmax: zmax, 
        annotations: annotations,
    };
    var layout = {
        title: 'Confusion Matrix',
        xaxis: { title: 'Prediksi Sentimen' },
        yaxis: { title: 'Sentimen Terlabel' },
        annotations: []
    };

    for (let i = 0; i < confMatrix.length; i++) {
        for (let j = 0; j < confMatrix[i].length; j++) {
            layout.annotations.push({
                x: j, 
                y: i, 
                text: confMatrix[i][j], 
                font: {
                    color: 'black', 
                    size: 12 
                },
                showarrow: false
            });
        }
    }

    Plotly.newPlot('heatmap', [trace], layout);
    function displayConfusionMatrix2(confMatrix, classes) {
        let actualTotals = [];
        let predictedTotals = [];
        
        for (let i = 0; i < classes.length; i++) {
            const trueCount = confMatrix[i].reduce((acc, val) => acc + val, 0);
            const predCount = confMatrix.reduce((acc, val) => acc + val[i], 0);

            actualTotals.push(trueCount);
            predictedTotals.push(predCount);
        }

        const ctx = document.getElementById('pieChart').getContext('2d');
        new Chart(ctx, {
            type: 'pie',
            data: {
                labels: classes, 
                datasets: [
                    {
                        label: 'Sentimen Terlabel',
                        data: actualTotals,
                        backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56'],
                    },
                    {
                        label: 'Prediksi Sentimen',
                        data: predictedTotals,
                        backgroundColor: ['#FF9AA2', '#B5EAD7', '#FFDAC1'],
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            generateLabels: function(chart) {
                                const datasets = chart.data.datasets;
                                const labels = chart.data.labels;
                                let legendLabels = [];
        
                                datasets.forEach((dataset, datasetIndex) => {
                                    labels.forEach((label, labelIndex) => {
                                        legendLabels.push({
                                            text: `${dataset.label} - ${label}`,
                                            fillStyle: dataset.backgroundColor[labelIndex],
                                            hidden: !chart.isDatasetVisible(datasetIndex),
                                            datasetIndex: datasetIndex,
                                            index: labelIndex
                                        });
                                    });
                                });
        
                                return legendLabels;
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.dataset.label || '';
                                const dataLabel = context.label || '';
                                const value = context.raw || '';
                                return `${label} - ${dataLabel}: ${value}`;
                            }
                        }
                    }
                }
            }
        });
        const statsContainer = document.getElementById('statsContainer');
        const table = document.createElement('table');
        table.classList.add('stats-table');
        const thead = document.createElement('thead');
        const tbody = document.createElement('tbody');
        
        // Create table headers
        const headerRow = document.createElement('tr');
        headerRow.innerHTML = `
            <th>Class</th>
            <th>Sentimen Terlabel</th>
            <th>Prediksi Sentimen</th>
        `;
        thead.appendChild(headerRow);

        for (let i = 0; i < classes.length; i++) {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${classes[i]}</td>
                <td>${actualTotals[i]}</td>
                <td>${predictedTotals[i]}</td>
            `;
            tbody.appendChild(row);
        }

        table.appendChild(thead);
        table.appendChild(tbody);
        statsContainer.innerHTML = ''; 
        statsContainer.appendChild(table);
    }
    displayConfusionMatrix2(confMatrix, classes);
}

async function exportToExcel() {
    try {
        const db = await openDatabase();
        const storedData = await getData('MyObjectStore', 'test_data');
        if (storedData) {
            const data = storedData.value
            const response = await fetch(`http://127.0.0.1:5000/download`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(({
                    datas: data,
                }))
            });
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const result = await response.json();
            if (result.success) {
                window.location.href = result.file_url;

                alert('Excel file has been successfully exported!');
            } else {
                console.error('Error exporting to Excel:', result.error);
                alert('Error exporting to Excel: ' + result.error);
            }
        } else {
            console.error('Error: Data not found in IndexedDB');
            alert("Error: Data not found in IndexedDB");
        }
    } catch (error) {
        console.error('Error exporting to Excel:', error);
    }
}

async function showEvaluation(data) {
    try {
        const response = await fetch('http://127.0.0.1:5000/evaluate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                datas: data,
            })
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const result = await response.json();
        return result;
    } catch (error) {
        console.error('Error sending data to the server:', error);
        throw error;
    }
}

function displayClassificationReport(classReport) {
    const tableBody = document.getElementById('classificationReportBody');
    tableBody.innerHTML = '';
    const row = document.createElement('tr');

    const order = ['negatif', 'netral', 'positif', 'accuracy', 'macro avg', 'weighted avg'];

    order.forEach(label => {
        if (classReport[label]) {
            const metrics = classReport[label];
            const row = document.createElement('tr');
            if (label === 'accuracy') {
                row.innerHTML = `
                    <td style="text-align: center">${label}</td>
                    <td colspan="4">${metrics.toFixed(2)}</td>
                `;
            } else {
                row.innerHTML = `
                    <td style="text-align: center">${label}</td>
                    <td style="text-align: center">${metrics.precision.toFixed(2)}</td>
                    <td style="text-align: center">${metrics.recall.toFixed(2)}</td>
                    <td style="text-align: center">${metrics['f1-score'].toFixed(2)}</td>
                    <td style="text-align: center">${metrics.support}</td>
                `;
            }
            tableBody.appendChild(row);
        }
    });

    document.getElementById('classificationReport').style.display = 'table';
}
