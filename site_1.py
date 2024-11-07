from flask import Flask, request, render_template, jsonify
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from gensim.models.keyedvectors import KeyedVectors
from keras.models import load_model
import keras.backend as K

nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)

def sent2word(x):
    stop_words = set(stopwords.words('english')) 
    x = re.sub("[^A-Za-z]", " ", x)
    x = x.lower()
    filtered_sentence = []
    words = x.split()
    for w in words:
        if w not in stop_words: 
            filtered_sentence.append(w)
    return filtered_sentence

def makeVec(words, model, num_features):
    vec = np.zeros((num_features,), dtype="float32")
    noOfWords = 0
    index2word_set = set(model.index_to_key)  
    for word in words:
        if word in index2word_set:
            noOfWords += 1
            vec = np.add(vec, model[word])
    if noOfWords > 0:
        vec = np.divide(vec, noOfWords)
    return vec

def getVecs(essays, model, num_features):
    essay_vecs = np.zeros((len(essays), num_features), dtype="float32")
    for idx, essay in enumerate(essays):
        essay_vecs[idx] = makeVec(essay, model, num_features)
    return essay_vecs

def convertToVec(text):
    num_features = 300
    model = KeyedVectors.load_word2vec_format("word2vecmodel.bin", binary=True)
    clean_test_essays = []
    clean_test_essays.append(sent2word(text))
    testDataVecs = getVecs(clean_test_essays, model, num_features)
    testDataVecs = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))

    lstm_model = load_model("final_lstm.h5")
    preds = lstm_model.predict(testDataVecs)
    K.clear_session() 
    return str(round(preds[0][0]))

@app.route('/', methods=['GET', 'POST'])
def create_task():
    if request.method == 'POST':
        data = request.get_json()
        final_text = data.get("text", "")
        if len(final_text) > 20: 
            score = convertToVec(final_text)
        else:
            score = "0"
        return jsonify({'score': score}), 201
    else:
        return render_template('mainpage.html')

if __name__ == '__main__':
    app.run(debug=True)
