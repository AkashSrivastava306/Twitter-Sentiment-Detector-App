# Importing Flask lib

from flask import Flask, render_template, url_for, request
import pickle
import pandas as pd
import joblib



filename = open('pickle.pkl', 'rb')
clf = joblib.load(filename)
test = open('transform.pkl', 'rb')
cv = joblib.load(test)
app = Flask(__name__, template_folder='template')
@app.route('/')

def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        test_vector = cv.transform(data)
        my_prediction = clf.predict(test_vector)
    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
