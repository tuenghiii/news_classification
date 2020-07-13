from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

app = Flask(__name__)


count_vect = joblib.load('../count_vect.pkl') 
trans_vect = joblib.load('../trans_vect.pkl') 
classifier = joblib.load('../classifier.pkl') 


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        corpus = request.form['news']
        predicted_news_type = classify_news(corpus)
        return render_template('result.html', prediction=predicted_news_type)


def classify_news(news):
    """
    Nhận bài báo và trả về kết quả phân loại theo chủ đề
    
    params
    ------
    news: (str) Bài báo
    
    return
    ------
    
    return
    ------
    type_pred: (str) Kết quả phân loại, là một trong 5 chủ đề: 
    "Công Nghệ", "Giáo Dục", "Sức Khỏe", "Thể Thao", "Âm Nhạc"
    """
    if len(news) == 0:
        return None
    news            = np.array([news])
    news_tf_vect    = count_vect.transform(news)
    news_tfidf_vect = trans_vect.transform(news_tf_vect).toarray()
    type_preds      = classifier.predict(news_tfidf_vect)
    type_pred       = type_preds[0]
    return type_pred


if __name__ == '__main__':
    app.run("localhost", "9999", debug=True)