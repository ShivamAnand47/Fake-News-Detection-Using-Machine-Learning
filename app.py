import numpy as np
from flask import Flask, request,render_template
from flask_cors import CORS
from feature import *
import os
import pickle
import flask
from newspaper import Article
import urllib
import nltk
from sklearn.metrics import accuracy_score, classification_report
nltk.download('punkt')

#loading flask and assigning the model variable
app = Flask(__name__)
CORS(app)
app=flask.Flask(__name__,template_folder='templates')

with open('Clfpac.pkl','rb') as handle:
	Clfpac= pickle.load(handle)
    
@app.route('/')
def main():
	return render_template('index.html')

#receiving the input url from the user and using web scrapping to extract  the news content
@app.route('/predict',methods=['GET','POST']) 
def predict():
    url =request.get_data(as_text=True)[5:]
    url =urllib.parse.unquote(url) 
    article= Article(str(url))
    article.download()
    article.parse()
    article.nlp()
    article_author= article.authors
    article_date= article.publish_date
    news_text = article.text
    news = article.summary
    getting_input =remove_punctuation_lemma(news)
    print(article.summary)
    print(getting_input)
    #passing the news article to the model and returning whether it is fake or real
    
    pred = Clfpac.predict([getting_input])
    probability_real= Clfpac.predict_proba([getting_input])[:,0]
    probability_fake= Clfpac.predict_proba([getting_input])[:,1]
    print(probability_real)
    print(probability_fake)
    #print({np.round(max(probability[0])*100,2)})
    print(f'Accuracy real: {np.round(probability_real*100,2)}%')
    print(f'Accuracy fake: {np.round(probability_fake*100,2)}%')
    real_percentage = f' {np.round(probability_real*100,2)}%'
    fake_percentage = f' {np.round(probability_fake*100,2)}%'
    print(real_percentage)
   
    #decision_func = modele.decision_function([getting_input])
    #score =_predict_proba_lr(decision_func)
    #print(f'Accuracy: {np.round((score[0])*100,2)}%')
   
    if(pred[0]==0):         
                  return render_template('index.html',prediction_text='The news is "{}",with {} Accuracy'.format("REAL",real_percentage),original=news_text,processing= getting_input,author=article_author,date=article_date,fake=fake_percentage,real=real_percentage)
    else:
        return render_template('index.html',prediction_text='The news is "{}",with {} Accuracy'.format("FAKE",fake_percentage),original=news_text,processing= getting_input,author=article_author,date=article_date,fake=fake_percentage,real=real_percentage)
if __name__=="__main__":
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)
