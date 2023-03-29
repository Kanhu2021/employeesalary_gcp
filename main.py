import numpy as np
import pandas as pd
from flask import Flask,render_template,jsonify,request
import pickle

app = Flask(__name__)
model = pickle.load(open('regmodel.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    print(np.array(list(data.values())).reshape(1,-1))
    data_reshape = np.array(list(data.values())).reshape(1,-1)
    output = model.predict(data_reshape)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    return render_template('home.html', prediction_text='Employee Salary should be $ {}'.format(output))



if __name__ == "__main__":
    app.run(debug=True)