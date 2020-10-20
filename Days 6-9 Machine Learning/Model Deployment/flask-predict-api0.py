import pickle
from flask import Flask, request
import numpy as np

with open('rf.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Please use the /predict route to access the model'

@app.route('/predict')
def predict_iris():

    s_length = request.args.get("sl")
    s_width = request.args.get("sw")
    p_length = request.args.get("pl")
    p_width = request.args.get("pw")
    
#    s_length = request.form["s_length"]
#    s_width = request.form["s_width"]
#    p_length = request.form["p_length"]
#    p_width = request.form["p_width"]
    
    prediction = model.predict(np.array([[s_length, s_width, \
                                          p_length, p_width]]))
    # return str(prediction)

    if prediction==0:
        pred_class = 'iris-verscicolor'
    elif prediction == 1:
        pred_class = 'iris-setosa'
    else:
        pred_class = 'iris-virginica'
        
    return pred_class

if __name__ == '__main__':
    app.run()   
    