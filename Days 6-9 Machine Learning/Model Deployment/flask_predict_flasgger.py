import pickle
from flask import Flask, request
from flasgger import Swagger
import numpy as np
import pandas as pd

with open('rf.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)
swagger = Swagger(app)

@app.route('/predict')
def predict_iris():
    """Example endpoint returning a prediction of iris
    ---
    parameters:
      - name: s_length
        in: query
        type: number
        required: true
      - name: s_width
        in: query
        type: number
        required: true
      - name: p_length
        in: query
        type: number
        required: true
      - name: p_width
        in: query
        type: number
        required: true
    """
    s_length = request.args.get("s_length")
    s_width = request.args.get("s_width")
    p_length = request.args.get("p_length")
    p_width = request.args.get("p_width")
    
    prediction = model.predict(np.array([[s_length, s_width, p_length, p_width]]))
    
    if prediction==0:
        pred_class = 'iris-verscicolor'
    elif prediction == 1:
        pred_class = 'iris-setosa'
    else:
        pred_class = 'iris-virginica'
        
    return pred_class

    return pred_class

@app.route('/predict_file', methods=["POST"])
def predict_iris_file():
    """Example file endpoint returning a prediction of iris
    ---
    parameters:
      - name: input_file
        in: formData
        type: file
        required: true
    """
    input_data = pd.read_csv(request.files.get("input_file"), header=None)
    prediction = model.predict(input_data)
    pred_classes = []
    for pred in list(prediction):
        if pred==0:
            pred_classes.append('verscicolor')
        elif pred==1:
            pred_classes.append('setosa')
        else: pred_classes.append('virginica')
        
    return str(pred_classes)


if __name__ == '__main__':
    app.run()
    
   