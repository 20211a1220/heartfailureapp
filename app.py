from flask import Flask,request,jsonify
import pickle 
import numpy as np
model=pickle.load(open('venv/model2.pkl','rb'))
app=Flask(__name__)
@app.route('/')
def home():
    return "SARLE PADKO IGA"
@app.route('/predict',methods=['POST'])
def predict():
    age=request.form.get('age')
    sex=request.form.get('sex')
    cp=request.form.get('cp')
    trtbps=request.form.get('trtbps')
    chol=request.form.get('chol')
    fbs=request.form.get('fbs')
    restecg=request.form.get('restecg')
    thalachh=request.form.get('thalachh')
    exng=request.form.get('exng')
    oldpeak=request.form.get('oldpeak')
    slp=request.form.get('slp')
    caa=request.form.get('caa')
    thall=request.form.get('thall')
    input_query=np.array([[age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall]],dtype=float)
    
    result= model.predict(input_query)[0]
    return jsonify({'chances yes':str(result)})


    
if __name__=='__main__':
    app.run(debug=True)