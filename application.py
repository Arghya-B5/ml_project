from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import predict_pipelines,Custom_Data


#app=False(__name__)
application=Flask(__name__)
app = application

#rout for a app

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template("home.html")
    else:
        data=Custom_Data(
            gender=request.form.get('gender'),
            race_ethicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('ethnicity'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )

        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Befor prcdiction")
        predict_r=predict_pipelines()
        print("mid prediction")

        results=predict_r.predics_model(pred_df)
        print("after prediction")

        return render_template("home.html",results=results[0])
#,host="0.0.0.0"

if __name__=="__main__":
    app.run(host="0.0.0.0")