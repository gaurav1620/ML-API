from flask import make_response ,send_file,Flask ,redirect, url_for, request, render_template
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from flask_cors import CORS, cross_origin

app = Flask(__name__) 
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
import pandas as pd

@app.route("/") 
@cross_origin()
def home_view():
    return render_template("home.html")

@app.route("/getpred",methods=['POST','GET'])
@cross_origin()
def givePred():
    
    if(request.method == 'POST'):
        train = pd.read_csv(request.files['train'])
        test = pd.read_csv(request.files['test'])
        
        cols_train = train.columns
        cols_test = test.columns

        to_predict = list(set(cols_train)-set(cols_test))[0]
        print("Predicting : ",to_predict)
        
        X = train.drop(to_predict,axis = 1)
        y = pd.DataFrame(train[to_predict])
        
        #list of columns to drop 
        drops = []
        convert = []
        for colname in X.columns:
            if(X[colname].dtype.name == 'object'):
                if(len(X[colname].unique()) <= 2):
                    convert.append(colname)
                else:
                    drops.append(colname)
            elif(X[colname].isna().sum()/X[colname].shape[0]) > 0.2:
                drops.append(colname)
        
        print("Drops : ")
        print(drops)
        
        print("Conv : ")
        print(convert)

        for colname in drops:
            X.drop(colname,axis = 1,inplace=True)
            test.drop(colname,axis = 1,inplace=True)

        X = X.apply(lambda row: row.fillna(row.mode()[0]), axis=1)
        test = test.apply(lambda row: row.fillna(row.mode()[0]), axis=1)

        for colname in convert:
            type1 = X[colname].unique()[0]
            X[colname] = [int(1) if type1 == i else int(0) for i in X[colname]]
            test[colname] = [int(1) if type1 == i else int(0) for i in test[colname]]
        
        for colname in X.columns:
            X[colname] = pd.to_numeric(X[colname], errors='coerce')
            test[colname] = pd.to_numeric(test[colname], errors='coerce')
        
        '''
        print("INFO X ")
        X.info()
        print("DESC X ")
        X.describe()
        print("INFO X ")
        X.info()
        print("DESC X ")
        X.describe()
        X.info()
        y.info()
        test.info()
        '''
        
        model_type = request.form['options']
        if model_type == 'logistic':
            model = LogisticRegression()
        elif model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'svm':
            model = svm.SVC()
        elif model_type == 'dtree':
            model = DecisionTreeClassifier()


        model.fit(X,y)
        pred = model.predict(test)
        pred = pd.DataFrame(pred)
        print(pred)
        
        #Returns a file with all the training data as well as the predictions
        #Also NaN, Null values are filled
        test[to_predict] = pred
        pred = test
        resp = make_response(pred.to_csv(index=False))
        resp.headers["Content-Disposition"] = "attachment; filename=export.csv"
        resp.headers["Content-Type"] = "text/csv"
        return resp


if __name__ == "__main__":
    app.run()
