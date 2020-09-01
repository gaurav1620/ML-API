from flask import Flask, make_response
from flask_restful import Api, Resource, abort, reqparse
import random
import werkzeug
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

app = Flask(__name__)
api = Api(app)

class foo(Resource):
    def get(self):
        return {"foo" : "bar"}

# Basic args 
base_args = reqparse.RequestParser()
base_args.add_argument("train", type=werkzeug.datastructures.FileStorage,location='files',required=True, help="Need a training file (.csv)")
base_args.add_argument("test", type=werkzeug.datastructures.FileStorage,location='files',required=True, help="Need a testing file (.csv)")

def get_files(args):
    return pd.read_csv(args['train']), pd.read_csv(args['test'])

class logisticRegression(Resource):
    def get(self):
        args = base_args.parse_args()
        print(args)
        try :
            train = pd.read_csv(args['train'])
            test = pd.read_csv(args['test'])
        except :
            return {'message' : 'File Error'},422
        
        #######################################
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
        
        model = LogisticRegression()
        model.fit(X,y)
        pred = model.predict(test)
        pred = pd.DataFrame(pred)
        print(pred)
        
        #Returns a file with all the training data as well as the predictions
        #Also NaN, Null values are filled
        test[to_predict] = pred
        pred = test
        #######################################

        pred.to_csv(r'response_files/index.py',index=False)

        # save the file in a directory provide a link to it
        # finally make a method to retrivr these files from links
        return {'link':'http://127.0.0.1:5000/share/index.py'},200

class linearRegression(Resource):
    def get(self):
        args = base_args.parse_args()
        print(args)
        try :
            train,train = get_files(args)
        except :
            return {'message' : 'File Error'},422

        #######################################
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
        
        model = LinearRegression()
        model.fit(X,y)
        pred = model.predict(test)
        pred = pd.DataFrame(pred)
        print(pred)
        
        #Returns a file with all the training data as well as the predictions
        #Also NaN, Null values are filled
        test[to_predict] = pred
        pred = test
        #######################################

        pred.to_csv(r'response_files/index.py',index=False)

        # save the file in a directory provide a link to it
        # finally make a method to retrivr these files from links
        return {'link' : 'http://127.0.0.1:5000/share/index.py'}
        #return resp

class returnFiles(Resource):
    def get(self,filename):
        file = open('response_files/'+filename)
        df = pd.read_csv(file)
        print(df)
        resp = make_response(df.to_csv(index=False))
        resp.headers["Content-Disposition"] = ("attachment; filename=export.csv")
        resp.headers["Content-Type"] = "text/csv"
        return resp


api.add_resource(foo, "/foo/")
api.add_resource(logisticRegression, "/logi")
api.add_resource(linearRegression, "/line")

api.add_resource(returnFiles, "/share/<string:filename>")

if __name__ == "__main__":
    app.run(debug = True)
