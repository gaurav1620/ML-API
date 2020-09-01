import requests

BASE_URI = "http://127.0.0.1:5000/"
files = {}
files['train'] = open('titanic_data/train.csv')
files['test'] = open('titanic_data/test.csv')

resp = requests.get(BASE_URI + "logi", files=files)
print(resp.json())
