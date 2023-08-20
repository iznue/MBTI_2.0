import requests

pred = 'T'

while True:
    url = 'http://192.168.35.145:5556/test/question_3'
    response = requests.post(url, json=pred, timeout= 10)