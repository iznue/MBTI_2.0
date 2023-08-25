import requests

pred = 'T'

# while True:
#     url = 'http://192.168.0.232:5556/test/get_device_PC2'
#     response = requests.post(url, json=pred, timeout= 10)

url = 'http://192.168.0.232:5556/test/get_device_PC2'
response = requests.post(url, json=pred, timeout= 10)