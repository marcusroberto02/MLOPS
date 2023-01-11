import requests

#request = requests.post("http://127.0.0.1:8000/login/",params={"username": "test", "password": "test"})
#request = requests.post("http://127.0.0.1:8000/text_model/",json={"email": "test", "domain_match": "gmail"})

file = ('files', open('img.jpg', 'rb'))
request = requests.post("http://127.0.0.1:8000/cv_model/",files=[file,file])

print(request.status_code)
print(request.content)
