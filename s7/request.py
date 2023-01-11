import requests
import json

#response=requests.get("https://api.github.com/repos/SkafteNicki/dtu_mlops")

#print(response.json())

"""
response2 = requests.get(
   'https://api.github.com/search/repositories',
   params={'q': 'requests+language:python'},
)

print(response2.json())

with open('test.txt', 'w') as f:
  json.dump(response2.json(), f, ensure_ascii=False)


response = requests.get('https://hatrabbits.com/wp-content/uploads/2017/01/random.jpg')

#print(response.json())

with open(r'img.jpg','wb') as f:
   f.write(response.content)

"""

pload = {'username':'Olivia','password':'123'}
response = requests.post('https://httpbin.org/post', data = pload)

print(response.json())