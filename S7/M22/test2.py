import request
response = request.get('https://api.github.com')
print(response.status_code)
