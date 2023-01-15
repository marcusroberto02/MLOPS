import request
response = request.get('https://api.github.com/this-api-should-not-exist')
print(response.status_code)
