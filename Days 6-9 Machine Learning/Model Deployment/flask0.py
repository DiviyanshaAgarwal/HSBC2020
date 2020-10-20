from flask import Flask
app = Flask(__name__)

@app.route('/add')
def add():
    a = 10
    b = 20
    return str(a+b)
    
@app.route('/')
def hello():
    s1 = "<H1> 'Hello WOrld!' </H1>"
    return s1

if __name__=='__main__':
    app.run()