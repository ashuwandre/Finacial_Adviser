from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn 

app = FastAPI()

@app.get('/')
def  home():
    return {'messgae': ' welcom '}

@app.get('/predict')
def predict():
    return {'predict':'predict '}

class Inputdata(BaseModel):
    name : 'str'
    
@app.post('/name')
def name(data : Inputdata):
    name = data.name
    return {'name':name}
     
    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    
 