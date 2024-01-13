from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin
from model import optimization_model

# envName: optimizationEnv

app = Flask(__name__)

CORS(app)


@app.route("/", methods=['POST', 'GET'])
@cross_origin()
def indexRoute():
    return "abcabc"

@app.route("/optimization", methods=['POST'])
@cross_origin()
def chatRoute():
    return optimization_model(request.json['orders'], request.json['rawMaterials'],  request.json['algorithmKey'], request.json['operationsResearchMethodKey'], request.json['regressionModelKey'])

if __name__ == '__main__':
    # Debug modunu etkinleştirme
    app.run(debug=True)

