#flask app
from flask import Flask, request, jsonify
from inference import generate_mix
import base64

app = Flask(__name__)

@app.route('/generate', methods=['GET'])
def generate():
    # should serve this requests.get("http://81.16.3.115:8989/generate", params={"genres": genres})
    genres = request.args.get('genres')
    weights = request.args.get('weights')
    genres = genres.split(',')
    if weights:
        weights = weights.split(',')
    else:
        weights = []
    result_file_path = generate_mix(genres, weights)
    # senf file contentto client
    print("F", result_file_path)
    with open(result_file_path, 'rb') as f:
        result = f.read()
    encoded_content = base64.b64encode(result).decode('utf-8')
    return jsonify({'result': encoded_content})





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8989)