from flask import Flask, jsonify, render_template
import csv

app = Flask(__name__)

csv_file_path = 'touch_events.csv'

@app.route('/data', methods=['GET'])
def get_touch_events():
    data = []
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return jsonify(data)

@app.route('/')
def index():
    return render_template('index.html')

app.run(debug=True)