from flask import Flask, send_from_directory, render_template_string
import os

app = Flask(__name__)

@app.route('/')
def index():
    image_folder = '/app/output_results'
    images = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    images.sort()  # Sort images by name
    return render_template_string('''
        <!doctype html>
        <title>AI Training Visualization</title>
        <h1>AI Training Visualization</h1>
        <div>
            {% for image in images %}
                <img src="{{ url_for('image', filename=image) }}" style="width: 100%; max-width: 600px; margin-bottom: 20px;">
            {% endfor %}
        </div>
    ''', images=images)

@app.route('/images/<filename>')
def image(filename):
    return send_from_directory('/app/output_results', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)