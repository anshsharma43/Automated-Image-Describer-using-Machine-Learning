import os
import uuid
import requests
from whitenoise import WhiteNoise

from flask import (Flask, flash, redirect, render_template, request,
                   send_from_directory, url_for)

import model_test
from PIL import Image



UPLOAD_FOLDER = './static/images/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.wsgi_app = WhiteNoise(app.wsgi_app, root='static/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.after_request
def set_response_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


import requests
from io import BytesIO

# main directory of programme
@app.route('/', methods=['GET', 'POST'])


def upload_file():
    try:
        # remove files created more than 5 minute ago
        os.system("find static/images/ -maxdepth 1 -mmin +5 -type f -delete")
    except OSError:
        pass
    
    if request.method == 'POST':      
        content_file = request.files['content-file']
        image_url = request.form['image-url']
        print(image_url)
        print(content_file) 
        
        if 'content-file' in request.files and request.files['content-file'].filename != '':
             content_file = request.files['content-file']
             files = [content_file]
             content_name = str(uuid.uuid4()) + ".png"
             file_names = [content_name]
             for i, file in enumerate(files):
                if file and allowed_file(file.filename):
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_names[i]))
             path = "static/images/" + file_names[0]
             caption = model_test.generate_caption(image_path=path)
             params = {
        'content': "/static/images/" + file_names[0],
        'caption': caption,
        
    }


        elif 'image-url' in request.form:
            print("Bye")
            image_url = request.form['image-url']
            try:
                response = requests.get(image_url)
                image = Image.open(BytesIO(response.content))
                content_name = str(uuid.uuid4()) + ".png"
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], content_name)
                image.save(image_path)
                caption = model_test.generate_caption(image_path=image_path)
                params = {
                    'content': "/static/images/" + content_name,
                    'caption': caption,
                    
                }
            except:
                flash('Failed to fetch image from URL')
                return redirect(request.url)
            
        return render_template('success.html', **params)
        
    return render_template('upload.html')




@app.errorhandler(404)
def page_not_found(error):
    return render_template('page_not_found.html'), 404


if __name__ == "__main__":
    app.run(host='0.0.0.0')

