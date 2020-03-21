import os, shutil
from lsh import group, preprocessing
from flask import Flask, render_template, request, session, url_for, redirect
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, patch_request_class, IMAGES
from werkzeug.utils import secure_filename
from save_photos import save
from webbrowser import open
import functools

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = "key"
app.config['UPLOAD_FOLDER'] = "uploads/new_photos"
app.config.update(
    UPLOADED_PATH=app.config['UPLOAD_FOLDER'],
    DROPZONE_ALLOWED_FILE_TYPE='image',
    DROPZONE_MAX_FILE_SIZE=3,
    #DROPZONE_MAX_FILES=100,
    DROPZONE_PARALLEL_UPLOADS=100,
    DROPZONE_UPLOAD_MULTIPLE=True,
)
dropzone = Dropzone(app)

# Calls save_photos.py
@app.route('/open_save_google_photos', methods=['POST', 'GET'])
def open_save_google_photos():
    if request.method == 'POST':
        google_photo_urls = save()
        preprocessing(google_photo_urls)
    return render_template('index.html')

# Saves query images to session['query_filenames'] (there's no button associated with this)
# Called by the dropzone stuff
@app.route('/', methods=['POST', 'GET'])
def upload():
    if "query_filenames" not in session:
        session['query_filenames'] = []
    filenames = session['query_filenames']
    if request.method == 'POST':
        # Saves query images to session['query_filenames']
        for key, f in request.files.items():
            if key.startswith('file'):  
                query_image_path = os.path.join(app.config['UPLOADED_PATH'], f.filename)
                f.save(query_image_path)
                filenames.append(query_image_path)
        session['query_filenames'] = filenames
    return render_template('index.html')

# Button associated with "Group"
@app.route('/query', methods=['POST', 'GET'])
def query():
    if "query_filenames" not in session or session['query_filenames'] == []:
        return redirect(url_for('upload'))
    filenames = session['query_filenames']
    # List of lists
    groups = group(filenames)
    session.pop('query_filenames', None)
    return render_template('results.html', groups = groups)

if __name__ == '__main__':
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    open('http://localhost:5555') # opens up two tabs of 5555 for some reason?
    app.run(HOST, PORT, debug = True)