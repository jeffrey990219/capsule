import os, shutil
#from lsh import group, preprocessing
from Flann import preprocessing, group
from flask import Flask, render_template, request, session, url_for, redirect
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, patch_request_class, IMAGES
from werkzeug.utils import secure_filename
from save_photos import save
from webbrowser import open
import functools
from skimage import io
import cv2
from glob import glob

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = "key"
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'google_photos')
app.config.update(
    UPLOADED_PATH=os.path.join('static', 'new_photos'),
    DROPZONE_ALLOWED_FILE_TYPE='image',
    DROPZONE_MAX_FILE_SIZE=3,
    #DROPZONE_MAX_FILES=100,
    DROPZONE_PARALLEL_UPLOADS=100,
    DROPZONE_UPLOAD_MULTIPLE=True,
)
dropzone = Dropzone(app)
loading = False

# Calls save_photos.py
@app.route('/open_save_google_photos', methods=['POST', 'GET'])
def open_save_google_photos():
    loading = True
    if request.method == 'POST':
        save()
        preprocessing(glob("static/google_photos/*.jpg"))
    loading = False
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
    results = []
    for i, g in enumerate(groups):
        new_res = []
        for item in g:
            print(os.path.basename(item))
            new_res.append(os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(item)))
        results.append(new_res)
    print(results)
    return render_template('results.html', queries = filenames, groups = results)

if __name__ == '__main__':
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT, debug = True)