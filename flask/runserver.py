import os, shutil
from lsh import group
from flask import Flask, render_template, request, session, url_for, redirect
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, patch_request_class, IMAGES
from werkzeug.utils import secure_filename
from save_photos import save
from webbrowser import open
import functools
print = functools.partial(print, flush=True)

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

@app.route('/open_save_google_photos', methods=['POST', 'GET'])
def open_save_google_photos():
    if request.method == 'POST':
        temp = save()
        print(temp, flush = True)
    return render_template('index.html')

# Saves query images to session['file_urls']
@app.route('/', methods=['POST', 'GET'])
def upload():
    if "file_urls" not in session:
        session['file_urls'] = []
    filenames = session['file_urls']
    if request.method == 'POST':
        # Saves query images to session['file_urls']
        for key, f in request.files.items():
            if key.startswith('file'):
                query_image_path = os.path.join(app.config['UPLOADED_PATH'], f.filename)
                f.save(query_image_path)
                filenames.append(query_image_path)
                print(f)
        session['file_urls'] = filenames
    return render_template('index.html')

@app.route('/query', methods=['POST', 'GET'])
def query():
    if "file_urls" not in session or session['file_urls'] == []:
        return redirect(url_for('upload'))
    filenames = session['file_urls']
    # List of lists
    groups = group(filenames)
    session.pop('file_urls', None)
    return render_template('results.html', groups = groups)

if __name__ == '__main__':
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT, debug = True)
    save()
