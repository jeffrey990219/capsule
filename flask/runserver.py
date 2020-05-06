import os, shutil
#from lsh import group, preprocessing
from Flann import preprocessing, group, CAPSULE
from flask import Flask, render_template, request, session, url_for, redirect, jsonify
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, patch_request_class, IMAGES
from werkzeug.utils import secure_filename
from save_photos import save
from webbrowser import open
import functools
from skimage import io
import cv2
from glob import glob
import shutil

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = "key"
app.config.update(
    EXISTING_IMG_PATH=os.path.join('static', 'google_photos'),
    NEW_IMG_PATH=os.path.join('static', 'new_photos'),
    TEMP_IMG_PATH=os.path.join('static', 'tmp'),
    DROPZONE_ALLOWED_FILE_TYPE='image',
    DROPZONE_PARALLEL_UPLOADS=100,
    DROPZONE_UPLOAD_MULTIPLE=True,
)

# Create required directories for storing images
try:
    os.mkdir(app.config['EXISTING_IMG_PATH'])
except OSError:
    pass
try:
    os.mkdir(app.config['NEW_IMG_PATH'])
except OSError:
    pass
try:
    os.mkdir(app.config['TEMP_IMG_PATH'])
except OSError:
    pass
    
dropzone = Dropzone(app)

# Calls save_photos.py
@app.route('/open_save_google_photos', methods=['POST', 'GET'])
def open_save_google_photos():
    # Empty EXISTING_IMG_PATH folder
    try:
        shutil.rmtree(app.config['EXISTING_IMG_PATH'], ignore_errors=True)
        os.mkdir(app.config['EXISTING_IMG_PATH'])
    except OSError:
        pass
    
    save()
    preprocessing(google_photo_urls=glob("static/google_photos/*.jpg"))
    print("LOADED!")
    return redirect('/')

# Saves query images to session['NEW_IMGS'] (there's no button associated with this)
# Called by the dropzone stuff
@app.route('/', methods=['POST', 'GET'])
def main():
    # GET Request
    if request.method == 'GET':
        # Empty TEMP_IMG_PATH folder
        try:
            shutil.rmtree(app.config['TEMP_IMG_PATH'], ignore_errors=True)
            os.mkdir(app.config['TEMP_IMG_PATH'])
        except OSError:
            pass
        # Initialize Session Info
        session['NEW_IMGS'] = []
        session['SIMILAR_IMGS_MAPPING'] = {}
        return render_template('index.html')
    
    # POST Request
    else: 
        # Saves query images to session['NEW_IMGS']
        for key, f in request.files.items():
            if key.startswith('file'):  
                # Update NEW_IMGS
                session['NEW_IMGS'].append(f.filename)
                # Save this image to new_photos location first.
                tmp_path = os.path.join(app.config['TEMP_IMG_PATH'], f.filename)
                f.save(tmp_path)
                # Then, query similar images to this image
                res = CAPSULE.query_similar_imgs(query_img_url=tmp_path)
                if len(res) == 0:
                    # If there's no similar image, move this image from new_photos to google_photos
                    dst_url = os.path.join(app.config['NEW_IMG_PATH'], f.filename)
                    shutil.move(
                        src=tmp_path, 
                        dst=dst_url)
                    CAPSULE.insert(dst_url)
                    print("IMAGE SAVED AT {}").format(dst_url)
                else:
                    # If there're similar images, update session.
                    session['SIMILAR_IMGS_MAPPING'][tmp_path] = []
                    for similar_img_url in res:
                        session['SIMILAR_IMGS_MAPPING'][tmp_path].append(similar_img_url)
                    print("SIMILAR_IMGS_MAPPING:", session['SIMILAR_IMGS_MAPPING'])
        session.modified = True
        return render_template('index.html')

@app.route('/confirm-upload/<int:upload>/<path:img_url>')
def confirm_upload(upload, img_url):
    # FIXME: This is a hack for getting file_name; should store the file_name upon upload instead.
    del session['SIMILAR_IMGS_MAPPING'][img_url]
    if upload:
        filename = img_url.split('/')[-1]
        # Move this image from new_photos to google_photos
        dst_url = os.path.join(app.config['NEW_IMG_PATH'], filename)
        shutil.move(
            src=img_url, 
            dst=dst_url)
        CAPSULE.insert(dst_url)
    else:
        os.remove(img_url)
    session.modified = True
    return redirect(url_for('upload_complete'))


@app.route('/upload-complete', methods=['GET', 'POST'])
def upload_complete():
    # print("SIMILAR_IMGS_MAPPING:", session['SIMILAR_IMGS_MAPPING'])
    if session['NEW_IMGS'] == []:
        return redirect(url_for('main'))
    elif session['SIMILAR_IMGS_MAPPING'] == {}:
        return render_template('no_similar_image.html')
    else:
        print("size of SIMILAR_IMGS_MAPPING = %d", len(session['SIMILAR_IMGS_MAPPING']))
        return render_template('similar_image.html', similar_img_mapping=session['SIMILAR_IMGS_MAPPING'])

# Button associated with "Group"
@app.route('/query', methods=['POST', 'GET'])
def query():
    if "NEW_IMGS" not in session or session['NEW_IMGS'] == []:
        return redirect(url_for('upload'))
    filenames = session['NEW_IMGS']
    # List of lists
    groups = group(filenames)
    session.pop('NEW_IMGS', None)
    results = []
    try:
        shutil.rmtree('albums')
    except:
        pass
    os.mkdir('albums')
    for i, g in enumerate(groups):
        if not os.path.exists('albums/Album ' + str(i)):
            os.mkdir('albums/Album ' + str(i))
        shutil.copy(filenames[i], 'albums/Album ' + str(i) + '/' + os.path.basename(filenames[i]))
        new_res = []
        for item in g:
            filepath = os.path.join(app.config['EXISTING_IMG_PATH'], os.path.basename(item))
            new_res.append(filepath)
            shutil.copy(filepath, 'albums/Album ' + str(i) + '/' + os.path.basename(item))
        results.append(new_res)
    return render_template('results.html', queries = filenames, groups = results)



if __name__ == '__main__':
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT, debug = True)