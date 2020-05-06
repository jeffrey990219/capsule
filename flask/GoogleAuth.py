import os
from skimage import io
import cv2
from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools
import time
SCOPES = 'https://www.googleapis.com/auth/photoslibrary.readonly'

def save(google_photos_path, client_secret_path, credentials_path):
	# Google Photos authentication
	start = time.time()
	# Store credentials.json in the same directory as client_secret.json
	store = file.Storage(credentials_path)
	creds = None
	if not creds or creds.invalid:
	    flow = client.flow_from_clientsecrets(client_secret_path, SCOPES)
	    creds = tools.run_flow(flow, store)
	google_photos = build('photoslibrary', 'v1', http=creds.authorize(Http()))

	# Get items from all albums, as well as stray photos
	results = google_photos.albums().list(
	    pageSize=10).execute()
	items = results.get('albums', [])
	for item in items:
		print(u'{0} ({1})'.format(item['title'].encode('utf8'), item['id']))
	results = google_photos.mediaItems().list(pageSize = 100).execute()

	# Get all photos
	items = results.get('mediaItems', [])

	# Save all photos locally
	print (len(items))
	urls = []
	for i in range(len(items)):
		url = items[i]['baseUrl']
		img = io.imread(url)
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		img_url = str(i) + ".jpg"
		cv2.imwrite(os.path.join(google_photos_path, img_url), img)
		urls.append(url)
	end = time.time()

	print("%d images have been downloaded into %s in %d seconds." %(len(items), google_photos_path, end - start))
	return urls
