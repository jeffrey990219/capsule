import os
from skimage import io
import cv2
from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools
import time
import argparse
SCOPES = 'https://www.googleapis.com/auth/photoslibrary'

def save(google_photos_path, client_secret_path, credentials_path):
	# Google Photos authentication
	start = time.time()
	# Store credentials.json in the same directory as client_secret.json
	store = file.Storage(credentials_path)
	creds = None
	# creds = store.get() # can specify store.get() here if credentialgis are already available
	if not creds or creds.invalid:
		parser = argparse.ArgumentParser(parents=[tools.argparser])
		flags = parser.parse_args()
		flow = client.flow_from_clientsecrets('client_secret.json', SCOPES)
		creds = tools.run_flow(flow, store, flags)
	google_photos = build('photoslibrary', 'v1', http=creds.authorize(Http()))

	# Get items from all albums, as well as stray photos
	nextPageToken = ''
	items = []
	for _ in range(2):
		results = google_photos.mediaItems().list(pageSize = 25, pageToken = nextPageToken).execute()
		# Get all photos 100 at a time (200 max)
		items.extend(results.get('mediaItems', []))
		nextPageToken = results.get('nextPageToken', '')
		if nextPageToken == '':
			break

	# Save all photos locally
	print("Downloading %d images into 'static/google_photos/' ..." %(len(items)))
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
