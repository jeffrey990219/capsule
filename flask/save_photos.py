from os.path import join, dirname
from skimage import io
import cv2
from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools
import time
SCOPES = 'https://www.googleapis.com/auth/photoslibrary.readonly'

def save():
	# Google Photos authentication
	start = time.time()
	store = file.Storage('credentials.json')
	creds = None
	if not creds or creds.invalid:
	    flow = client.flow_from_clientsecrets('client_secret.json', SCOPES)
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
		cv2.imwrite("static/google_photos/" + str(i) + ".jpg", img)
		urls.append(url)
	end = time.time()
	print (end - start)
	return urls
