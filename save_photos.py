from os.path import join, dirname
from skimage import io
import cv2
from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools
import time
SCOPES = 'https://www.googleapis.com/auth/photoslibrary.readonly'


store = file.Storage('credentials.json')
creds = None
print("hi")
if not creds or creds.invalid:
    flow = client.flow_from_clientsecrets('client_secret_839718518795-advg62g6djcnggvn6dm66idufa1pa8nk.apps.googleusercontent.com.json', SCOPES)
    creds = tools.run_flow(flow, store)
google_photos = build('photoslibrary', 'v1', http=creds.authorize(Http()))

results = google_photos.albums().list(
    pageSize=10).execute()
items = results.get('albums', [])
print (items)
for item in items:
	print(u'{0} ({1})'.format(item['title'].encode('utf8'), item['id']))

# nextpagetoken = 'Dummy'
# while nextpagetoken != '':
#     nextpagetoken = '' if nextpagetoken == 'Dummy' else nextpagetoken
results = google_photos.mediaItems().list(pageSize = 100).execute()
# The default number of media items to return at a time is 25. The maximum pageSize is 100.
items = results.get('mediaItems', [])
# nextpagetoken = results.get('nextPageToken', '')

start  =time.time()
print (len(items))
for i in range(len(items)):
	url = items[i]['baseUrl']
	img = io.imread(url)
	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	cv2.imwrite("photos/" + str(i) + ".jpg", img)
end = time.time()
print (end - start)