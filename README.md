**Google Photo Organizer**

HOW TO USE THE APP
1. Execute flask/RUN_SERVER in the terminal. This is the server that needs to be run in the back end.
   1. If some modules are not installed, then install these modules using 'pip install'.
2. When the server is on, open a browser and go to http://localhost:5555/
3. Click the green "Google Login" button to allow the app to access your Google Photos. Then start uploading photos.
   1. If the app detects similar photos in Google Photos to a photo to be uploaded, the app will notify and let the user decide whether to still upload that photo or to skip it.
4. After you are done with uploading, 'FILTERED' folder will contain new images that are either distinct from existing photos in Google Photos or similar to an existing photo yet the user still decided to upload.
5. Now you can *manually* upload all images in 'FILTERED' to Google Photos free of concern about having near duplicates.