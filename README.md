**Google Photo Organizer**

Demo Video: https://youtu.be/wOX6c0Skjjg
Report: https://docs.google.com/document/d/1MoAxWuvk_12WVReot2gNLqAcPsRuIY409TE8VgW3Vm4/edit?usp=sharing

HOW TO USE THE APP
1. "cd" into "flask" and run **"./RUN_SERVER"**. This is the server that needs to be run in the back end.
   1. If some Python modules are not installed, then install these modules using "pip install".
   2. If running this file does not work, run Server.py directly via either Python 2 or 3 **"python flask/Server.py"**
2. When the server is on, open a browser and go to http://localhost:5555/
3. Click the green "Google Login" button to allow the app to authenticate and access your Google Photos (new tab will open).
   1. Make sure that whichever Google account you select actually has photos in Google Photos.
4. Close the tab after the authentification workflow has been completed, and a loading screen will appear on the main application tab.
5. After the photos are loaded, you can start uploading photos in the drop area. Click 'Upload' when finished.
6. If the app detects similar photos in your Google Photos to an uploaded photo, the app will notify and let the user decide whether to still upload that photo (Save) or not (Skip). 
7. After you are done with uploading, the **"FILTERED"** folder will contain new images that are either distinct from the existing photos in Google Photos or similar to an existing photo yet the user still decided to upload.
8. Now you can *manually* upload all images in 'FILTERED' to Google Photos free of concern about having near duplicates.
