# Steps to Follow
* Remove those csv if you arent using it for DEV purposes
* create folders video val and frames with subfolders real and fake
* create models folder in data
* download the requirements from pip into a virtual env
* take your dataset and put the videos in videos real and fake folders accordingly
* if more than one real and fake folders write it down in the list inside prepare_data.py you can read and understand it
* run prepare_data.py
* then train_vit.py
* then you can use predict.py give filepath of your video and check for ai authenticity
