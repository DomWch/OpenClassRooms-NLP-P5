git init
git remote add origin https://github.com/DomWch/OpenClassRooms-NLP-P5.git
git fetch
git checkout OpenClassRooms-NLP-P5/dev -- api
rm -r .git
git remote add origin https://huggingface.co/spaces/Domw/p5-nlp
git pull origin main
git add api/*
git commit -m "$1" #commit message from github
git push -u origin main