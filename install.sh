# Update package lists
sudo apt-get -y update

# Install Python/Pip, dependencies, and submodule(s)
sudo apt-get -y install ffmpeg
sudo apt-get -y install python3
sudo apt-get -y install python3-pip
pip3 install --upgrade pip
pip3 install -r requirements.txt
pip3 install gdown
pip3 install opencv-python
git submodule update --init
cd flowcalc
git checkout master
bash install.sh
cd -

# Download style models
gdown https://drive.google.com/uc?id=1GkMMR6yZ29DvGrl-nFybn51Nih-ycll0
unzip styles.zip
rm styles.zip
