# Install Python/Pip, dependencies, and submodule(s)
sudo apt-get -y install ffmpeg
sudo apt-get -y install python3
sudo apt-get -y install python3-pip
pip3 install -r requirements.txt
pip3 install gdown

# Download Deepmatching and Deepflow
curl https://thoth.inrialpes.fr/src/deepmatching/code/deepmatching_1.2.2.zip
unzip deepmatching_1.2.2.zip
mv deepmatching_1.2.2/deepmatching-static .
curl http://pascal.inrialpes.fr/data2/deepmatching/files/DeepFlow_release2.0.tar.gz
tar -xf DeepFlow_release2.0.tar.gz
mv DeepFlow_release2.0/deepflow2-static .

# Download style models
gdown https://drive.google.com/open?id=1GkMMR6yZ29DvGrl-nFybn51Nih-ycll0
unzip styles.zip