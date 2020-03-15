# Update package lists
sudo apt-get -y update

# Install Python/Pip, dependencies, and submodule(s)
sudo apt-get -y install ffmpeg
sudo apt-get -y install python3
sudo apt-get -y install python3-pip
pip3 install -r requirements.txt
pip3 install gdown

# Download Deepmatching and Deepflow
wget https://thoth.inrialpes.fr/src/deepmatching/code/deepmatching_1.2.2.zip
unzip deepmatching_1.2.2.zip
mv deepmatching_1.2.2_c++/deepmatching-static .
rm -r deepmatching_1.2.2_c++
rm deepmatching_1.2.2.zip
wget http://pascal.inrialpes.fr/data2/deepmatching/files/DeepFlow_release2.0.tar.gz
tar -xf DeepFlow_release2.0.tar.gz
mv DeepFlow_release2.0/deepflow2-static .
rm -r DeepFlow_release2.0
rm DeepFlow_release2.0.tar.gz

# Make ConsistencyChecker
cd consistencyChecker/
make
cd -

# Download style models
gdown https://drive.google.com/uc?id=1GkMMR6yZ29DvGrl-nFybn51Nih-ycll0
unzip styles.zip
rm styles.zip
