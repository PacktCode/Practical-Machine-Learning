mkdir -p data || exit 1
cd data
wget http://ufldl.stanford.edu/wiki/resources/sparseae_exercise.zip
unzip sparseae_exercise.zip
cp starter/IMAGES.mat IMAGES.mat
rm -rf starter
rm sparseae_exercise.zip
cd ..
