export WORK_DIR=/chapter9/mahout/naive-bayes
mkdir $WORK_DIR
cd $WORK_DIR
wget http://people.csail.mit.edu/jrennie/20Newsgroups/20news-bydate.tar.gz
tar –xvzf 20news-bydate.tar.gz
mkdir ${WORK_DIR}/20news-all
mkdir ${WORK_DIR}/20news-seq
cp -R ${WORK_DIR}/20news-bydate*/*/* ${WORK_DIR}/20news-all


mahout seqdirectory   -i ${WORK_DIR}/20news-all  -o ${WORK_DIR}/20news-seq -ow

mahout seq2sparse  -i ${WORK_DIR}/20news-seq   -o ${WORK_DIR}/20news-vectors  -lnorm -nv  -wt tfidf -ow

mahout split  -i ${WORK_DIR}/20news-vectors/tfidf-vectors     --trainingOutput ${WORK_DIR}/20news-train-vectors     --testOutput ${WORK_DIR}/20news-test-vectors      --randomSelectionPct 40 --overwrite --sequenceFiles -xm sequential 