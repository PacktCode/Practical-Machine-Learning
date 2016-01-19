#!/bin/sh

REUTERS_FILE="reuters21578.tar.gz"
if [ ! -f $REUTERS_FILE ]
then
  wget http://www.daviddlewis.com/resources/testcollections/reuters21578/$REUTERS_FILE
fi
mkdir -p reuters
(cd reuters; tar xvfz ../$REUTERS_FILE)
