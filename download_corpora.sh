#!/usr/bin/env bash

# TREC
mkdir data
mkdir data/TREC
cd data/TREC
mkdir train;mkdir test
cd train
wget http://cogcomp.org/Data/QA/QC/train_1000.label
wget http://cogcomp.org/Data/QA/QC/train_2000.label
wget http://cogcomp.org/Data/QA/QC/train_3000.label
wget http://cogcomp.org/Data/QA/QC/train_4000.label
wget http://cogcomp.org/Data/QA/QC/train_5500.label
cd ../test
wget http://cogcomp.org/Data/QA/QC/TREC_10.label
cd ../../..

# SSTB
mkdir data/SSTB
cd data/SSTB
wget http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
tar -xzf stanfordSentimentTreebank.zip
rm -rf __MACOSX
rm stanfordSentimentTreebank.zip
cd ../..

# MR
mkdir data/MR
cd data/MR
wget https://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz
tar -xvzf review_polarity.tar.gz
rm review_polarity.tar.gz
cd ../..

python process_trec.py -d data/TREC/
python process_treebank.py -d data/SSTB/stanfordSentimentTreebank