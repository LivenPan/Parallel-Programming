#!/bin/bash

# Do not uncomment these lines to directly execute the script
# Modify the path to fit your need before using this script

INPUT_FILE=/user/ta/PageRank/Input/input-100M
OUTPUT_FILE=PageRank/Output
RANK_FILE=PageRank/Rank
PARSE_FILE=PageRank/Parse
JAR=PageRank.jar

hdfs dfs -rm -r $PARSE_FILE
hdfs dfs -rm -r $RANK_FILE
hdfs dfs -rm -r $OUTPUT_FILE
hadoop jar $JAR pagerank.PageRank $INPUT_FILE $PARSE_FILE $RANK_FILE $OUTPUT_FILE 2
hdfs dfs -getmerge $OUTPUT_FILE pagerank.txt
