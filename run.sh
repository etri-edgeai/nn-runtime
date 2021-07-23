#!/bin/bash
if [ $1 == 'baseline' ]
then
  python run.py -t models/yolov5.tflite -i data/HABBOF/Meeting2
elif [ $1 == 'perf' ]
then
  python run.py -t models/1.tflite -i data/HABBOF/Meeting2
elif [ $1 == 'balance' ]
then
  python run.py -t models/3.tflite -i data/HABBOF/Meeting2
elif [ $1 == 'fast' ]
then
  python run.py -t models/5.tflite -i data/HABBOF/Meeting2
fi


