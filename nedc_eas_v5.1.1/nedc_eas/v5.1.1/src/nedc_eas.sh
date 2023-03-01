#!/bin/sh

# file: $NEDC_NFC/bin/nedc_eas
#
# This is a simple driver script for the NEDC annotation system.
#

# set an appropriate environment variable for the root node
# 
NEDC_NFC=$(pwd);
export NEDC_NFC;

# add this tool to the Python library path
#
export PYTHONPATH=$PYTHONPATH:$NEDC_NFC/lib;
echo $PYTHONPATH
# execute the tool
#
python $NEDC_NFC/src/main.py;

#
# end of file
