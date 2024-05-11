#! /bin/bash

rsync -av --exclude-from="transfer/excluded_files_download.txt" anygard@login.delftblue.tudelft.nl:~/repo ./from-delftblue
