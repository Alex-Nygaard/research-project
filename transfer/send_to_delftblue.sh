#! /bin/bash

rsync -av --exclude-from="transfer/excluded_files_upload.txt" ./ anygard@login.delftblue.tudelft.nl:~/repo
