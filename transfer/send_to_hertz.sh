#! /bin/bash

rsync -av --exclude-from="transfer/excluded_files_upload.txt" ./ root@167.235.241.219:~/research-project
