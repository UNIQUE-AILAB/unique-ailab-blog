#!/bin/sh
npm install
curl https://raw.githubusercontent.com/ver217/marked/master/lib/marked.js > ./node_modules/marked/lib/marked.js
if [! -d "./themes"];   then
    mkdir ./themes
fi
git clone -b blog git@github.com:ver217/hexo-theme-Mic_Theme.git ./themes/mic
