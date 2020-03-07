#!/bin/bash

rename () {
    for file in $1*;
    do
        new_file=$(echo $file | sed "s/$2//")
        if [ ${new_file: -4} == ".txt" ];
        then
            mv "$file" "$new_file"
        else
            mv "$file" "$new_file.txt"
        fi
    done;
}

rename data/split-2/train/train.txt. train.txt.
rename data/split-2/dev/dev.txt.shuffle.dev. dev.txt.shuffle.dev.
rename data/split-2/test/dev.txt.shuffle.test. dev.txt.shuffle.test.
