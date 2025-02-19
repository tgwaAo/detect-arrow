#!/bin/env python3

import sys
import os

if not len(sys.argv) == 2:
    exit(1)

keyword = sys.argv[1]

for searched_filename in os.listdir('.'):
    if not searched_filename.endswith('.py'):
        continue



    with open(searched_filename, 'r') as fid:
        content = fid.readlines()

    found_key = False
    func_content= []
    for line in content:
        if found_key:
            func_content.append(line)

            if 'def' in line:
                break

        if keyword in line and 'def ' in line:
            found_key = True
            func_content.append(line)

    if func_content:
        saved_filename = f'{keyword}_{searched_filename}.txt'
        with open(saved_filename, 'w') as fid:
            fid.writelines(func_content)






