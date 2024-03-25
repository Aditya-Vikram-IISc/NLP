# utils.py
# Basic text processing functions

import os
from string import punctuation


def get_txtfiles_in_a_folder(folderpath:str)-> list[str]:
    all_files = [os.path.join(folderpath,x) for x in os.listdir(folderpath) if x.endswith(".txt")]
    return all_files

def get_data_from_txtfilepath(txtfilepath:str)-> tuple[str, int]:
    # get the rating
    rating = int(os.path.splitext(txtfilepath)[0].split("_")[-1])

    # get the review
    with open(txtfilepath, "r") as f:
        review = f.read()

    # process the review by smallcasing everything
    review = review.lower()

    # remove punctuation
    review = "".join([x for x in review if x not in punctuation])

    return review, rating