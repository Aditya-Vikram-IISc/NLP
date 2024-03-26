# utils.py
# Basic text processing functions

import os
import numpy as np
from string import punctuation
from collections import Counter


def get_txtfilepaths_in_a_folder(folderpath:str)-> list[str]:
    all_filepaths = [os.path.join(folderpath,x) for x in os.listdir(folderpath) if x.endswith(".txt")]
    return all_filepaths


def get_data_from_txtfilepath(txtfilepath:str)-> tuple[str, int]:
    # get the rating
    rating = int(os.path.splitext(txtfilepath)[0].split("_")[-1])

    # get the review
    with open(txtfilepath, "r", encoding="utf8") as f:
        review = f.read()

    # process the review by smallcasing everything
    review = review.lower()

    # remove punctuation
    review = "".join([x for x in review if x not in punctuation])

    return review, rating


def read_data_from_folder(folderpath:str)->list[list[str], list[int]] :
    # get path of all textfiles in the folder
    all_filepathss = get_txtfilepaths_in_a_folder(folderpath)

    # parse the data
    ratings = []
    reviews = []

    for fp in all_filepathss:
        # get the ratings and reviews
        try:
            review, rating = get_data_from_txtfilepath(fp)

            ratings.append(rating)
            reviews.append(review)
        except:
            print(fp)
    return reviews, ratings 


def get_word_count(review: list[str]):
    
    # get all reviews concatenated into a giant string
    reviews_str = " ".join(review)

    # create a list of words from concatenated reviews
    words_list = reviews_str.split(" ")

    # get a word count object
    word_counts = Counter(words_list)

    return word_counts


def map_words_to_index(review: str, vocab_to_int_mapper: dict)-> list[int]:
    review_encoded = []
    for word in review.split(" "):
        try:
            review_encoded.append(vocab_to_int_mapper[word])
        except:
            review_encoded.append(1)

    return review_encoded
    

def pad_and_truncate_reviews(reviews_encoded: list[list[int]], seq_length:int):

    # create a zero array of size #num_reviews X seq_length
    feature_array = np.zeros((len(reviews_encoded), seq_length), dtype = int)

    for index, review in enumerate(reviews_encoded):
        
        # trucate if length if review is larger than SEQUENCE_LENGTH
        if len(review) >= seq_length:
            review = review[:seq_length]
        
        # else pad it with zeros
        elif len(review) < seq_length:
            review = list(np.zeros((seq_length-len(review)), dtype =int)) + review

        feature_array[index, :] = np.array(review)

    
    return feature_array

