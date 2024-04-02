from string import punctuation
import numpy as np
import torch



def process_test_reviewx(review: str, vocab_to_int_mapper: dict):
    # all smallcase
    review = review.lower()

    # remove punctuations
    review = "".join([character for character in review if character not in punctuation])
    review = review.split(" ")
    # map words to integer
    review_encoded = []
    for word in review:
        try:
            review_encoded.append(vocab_to_int_mapper[word])
        except:
            continue

    # pad / truncate the review is not nessesary
    review_encoded = np.expand_dims(review_encoded, axis = 0)

    return review_encoded


def predict_sentiment(review_encoded, model, device, batch_size = 1):
    model.eval()
    h = model.init_hidden(batch_size = batch_size, device = device)
    
    # get the input
    review_encoded = torch.tensor(review_encoded).to(device)

    output,_ = model(review_encoded, h)

    return output.item()