# Import the pandas package, then use the "read_csv" function to read
# the labeled training data
import pandas as pd       
# Import BeautifulSoup into your workspace
from bs4 import BeautifulSoup             
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import *

input_file = open('shakespeare.txt')
train = []
sonnet = ""
num_lines = 0
for line in input_file.readlines():
    print line
    # Between sonnets
    if line == "\n":
        # If there is the wrong number of lines, skip
        if num_lines == 15:
            train.append(sonnet)
        num_lines = 0
        sonnet = ""
    # Add line to sonnet
    else:
        # First line is just a number
        if num_lines != 0:
            sonnet += line
        num_lines += 1


def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             

# Get the number of reviews based on the dataframe column size
#num_reviews = train["review"].size

# Initialize an empty list to hold the clean reviews
clean_train_sonnets = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 
"""for i in xrange( 0, num_reviews ):
    if( (i+1)%1000 == 0 ):
        print "Review %d of %d\n" % ( i+1, num_reviews )
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_train_reviews.append( review_to_words( train["review"][i]))

print "Creating the bag of words...\n"
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)"""
