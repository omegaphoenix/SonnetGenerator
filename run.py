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
# Fill train with the training examples
for line in input_file.readlines():
    # Between sonnets
    if line in ['\n', '\r\n']:
        # If there is the wrong number of lines, skip
        if num_lines == 15:
            train.append(sonnet)
        else:
            print num_lines
        num_lines = 0
        sonnet = ""
    # Add line to sonnet
    else:
        # First line is just a number
        if num_lines != 0:
            sonnet += line.strip()
        num_lines += 1

<<<<<<< HEAD
def sonnet_to_words( raw_sonnet ):
    # Function to remove non-letters
    #
    # 1. Remove HTML
    #review_text = BeautifulSoup(raw_sonnet).get_text()
    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", raw_sonnet)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. Join the words back into one string separated by space,
    # and return the result.
    return( " ".join( words ))


# Get the number of sonnets 
num_sonnets = len(train)

# Initialize an empty list to hold the clean sonnets
clean_train_sonnets = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 
for i in xrange( 0, num_sonnets ):
    if( (i+1)%1000 == 0 ):
        print "Sonnet %d of %d\n" % ( i+1, num_reviews )
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_train_sonnets.append( sonnet_to_words( train[i]))

print clean_train_sonnets
"""print "Creating the bag of words...\n"
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
