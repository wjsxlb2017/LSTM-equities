# LSTM-equities
Georgia Tech's course CS 7642 Machine Learning for Trading introduces students to Q-Learning as a tool for stock trading. I was curious to try ideas from Natural Language Processing in particular word2vec for both dimensionality reduction and identification of semantic relationships among price patterns (if they exist at all).

<strong>stock2vec_basic.py</strong>
<br>Runs the tokenize function from stock_preprocessing.py and learns a vector space embedding of the tokenized stock data using the basic word2vec example provided by tensorflow. A directory stock2vec_log will be created. It contains a file metadata.tsv which must be copied and renamed TokenToRow.txt in the next higher directory. This file contains all unique tokens in decending order of frequency which corresponds to each token's row position in the embedding matrix.

<strong>stock_preprocessing.py</strong>
<br>In addition to tokenization, contains several functions for generating class labels for historical stock price data based on daily price changes, trading ranges, trading volume, moving averages, and S&P500 behavior. Also contains a function for packaging the transformed stock data and class labels into the file LSTM_data.h5 which will feed the LSTM.

<strong>LSTM_stock2vec_1.py</strong>
<br>Two generators feed batches of training and validation data to a keras/tensorflow LSTM. Chronological ordering of batches is preserved (roll-forward cross-validation).
