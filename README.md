Data Preprocessing on Cyberbullying Data
This repository contains a project focused on the preprocessing and analysis of cyberbullying data. The goal is to apply various preprocessing techniques and machine learning models to classify instances of cyberbullying in social media text.

Overview
Cyberbullying is a critical issue in today's digital age. To address this, we have applied several preprocessing techniques and machine learning models to a dataset of cyberbullying text. The models used include SVM, Linear Regression, Neural Networks, LSTM, and various embeddings such as GloVe, BERT, DistilBERT, TF-IDF, One Hot Encoder, and Word2Vec.

Dataset
The dataset used for this project is sourced from the Hugging Face datasets. It contains labeled instances of text data that indicate whether a piece of text is related to cyberbullying or not.

Preprocessing Techniques
1. GloVe (Global Vectors for Word Representation)
GloVe is an unsupervised learning algorithm for obtaining vector representations for words. We used pre-trained GloVe embeddings to convert text data into numerical vectors.

2. BERT (Bidirectional Encoder Representations from Transformers)
BERT is a transformer-based model designed to understand the context of words in search queries. We utilized pre-trained BERT embeddings to capture the contextual information of words.

3. DistilBERT
DistilBERT is a smaller, faster, and cheaper version of BERT. It retains 97% of BERT's language understanding while being more efficient.

4. TF-IDF (Term Frequency-Inverse Document Frequency)
TF-IDF is a statistical measure used to evaluate the importance of a word in a document relative to a collection of documents. It helps in converting text data into numerical vectors based on the frequency of words.

5. One Hot Encoder
One Hot Encoding is a process by which categorical variables are converted into a form that could be provided to machine learning algorithms to do a better job in prediction.

6. Word2Vec
Word2Vec is a group of related models that are used to produce word embeddings. These models are shallow, two-layer neural networks that are trained to reconstruct linguistic contexts of words.

Models Implemented
1. Support Vector Machine (SVM)
SVM is a supervised machine learning algorithm that can be used for both classification or regression challenges. It works by finding the hyperplane that best divides a dataset into classes.

2. Linear Regression
Linear Regression is a linear approach to modeling the relationship between a dependent variable and one or more independent variables. Although primarily used for regression tasks, it can be adapted for classification as well.

3. Neural Network
A neural network is a series of algorithms that attempt to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates.

4. Long Short-Term Memory (LSTM)
LSTM is a type of recurrent neural network capable of learning order dependence in sequence prediction problems. This makes it suitable for tasks like text classification where the context and order of words matter.

Data Splitting
The dataset was divided into training and testing sets to evaluate the performance of the models. The training set was used to train the models, while the testing set was used to assess their performance.

Conclusion
This project demonstrates the application of various text preprocessing techniques and machine learning models to classify cyberbullying instances in text data. By using advanced embeddings and models, we aim to improve the accuracy and efficiency of cyberbullying detection systems.

Feel free to explore the code and the detailed notebooks provided in this repository for more insights into the data preprocessing and modeling techniques used.

