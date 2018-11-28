from nltk.corpus import movie_reviews
from nltk.tokenize import sent_tokenize 
from nltk.classify import NaiveBayesClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify.util import accuracy as nltk_accuracy
import numpy as np
import random
 
# Extract features from the input list of words
def extract_features(words):
    return dict([(word, True) for word in words])
 
if __name__=='__main__':
    # Load the reviews from the corpus 
    fileids_pos = movie_reviews.fileids('pos')
    fileids_neg = movie_reviews.fileids('neg')
    random.seed(28)
    random.shuffle(fileids_pos)
    random.shuffle(fileids_neg)

    # Extract the features from the reviews
    features_pos = [(extract_features(movie_reviews.words(
            fileids=[f])), 'Positive') for f in fileids_pos]
    features_neg = [(extract_features(movie_reviews.words(
            fileids=[f])), 'Negative') for f in fileids_neg] 

    # Create training and testing datasets
    length = int(0.95*len(features_pos))
    features_train = features_pos[:length] + features_neg[:length] 
    features_test =  features_pos[length:] + features_neg[length:]
    
    # Train classifiers
    ONB_classifier = NaiveBayesClassifier.train(features_train)
    MNB_classifier = SklearnClassifier(MultinomialNB(alpha=1)).train(features_train)
    BNB_classifier = SklearnClassifier(BernoulliNB(alpha=1,binarize=0)).train(features_train)
    LGR_classifier = SklearnClassifier(LogisticRegression()).train(features_train)
    SDGC_classifier = SklearnClassifier(SGDClassifier(max_iter=1000,tol=1e-3)).train(features_train)
    SVC_classifier = SklearnClassifier(SVC()).train(features_train)
    LSVC_classifier = SklearnClassifier(LinearSVC()).train(features_train)
    NuSVC_classifier = SklearnClassifier(NuSVC()).train(features_train) #nu <= 0 or nu > 1

    # N = 15
    # print('\nTop ' + str(N) + ' most informative words:')
    # for i, item in enumerate(MNB_classifier.most_informative_features()):
    #     print(str(i+1) + '. ' + item[0])
    #     if i == N - 1:
    #         break

    print('ONB_classifier accuracy: ',nltk_accuracy(ONB_classifier,features_test))
    print('MNB_classifier accuracy: ',nltk_accuracy(MNB_classifier,features_test))
    print('BNB_classifier accuracy: ',nltk_accuracy(BNB_classifier,features_test))
    print('LGR_classifier accuracy: ',nltk_accuracy(LGR_classifier,features_test))
    print('SDGC_classifier accuracy: ',nltk_accuracy(SDGC_classifier,features_test))
    print('SVC_classifier accuracy: ',nltk_accuracy(SVC_classifier,features_test))
    print('LSVC_classifier accuracy: ',nltk_accuracy(LSVC_classifier,features_test))
    print('NuSVC_classifier accuracy: ',nltk_accuracy(NuSVC_classifier,features_test))
    
    # Test input movie reviews
    with open('text.txt','r',encoding='utf-8') as f1:
        input_reviews = sent_tokenize(f1.read())

    f1.close()

    f = open('result.txt','w',encoding='utf-8')
    f.write("Review\tPredicted sentiment\tProbability\n")
    for review in input_reviews:
        review = review.replace('\n',' ')

        f.write(review + '\t')

        # Compute the probabilities
        probabilities = LGR_classifier.prob_classify(extract_features(review.split()))

        # Pick the maximum value
        predicted_sentiment = probabilities.max()

        # Print outputs
        f.write(predicted_sentiment + '\t')
        f.write('{}'.format(round(probabilities.prob(predicted_sentiment), 2)) + '\n')
    
    f.close()