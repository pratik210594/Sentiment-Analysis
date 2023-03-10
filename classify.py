import math, re

# A simple tokenizer. Applies case folding
def tokenize(s):
    tokens = s.lower().split()
    trimmed_tokens = []
    for t in tokens:
        if re.search('\w', t):
            # t contains at least 1 alphanumeric character
            t = re.sub('^\W*', '', t) # trim leading non-alphanumeric chars
            t = re.sub('\W*$', '', t) # trim trailing non-alphanumeric chars
        trimmed_tokens.append(t)
    return trimmed_tokens

def categorize(text, pos,neg):
    
    classify = []
    for test in text:
        pcount, ncount = 0,0
        tokens = tokenize(test)
        for token in tokens:
            if token in pos:
                pcount += 1
            elif token in neg:
                ncount += 1
        if pcount > ncount:
            classify.append("positive")
        elif ncount > pcount:
            classify.append("negative")
        else:
            classify.append("neutral")
    return classify

def categorizenb(test_texts, trainklasses,train_texts, method):
    probpositive, probnegative, probneutral =0,0,0
    total, totalpositive, totalnegative, totalneutral =0,0,0,0
    pcount, ncount, neutralcount = 0,0,0
    classify = []
    for klass in trainklasses:
        if klass == "positive":
            pcount+=1
        elif klass == "negative":
            ncount+=1
        else:
            neutralcount +=1 
    total = pcount + ncount + neutralcount
    probpositive = pcount/total
    probnegative = ncount/total
    probneutral = neutralcount/total
    # print(probpositive,probnegative,probneutral)

    neg = {}
    pos = {}
    neutral = {}
    positivecount, negativecount,neutralcount = 0,0,0
    # list = train(train_klasses)
    if method == "nb":
        for klass, train in zip(trainklasses, train_texts):
            # print(klass,train)
            token = tokenize(train)
            for t in token:
                if klass == "negative":
                    if neg.get(t) == None:
                        neg[t] = 1
                        negativecount+=1
                    else:
                        neg[t] +=1
                        negativecount+=1
                elif klass == "positive":
                    if pos.get(t) == None:
                        pos[t] = 1
                        positivecount+=1
                    else:
                        pos[t] +=1
                        positivecount+=1
                else:
                    if neutral.get(t) == None:
                        neutral[t] = 1
                        neutralcount+=1
                    else:
                        neutral[t] +=1
                        neutralcount+=1

        argpos,argneg,argneu = probpositive,probnegative,probneutral

        # vocab = pos.union(neg)
        # d3={**d1,**d2}
        vocab = {**pos,**neg,**neutral}     #to get types
        # print(len(vocab))                   #5969 vocab size
        
        for test in test_texts:
            resultneg, resultpos, resultneutral = 0,0,0
            negprior,posprior,neuprior = 0,0,0
            token = tokenize(test)
            for t in token:
                numerator = (neg.get(t,0)+1)
                denominator = (negativecount+len(vocab))
                # prior = math.log(argneg)
                resultneg = (math.log(numerator)-math.log(denominator))
                negprior += resultneg
                # print(resultneg)
                numerator = (pos.get(t,0)+1)
                denominator = (positivecount+len(vocab))
                # prior = math.log(argpos)
                resultpos =(math.log(numerator)-math.log(denominator))
                posprior += resultpos
                # print(resultpos)
                numerator = (neutral.get(t,0)+1)
                denominator = (neutralcount+len(vocab))
                # prior = math.log(argneu)
                resultneutral =(math.log(numerator)-math.log(denominator))
                neuprior += resultneutral
                # print(resultneutral)
            negprior += math.log(argneg)
            posprior += math.log(argpos)
            neuprior += math.log(argneu)
            if negprior > posprior and negprior > neuprior:
                # print ("negative")
                classify.append("negative")
            elif posprior > negprior and posprior > neuprior:
                # print ("positive")
                classify.append("positive")
            else:
                # print("neutral")
                classify.append("neutral")
        return classify
    elif method == "nbbin":
        for klass, train in zip(trainklasses, train_texts):
            # print(klass,train)
            token = set(tokenize(train))
            for t in token:
                if klass == "negative":
                    if neg.get(t) == None:
                        neg[t] = 1
                        negativecount+=1
                    else:
                        continue
                elif klass == "positive":
                    if pos.get(t) == None:
                        pos[t] = 1
                        positivecount+=1
                    else:
                        continue
                else:
                    if neutral.get(t) == None:
                        neutral[t] = 1
                        neutralcount+=1
                    else:
                        continue
        argpos,argneg,argneu = probpositive,probnegative,probneutral
        vocab = {**pos,**neg,**neutral} 
      
        for test in test_texts:
            token = tokenize(test)
            resultneg, resultpos, resultneutral = 0,0,0
            negprior,posprior,neuprior = 0,0,0
            for t in set(token):
                # if t in neg:
                numerator = (neg.get(t,0)+1)
                denominator = (negativecount+len(vocab))
                # prior = math.log(argneg)
                resultneg = (math.log(numerator)-math.log(denominator))
                negprior += resultneg
                # print(resultneg)
            # elif t in pos:
                numerator = (pos.get(t,0)+1)
                denominator = (positivecount+len(vocab))
                # prior = math.log(argpos)
                resultpos = (math.log(numerator)-math.log(denominator))
                posprior += resultpos
                # print(resultpos)
            # elif t in neutral:
                numerator = (neutral.get(t,0)+1)
                denominator = (neutralcount+len(vocab))
                # prior = math.log(argneu)
                resultneutral =(math.log(numerator)-math.log(denominator))
                neuprior += resultneutral
                # print(resultneutral)
            negprior += math.log(argneg)
            posprior += math.log(argpos)
            neuprior += math.log(argneu)
               
            if negprior > posprior and negprior > neuprior:
                # print ("negative")
                classify.append("negative")
            elif posprior > negprior and posprior > neuprior:
                # print ("positive")
                classify.append("positive")
            else:
                # print("neutral")
                classify.append("neutral")
        return classify
            
            

# A most-frequent class baseline
class Baseline:
    def __init__(self, klasses):
        self.train(klasses)

    def train(self, klasses):
        # Count classes to determine which is the most frequent
        klass_freqs = {}
        for k in klasses:
            klass_freqs[k] = klass_freqs.get(k, 0) + 1
        self.mfc = sorted(klass_freqs, reverse=True, 
                          key=lambda x : klass_freqs[x])[0]
    
    def classify(self, test_instance):
        return self.mfc

if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    # Method will be one of 'baseline', 'lr', 'lexicon', 'nb', or
    # 'nbbin'
    method = sys.argv[1]

    train_texts_fname = sys.argv[2]
    train_klasses_fname = sys.argv[3]
    test_texts_fname = sys.argv[4]
    
    train_texts = [x.strip() for x in open(train_texts_fname,
                                           encoding='utf8')]
    train_klasses = [x.strip() for x in open(train_klasses_fname,
                                             encoding='utf8')]
    test_texts = [x.strip() for x in open(test_texts_fname,
                                          encoding='utf8')]
    positive_words = [x.strip() for x in open("pos-words.txt",
                                          encoding='utf8')]
    negative_words = [x.strip() for x in open("neg-words.txt",
                                          encoding='utf8')]


    if method == 'baseline':
        classifier = Baseline(train_klasses)
        results = [classifier.classify(x) for x in test_texts]

    elif method == 'lr':
        # Use sklearn's implementation of logistic regression
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression

        # sklearn provides functionality for tokenizing text and
        # extracting features from it. This uses the tokenize function
        # defined above for tokenization (as opposed to sklearn's
        # default tokenization) so the results can be more easily
        # compared with those using NB.
        # http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
        count_vectorizer = CountVectorizer(analyzer=tokenize)

        # train_counts will be a DxV matrix where D is the number of
        # training documents and V is the number of types in the
        # training documents. Each cell in the matrix indicates the
        # frequency (count) of a type in a document.
        train_counts = count_vectorizer.fit_transform(train_texts)

        # Train a logistic regression classifier on the training
        # data. A wide range of options are available. This does
        # something similar to what we saw in class, i.e., multinomial
        # logistic regression (multi_class='multinomial') using
        # stochastic average gradient descent (solver='sag') with L2
        # regularization (penalty='l2'). The maximum number of
        # iterations is set to 1000 (max_iter=1000) to allow the model
        # to converge. The random_state is set to 0 (an arbitrarily
        # chosen number) to help ensure results are consistent from
        # run to run.
        # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        lr = LogisticRegression(multi_class='multinomial',
                                solver='sag',
                                penalty='l2',
                                max_iter=1000,
                                random_state=0)
        clf = lr.fit(train_counts, train_klasses)

        # Transform the test documents into a DxV matrix, similar to
        # that for the training documents, where D is the number of
        # test documents, and V is the number of types in the training
        # documents.
        test_counts = count_vectorizer.transform(test_texts)
        # Predict the class for each test document
        results = clf.predict(test_counts)

    elif method =='lexicon':
           results = categorize(test_texts,positive_words,negative_words)
    elif method == "nb" or method == "nbbin":
        results = categorizenb(test_texts,train_klasses,train_texts,method)

    for r in results:
        print(r.encode('ascii','ignore').decode('utf-8'))
