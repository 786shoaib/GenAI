import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


paragraph = """The modern Olympic Games or Olympics (French: Jeux olympiques)[a][1] are the leading international sporting events featuring summer and winter sports competitions in which thousands of athletes from around the world participate in a variety of competitions. The Olympic Games are considered the world's foremost sports competition with more than 200 teams, representing sovereign states and territories, participating. By default, the Games generally substitute for any world championships during the year in which they take place (however, each class usually maintains its own records).[2] The Olympic Games are held every four years. Since 1994, they have alternated between the Summer and Winter Olympics every two years during the four-year Olympiad.[3][4]

Their creation was inspired by the ancient Olympic Games, held in Olympia, Greece from the 8th century BC to the 4th century AD. Baron Pierre de Coubertin founded the International Olympic Committee (IOC) in 1894, leading to the first modern Games in Athens in 1896. The IOC is the governing body of the Olympic Movement, which encompasses all entities and individuals involved in the Olympic Games. The Olympic Charter defines their structure and authority."""

# Tokensize -> convert paragraph sentence words
# nltk.download('punkt')
sentences = nltk.sent_tokenize(paragraph)

# nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

import re
corpus = []
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]',' ',sentences[i])
    review = review.lower().split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in stop_words]
    corpus.append(' '.join(review))
    
print(corpus)
    
# Applying Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

Bow = cv.fit_transform(corpus)

print(len(cv.vocabulary_))
print(cv.vocabulary_)

for i in range(len(corpus)):
    print(corpus[i])
    print(Bow[i].toarray())