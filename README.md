# 파이썬 텍스트 마이닝 완벽 가이드 - 박상언, 강주영 저

### 중요사항
1. Python 텍스트 마이닝과 관련하여 보고 복습할 수 있도록 다시 학습 및 코드 추가
2. 책에 있는 코드를 단순히 따라하는 것이 아니라 나만의 코드로 작성
3. 변수명은 Snake Case 사용

### Chapter.1 텍스트 마이닝 기초
1. 텍스트 마이닝이란?
    * NLP를 이용해 텍스트를 정형화된 데이터로 변환하고 머신러닝 기법을 적용하여 고품질 정보를 추출
    * 정형화된 데이터는 일정한 길이의 벡터를 의미 → Embedding
    * Sparse 벡터로 표현할 지 Dense 벡터로 표현할 지는 사용할 머신러닝 방법론과 관계
    * 텍스트 전처리 : 토큰화, 어간/표제어 추출 등의 정규화, 품사 태깅 등
    * 딥러닝의 경우 최근에는 BERT, GPT와 같은 Transfer Learning을 활용
2. 텍스트 마이닝의 주요 적용 분야
    * Text Classification, Text Generation, Summarization, Question Answering, Machine Translation, Topic Modeling 등

### Chapter.2 텍스트 전처리(Text Preprocessing)
1. 텍스트 전처리
    * 주어진 텍스트에서 노이즈 등 불필요한 부분을 제거하고 문장을 표준 단어들로 분리한 후에 각 단어의 품사를 파악
    * 정제(Cleaning), 토큰화(Tokenization), 정규화(Normalization), 품사 태깅(Pos-tagging) 등
    * NLTK, KoNLpPy 등을 이용
2. 토큰화(Tokenization)
    * 문장 토큰화 : 여러 문장으로 이루어진 텍스트를 각 문장으로 나누는 것 → sent_tokenize(para) 함수 이용
    * 단어 토큰화 : 텍스트를 단어 단위로 분리 → word_tokenize(para) 함수 등 이용
        - 클래스를 이용해 tokenizer 객체를 생성한 후 tokenize(para) 메서드 활용
        - 여러 토크나이저가 서로 다른 알고리즘에 기반(ex. WordPunctTokenizer 등)
        - 정규표현식을 이용한 RegexpTokenizer 사용도 가능
        - 한글은 형태소로 분리하는 것이 필요 → Word Segmentation을 적용하는 다른 방법을 사용
    * 불용어(Stopwords) : 빈도가 너무 적거나 많아서 별 필요가 없는 단어들
        - stopwords 라이브러리를 이용해 Stopwords 제거 가능
        - re.sub(regex, sub, para)를 활용하여 Punctuation 제거 가능
3. 정규화(Normalization)
    * 어간 추출(Stemming) : 단어, 특히 용언(Verb, Adjective)의 Stem을 분리해 내는 작업
        - 통시적 어형변화와 공시적 어형변화 존재
        - 영어는 복수형 명사를 단수형으로 바꾸는 작업도 Stemming에 포함
        - Porter Stemmer, Lancaster Stemmer 등 존재 → stemmer 객체 생성 후 stem(token) 메서드 활용
    * 표제어 추출(Lemmatization) : 주어진 단어를 기본형으로 변환하는 것
        - WordNetLemmatizer를 주로 사용 → lemma 객체 생성 후 lemmatize(token) 메서드 활용
        - 품사가 필요할 수도 있으며, pos='v' 등의 파라미터 지정 가능 → 문맥을 통해 파악
4. 품사 태깅(Pos-tagging)
    * 품사 태깅 : 형태소에 대해 Part-of-speech를 파악해 tagging하는 작업
        - 상황에 따라 특정 품사만 필요로 할 때도 활용 가능
    * NLTK는 Penn Treebank Tagset을 주로 활용 → 세분화된 품사 분류
    * 한글의 경우 형태소 단위로 분리하여 토큰화 후 Pos-tagging을 진행해야 → KoNLPy 사용
        - 속도를 위해 주로 Twitter 클래스(Okt) 활용
        - morphs(para) 메서드 : 텍스트를 형태소 단위로 분리
        - nouns(para) 메서드 : 텍스트를 형태소 단위로 분리 후 명사만 반환
        - pos(para) 메서드 : 텍스트를 형태소 단위로 분리 후 품사를 부착하여 튜플로 반환

### Chapter.3 그래프와 워드 클라우드
1. 그래프로 표현하기 위해서는 각 token의 개수 파악이 필요
    * 딕셔너리의 get(word, 0) + 1 메서드를 이용하여 정리 가능
    * 일반적으로 단어의 빈도는 Zipf's law가 적용 → 상위 빈도수 단어들만 정리
2. 워드 클라우드(Word Cloud)
    * 빈도가 높은 단어는 크게 보여줌으로써 전체적인 현황 파악
    * WordCloud 패키지 활용
        - max_font_size, max_words, mask, width, height 등의 파라미터 활용 가능
        - generate(doc) 메서드 활용 → 알아서 토큰화 등의 작업 실행
        - generate_from_frequencies(count_dict) 메서드를 통해 계산된 빈도 사용 가능
        - to_file(path) 메서드를 통해 이미지 파일로 저장 가능

### Chapter.4 카운트 기반의 문서 표현
1. 카운트 기반 문서 표현의 개념
    * 각 단어를 Feature로 두고, 그 단어가 텍스터에서 나타난 횟수를 값으로 표현
    * 성질 상 Sparse 벡터의 형태로 구성 → 효율적으로 처리할 수 있는 방법이 필요
2. sklearn으로 카운트 벡터 생성
    * CountVectorizer() 클래스 활용
        - tokenizer, stop_words, ngram_range, max_df, min_df, max_features, binary 등 파라미터
        - 한글의 경우 KoNLPy를 통한 형태소 분석으로 별도의 tokenizer 활용
        - fit_transform(doc) 메서드 적용하면 Compressed Sparse Row format의 Sparse Matrix 반환 → toarray() 메서드 활용 가능
        - Document Term Matrix는 문서를 행으로, 단어를 열로 해서 빈도를 나타낸 행렬
    * 코사인 유사도(Cosine Similarity)
        - sklearn의 cosine_similarity 함수를 통해 Corpus 내 문서 간 유사도 파악 가능
3. TF-IDF를 통한 카운트 벡터의 발전
    * TfidfVectorizer() 혹은 TfidfTransformer() 클래스 활용

### Chapter.5 BOW 기반의 문서 분류
1. 머신러닝과 문서 분류
    * Naive Bayes : sklearn의 MultinomialNB() 클래스 활용
        - 원칙적으로는 Discrete한 X에 대해서 적용해야하나 Continuous에도 잘 작용 → Tfidf 결과 활용 가능
    * Logistic Regression : sklearn의 LogisticRegression() 클래스 활용
        - Binary일 때와 Multiclass일 때 원래는 서로 다른 알고리즘이나 sklearn에서는 구분 없이 사용 가능
        - L1 Penalty(solver='liblinear') 혹은 L2 Penalty 활용
    * Tree Model : 트리 모델은 일반적으로 텍스트의 카운트 벡터와 잘 맞지 않는 경향이 있음