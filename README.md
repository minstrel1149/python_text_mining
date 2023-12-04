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
        - 참고 : MultinomialNB() 클래스는 X에 Negative Value를 허용하지 않음
    * Logistic Regression : sklearn의 LogisticRegression() 클래스 활용
        - Binary일 때와 Multiclass일 때 원래는 서로 다른 알고리즘이나 sklearn에서는 구분 없이 사용 가능
        - L1 Penalty(solver='liblinear') 혹은 L2 Penalty 활용
    * Tree Model : 트리 모델은 일반적으로 텍스트의 카운트 벡터와 잘 맞지 않는 경향이 있음
2. N-gram을 이용한 카운트 벡터의 보완
    * N-gram : 연속적인 단어의 나열. Unigram, Bigram, Trigram 등
    * BOW 기반 방식은 벡터의 크기가 커서 Overfitting의 문제 → 많아야 Trigram까지 쓰는 것이 일반적
    * Vectorizer에서 ngram_range(min, max) 파라미터 삽입
    * 긴 단어 시퀀스로 인한 문맥은 여전히 파악 불가 → 딥러닝 방식 필요

### Chapter.6 차원 축소
1. 주성분 분석(PCA)
    * PCA : 데이터의 분산을 최대한 보존하는 새로운 축을 찾아 변환함으로써 차원을 축소
    * sklearn의 PCA() 클래스 활용 → Sparse 벡터는 Dense 벡터로 바꿔줘야 사용 가능
        - explained_variance_ratio_ 속성으로 보존된 분산의 비중 확인 가능
2. 잠재 의미 분석(LSA, Latent Semantic Analysis)
    * LSA : 문서 및 단어들의 잠재된 의미 분석 가능
        - 의미는 문서와 단어를 연결하는 매개체 → 축소된 차원이 그 역할
        - k개의 축소된 차원은 각각 잠재된 의미를 표현
    * sklearn의 TruncatedSVD() 클래스 활용 → Sparse 벡터도 이용 가능
        - singular_values_ 속성, components_ 속성 등 존재 → np.diag(svd.singular_values_).dot(svd.components_)
3. tSNE를 이용한 시각화와 차원 축소
    * tSNE : 다차원 데이터 사이 거리를 가장 잘 보존하는 2차원 좌표를 찾기 위해 사용(MDS와 유사?)
    * sklearn의 TSNE() 클래스 활용 → TSNE() 클래스는 transform() 메서드 없이 fit_transform()만 존재하는듯
    * LSA를 이용해 차원 축소를 한 후 tSNE를 통해 시각화 하면 Semantic에 따른 분류를 2차원에 표현 가능

### Chapter.7 토픽 모델링으로 주제 찾기
1. 토픽 모델링(Topic Modeling)
    * 다양한 문서 집합에 내재한 토픽을 파악할 때 쓰는 기법 → 예측보다는 내용의 분석 자체가 목적
    * 함께 사용되는 단어의 집합으로 문서에 담긴 토픽을 표현
2. Latent Dirichlet Allocation(LDA)
    * LDA : Latent Topic들을 유추하고자 하는 통계적 방법론
    * LDA의 기본 가정 : 문서를 구성하는 몇 개의 토픽이 존재, 각 토픽은 단어의 집합으로 구성
    * Corpus 속 여러 Topic을 선택하는데 다항분포 필요. Topic 속 여러 단어를 선택하는데 다항분포 필요
        - 다항분포의 Conjugate Prior가 디리클레 분포 → LDA는 디리클레 분포 사용
    * 성능에 대한 척도 : Perplexity 및 Topic Coherence
        - Perplexity : 값이 작을수록 토픽 모델이 문서집합을 잘 반영
        - Topic Coherence : 각 토픽에서 상위 비중을 차지하는 단어들이 의미적으로 유사한지. 값이 클수록 좋음
        - 단, 토픽의 해석이 사람이 보기에 자연스러운 것이 더 중요
3. sklearn과 토픽 모델링
    * sklearn의 LatentDirichletAllocation() 클래스 활용
        - n_components, topic_word_prior, doc_topic_prior, learning_method 등 파라미터 존재
        - topic_word_prior : 토픽의 사전 단어분포를 결정하는 파라미터. default= 1/n_components, 0.1 내외 추천
        - doc_topic_prior : 문서의 사전 토픽분포를 결정하는 파라미터. default= 1/n_components, 1.0 내외 추천
    * lda.components_를 활용하여 토픽의 단어 분포를 보고 토픽의 내용을 짐작
4. Gensim을 이용한 토픽 모델링
    * Dictionary() 클래스를 통해 토큰화 결과로부터 내부적으로 사용하는 id를 매칭하는 사전 생성
        - keep_n, no_below, no_above 파라미터 활용
        - doc2bow(text) 메서드로 카운트 벡터, 즉 BOW 형태로 변환 → Gensim에서는 이 결과를 corpus로 지칭
    * LdaModel() 클래스를 통해 LDA 모델링을 수행
        - corpus, num_topics, id2word, passes, alphs, eta 등 파라미터
        - corpus는 BOW 결과, id2word는 만들어진 사전
        - print_topics() 메서드, get_document_topics(corpus) 등 메서드 활용 가능
    * pyLDAvis 라이브러리로 시각화 가능
    * Perplexity와 Topic Coherence
        - Perplexity : log_perplexity(corpus) 메서드 사용
        - Topic Coherence : CoherenceModel 클래스 사용 → 코드 내 show_coherence 함수 참고
5. 토픽 트렌드(Topic trend)
    * 토픽 트렌드 : 주어진 시간 동안 토픽들의 비중이 어떻게 변화했는지 확인 → 주요 토픽의 추이 분석
        - 각 문서의 날짜와 카운트 벡터를 결합하여 트렌드 확인 → groupby() 메서드를 통해 평균 확인
        - 날짜의 변화에 따라 토픽에 대한 관심도 변화 확인 가능 → 시각화
        - 초기에 비중이 높았다가 줄어드는 토픽은 Cold Topic, 뒤로 가면서 비중이 높아지는 토픽은 Hot Topic
6. 동적 토픽 모델링(Dynamic Topic Modeling)
    * 동적 토픽 모델링 : 최대한 이전 토픽을 반영하여 다음 시간의 토픽을 추출
        - 토픽은 단어의 확률분포로 표현, 시간이 지나면서 토픽의 내용이 바뀔 수 있기 때문에 Dynamic이 필요
    * Gensim의 LdaSeqModel() 클래스 활용
        - corpus, num_topics, id2word, passes, alphs, eta, time_slice 등 파라미터
        - time_slice : 순서대로 정렬된 각 시간 단위에 속한 문서의 수
        - 그런데, LdaMulticore와 달리 멀티코어 개념이 없어서 엄청난 시간 소요.. 이걸 쓸 수 있을까..?

### Chapter.8 감성 분석
1. 감성 분석(Sentiment Analysis)
    * 텍스트에 나타난 의견, 평가, 태도 등 주관적인 정보를 Positive, Neutral, Negative로 분류
    * 어휘 기반(Lexicon-Based)의 분석 및 머신러닝 기반의 분석 존재
    * 감성의 정도를 극성(Polarity)로 표현 → 0이면 Neutral, 양수면 Positive
2. 어휘 기반(Lexicon-Based) 감성 분석
    * 모든 단어에 대해 긍정/부정의 감성을 붙여 감성 사전 구축 → 이를 기반으로 분석 수행
        - 하위 단어로부터 상위 구로 이동하면서 단계적으로 긍정/부정을 결정하는 방식
    * TextBlob 라이브러리를 이용한 감성 분석
        - TextBlob() 클래스에 대하여 sentiment 속성 활용 → polarity와 subjectivity(주관 정도)
        - sentiment_TextBlob(docs)과 같은 함수를 이용하여 처리 → 코드 참고
    * AFINN 라이브러리를 이용한 감성 분석
        - Afinn() 클래스에 대하여 score(text) 메서드 활용
    * VADER 라이브러리를 이용한 감성 분석
        - 소셜 미디어의 텍스트에서 좋은 성능이 나올 수 있도록 개발
        - nltk.sentiment.vader의 SentimentIntensityAnalyzer() 클래스 활용
        - polarity_scores(doc)의 compound 키 활용
    * 정확도가 그리 높지는 않은 것이 문제
3. 머신러닝 기반 감성 분석
    * 카운트 벡터 등에 대하여 target 값을 가지고 문서 분류를 수행하는 형태로 진행
    * imbalanced 데이터인 경우가 많으므로 Precision과 Recall 등을 확인할 필요
    * 한글 데이터의 경우 보통 감성을 가지는 단어는 명사, 동사, 형용사

### Chapter.10 RNN - 딥러닝을 이용한 문서 분류
1. RNN(Recurrent Neural Networks)
    * RNN : 순차적인 영향을 표현하기 위한 모형 → 텍스트 마이닝에서 문맥을 이해하는 것도 이와 유사
        - 단, Vanishing Gradient 문제가 심각하여 현재 많이 쓰이는 것은 아님
        - One-hot 벡터를 Dense 벡터로 변경(Embedding)한 후 입력 크기에 맞게 잘라내어 입력으로 사용
        - (단어의 수, 한 단어를 표현하는 Dense 벡터의 크기)의 2차원 행렬이 입력
        - 마지막 출력 노드를 이용해 문서를 분류
    * keras를 이용해 RNN모델 구축
        - keras가 제공하는 토크나이저를 사용해 모형에 적합한 형태로 입력 데이터를 변환
        - tokenizer 객체 생성 후 fit_on_texts(corpus), texts_to_sequences(corpus) 등의 메서드 활용
        - pad_sequences(X) 함수를 통해 같은 길이를 갖도록 Truncating 및 Padding
        - Embedding 레이어, GRU 레이어, Dense 레이어를 통과
        - model의 compile() 메서드를 이용해 optimizer와 loss 지정 → fit() 메서드로 학습
        - model의 evaluate() 메서드로 평가
2. 워드 임베딩(Word Embedding)
    * 워드 임베딩 : 단어에 대해 One-hot 인코딩 수행 후 다시 축소된 Dense 벡터로 변환하는 과정
        - 단어의 순서를 고려해 문맥 파악 가능
        - BOW에서 문서가 1d 벡터, 워드 임베딩 후에는 2d 행렬 → corpus는 3d tensor가 되는 형태
        - 연산 효율성 증대, 대상 간의 의미적 유사도 계산, 의미적 정보 함축, Transfer Learning 등 가능
        - 학습된 가중치 행렬을 이용해 Dense 벡터로 변경 → 해당 행렬을 이용해 타 문제에 적용

### Chapter.11 Word2Vec, ELMo, Doc2Vec의 이해
1. Word2Vec - 대표적인 워드 임베딩 기법
    * CBOW(Continuous Bag of Words), Skip-Gram 두 가지 학습 방식
    * CBOW : 주변의 단어를 이용해 중심 단어를 예측하도록 학습
    * Skip-Gram : 중심의 한 단어를 이용해 주변 단어를 예측
    * Gensim의 api를 사용하여 기존 학습된 Embedding 벡터 활용 가능
        - similarity(), most_similar(), doesnt_match(), distance() 메서드 등
2. ELMo(Embedding from Language Model) - 문맥에 따른 단어 의미의 구분
    * Word2Vec에서는 Embedding 벡터가 고정, but ELMo는 주어진 문장에 맞게 가변적 벡터 생성
        - 모형 자체를 전이하고 Embedding 벡터는 주어진 문장을 모형에 적용시켜 생성
    * Bi-LSTM을 사용해 Embedding 수행
    * BERT의 중요한 기반. 현재는 BERT에 밀려 활용도 저하
3. Doc2Vec - 문맥을 고려한 문서 임베딩
    * DM(CBOW에 문서 ID 추가) 및 DBOW(Skip-Gram에 문서 ID 추가)

### Chapter.12 CNN - 이미지 분류를 응용한 문서 분류
1. CNN(Convolutional Neural Networks)
    * CNN : 2d matrix로부터 주변 정보를 요약해 이미지를 분류할 수 있는 특성 추출
        - 컬러 이미지의 경우 RGB에 대한 각각 2d matrix → 3d tensor
    * CNN의 주변 정보를 요약하는 점 → 앞뒤 단어들 간 주변 정보 요약으로 문맥 파악
    * keras를 이용해 CNN 모델 구축
        - keras가 제공하는 토크나이저를 사용해 모형에 적합한 형태로 입력 데이터를 변환
        - tokenizer 객체 생성 후 fit_on_texts(corpus), texts_to_sequences(corpus) 등의 메서드 활용
        - pad_sequences(X) 함수를 통해 같은 길이를 갖도록 Truncating 및 Padding
        - Embedding 레이어, Conv1D + Maxpooling1D 레이어, Dense 레이어를 통과
        - Conv1D에서 필터의 수가 몇 단어로 끊어가면서 문맥을 파악하냐의 개념인듯?
        - model의 compile() 메서드를 이용해 optimizer와 loss 지정 → fit() 메서드로 학습
        - model의 evaluate() 메서드로 평가

### Chapter.13 어텐션(Attention)과 트랜스포머(Transformer)
1. Seq2Seq(Sequence to Sequence) : 번역에서 시작된 딥러닝 기법
    * Seq2Seq : 입력으로 일련의 단어들이 들어오고 이를 이용하여 다시 일련의 단어를 생성하는 기법
        - 문장에 대해 이해한 내용을 지정한 형태로 저장하고 거기서 출발 → 워드 임베딩 필수
    * Encoder와 Decoder
        - Encoder는 문장을 이해하는 역할 → 입력은 번역하고자 하는 영어 문장
        - 마지막 <엔드> 입력을 받은 은닉층 노드는 전체의 문맥 정보를 내포
        - Decoder는 이 문맥 정보로부터 번역어 문장을 생성하는 역할 → 각 단어 예측 단계가 순차적으로 실행
2. Attention을 이용한 성능의 향상
    * 문맥 정보가 Encoder의 마지막 벡터 하나에 집중되는 현상을 해결 → Context 벡터 활용
    * Context 벡터가 Encoder의 마지막 벡터와 Decoder의 입력 값과 합쳐져서 단어 생성
    * Self-Attention : 같은 문장 내에서의 Attention → 문장 내에서의 단어 간 영향을 표현
        - 어떤 단어를 벡터로 임베딩할 때 그 단어에 영향을 미치는 다른 단어들 정보를 함께 인코딩
        - 인코딩 과정에서 문맥에 대한 정보는 각 단어에 골고루 분포
3. 트랜스포머(Transformer) : Attention is all you need
    * 기존 Seq2Seq 모델에서 오직 Attention에만 의지한 모형을 제안
    * Encoder에서 Self-Attention 정보를 추출
    * Decoder는 각 단어의 Embedding 벡터를 이용해 단어를 하나씩 예측, 자신의 Self-Attention 정보도 함께 사용
    * 토큰 임베딩으로 시작 + 위치 인코딩
        - Charater-based Tokenization, BPE, WordPiece, SentencePiece 등 다양한 토큰화 사용
        - 입력 시퀀스에서는 전체에 대하여, 출력 시퀀스에서는 현재까지 만들어진 시퀀스에 대하여 단계적으로
    * Encoder 층은 Multi-head Attention과 FeedForward 신경망으로 구성
        - Encoder의 Self-Attention은 query, key, value 세 개의 벡터를 이용하여 계산
        - 나에게 영향을 미치는 단어들의 정보를 결합 → 문맥을 차츰 파악
        - Multi-head Attention은 여러 Self-Attention을 병렬로 연결한 개념
    * Decoder 층은 Masked/Encoder-Decoder Multi-head Attention과 FeedForward 신경망으로 구성
        - Decoder는 Shifted 출력 시퀀스를 입력으로 받는 형태 → 토큰을 하나씩 예측
        - Masked Multi-head Attention : 순방향으로만 Attention이 향하는 것을 구현한 메커니즘
        - Encoder-Decoder Multi-head Attention : query를 던지는 단어는 Decoder에서 생성하는 단어

### Chapter.14 BERT의 이해와 간단한 활용
1. 언어 모델(Language Model)
    * LM : 문장 혹은 단어의 시퀀스에 대해 확률을 할당하는 모델 → 자연스러운 문장에 더 높은 확률을 부여
        - 시퀀스가 나타날 확률은 각 단어들의 결합확률로 표현 → P(output|input) 형태의 조건부 확률의 곱으로 계산
        - LM은 언어에 대한 이해를 높이는 학습. Unsupervised Learning이 가능
    * LM으로 언어에 대한 이해를 높인 후 Fine-tuning Supervised Learning 수행 가능
        - Pre-trained LM : 사전에 LM을 이용하여 미리 학습된 모델 → Transfer Learning 가능
        - 다양한 자연어 처리 문제를 해결할 수 있는 모델로 자연스럽게 확장
2. BERT의 구조
    * BERT : Transformer의 Encoder 부분만 사용한 모형 → 언어에 대한 이해를 높이는 것이 목적
        - Encoder의 Bidirectional Self-Attention 사용
    * Pre-training과 Fine-tuning 단계로 학습 진행
        - Masking을 통하여 LM을 학습 → Masked LM(단어를 가리고 가린 단어를 예측하는 형태)
        - Hugging Face에서 Pre-trained 모델들을 무료로 공개
        - Fine-tuning을 통해 미리 만들어진 가중치들이 목표에 맞게 세밀하게 조정
3. 자동 클래스를 이용한 Tokenizer와 모델
    * AutoTokenizer() 클래스, AutoModelForSequenceClassification() 클래스 활용
        - from_pretrained(prelm) 메서드를 이용하여 객체 설정

### Chapter.15 BERT 사전학습 모형에 대한 미세조정학습
1. BERT 학습을 위한 preprocessing
    * 입력 문장들을 표현하기 위해 세 개의 Embedding 이용
        - Token Embedding(input_ids) : 단어 + 특수 토큰(CLS, SEP)
        - Segment Embedding(token_type_ids) : 문장을 구분. 첫 문장의 끝을 나타내는 [SEP]까지 0, 나머지는 1
        - Position Embedding : BERT 토크나이저가 Position Embedding을 반환하지는 않음
        - 기타(attention_mask) : Self-Attention에의 포함 여부
2. Transformer의 Trainer를 이용한 Fine-tuning
    * torch의 Dataset 클래스를 상속받아 새로운 클래스 생성(OurDataset)
    * load_metric을 이용한 함수 생성을 통해 Trainer를 선언할 때 파라미터로 넘겨줌
    * TrainingArguments() 클래스를 이용한 정의
        - output_dir, num_train_epochs, per_device_train/eval_batch_size 등 파라미터
        - weight_decay 파라미터를 활용한 Overfitting 방지
        - warmup_steps 파라미터를 통한 Learning Rate Scheduler에서의 Warmup 구간 지정
    * Trainer() 클래스를 이용한 정의
        - model, args, train_dataset, compute_metrics 등 파라미터
3. Pytorch를 이용한 Fine-tuning
    * DataLoader() 클래스를 이용하여 batch_learning 작업 수행
    * 원형의 BERT 모델에 직접 Classifier 추가 → BertModel 클래스 활용
    * BERT Pre-trained 모델을 포함하는 NN 모델 선언 → nn.Module 상속
        - 출력 벡터의 크기인 token_size 설정 필요 → output.last_hidden_state[:, 0, :]
4. 한국어 문서에 대한 BERT 활용
    * base_multilingual-cased 모델 혹은 SKTBrain의 KoBERT 활용 가능