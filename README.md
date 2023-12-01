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