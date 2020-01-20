from stanfordcorenlp import StanfordCoreNLP

if __name__ == '__main__':
    # nlp = StanfordCoreNLP(r'C:\\tools\\stanford-corenlp-full-2018-10-05')
    # nlp = StanfordCoreNLP(r'http://127.0.0.1', port=9000)

    # sentence = 'Guangdong University of Foreign Studies is located in Guangzhou.'
    # print ('Tokenize:', nlp.word_tokenize(sentence))
    # print ('Part of Speech:', nlp.pos_tag(sentence))
    # print ('Named Entities:', nlp.ner(sentence))
    # print ('Constituency Parsing:', nlp.parse(sentence))#语法树
    # print ('Dependency Parsing:', nlp.dependency_parse(sentence))#依存句法
    # nlp.close() # Do not forget to close! The backend server will consume a lot memery

    # 中文中的应用，一定记得下载中文jar包，并标志lang=‘zh’
    nlp = StanfordCoreNLP(r'C:\\tools\\stanford-corenlp-full-2018-10-05', lang='zh')
    sentence = '清华大学位于北京。'
    print('Tokenize:', nlp.word_tokenize(sentence))
    print('Part of Speech:', nlp.pos_tag(sentence))
    print('Named Entities:', nlp.ner(sentence))
    print('Constituency Parsing:', nlp.parse(sentence))  # 语法树
    print('Dependency Parsing:', nlp.dependency_parse(sentence))  # 依存句法