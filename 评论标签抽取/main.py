from stanfordcorenlp import StanfordCoreNLP
import re

pattern = re.compile(".*?\\((.*?)-(.*?), (.*?)-(.*?)\\)")

def getWordsUnit(line):
    matcher = pattern.match(line)

if __name__ == '__main__':
    # nlp = StanfordCoreNLP(r'C:\\tools\\stanford-corenlp-full-2018-10-05', lang='zh')
    nlp = StanfordCoreNLP(r'http://127.0.0.1', port=9090, lang='zh')
    sentence = "料子挺好舍友买便宜好多衣服好看回来洗很好"
    # sentence = "衣服面料非常好做工精细贴合皮肤非常舒服洗不变形物美廉价下次光临本店"
    dependencies = nlp.dependency_parse(sentence)
    tokenize = nlp.word_tokenize(sentence)
    sent = ["ROOT"] + tokenize
    my_dependencies = [(dependencie[0], f"{sent[dependencie[1]]}-{str(dependencie[1])}", f"{sent[dependencie[2]]}-{str(dependencie[2])}") for dependencie in dependencies]

    for i, dependency in enumerate(dependencies):
        if dependency[0]=="nsubj":
            if i<len(dependencies)-2 and dependencies[i+1]=="advmod" and dependencies[i+2]=="advmod":
                # 1.处理nsubj + advmod + advmod
                print()
            elif i<len(dependencies)-1 and dependencies[i+1]=="advmod":
                # 2.处理nsubj + advmod
                print()
        elif dependency[0]=="advmod":
            if i<len(dependencies)-1 and dependencies[i+1]=="advmod":
                # 3.处理advmod + advmod
                print()
            elif i<len(dependencies)-1 and dependencies[i+1]=="amod":
                # 4.处理advmod + amod
                print()
            else:
                # 5.处理advmod
                print()

        print()