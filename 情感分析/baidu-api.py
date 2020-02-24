from aip import AipNlp

""" 你的 APPID AK SK """
APP_ID = '10677661'
API_KEY = 'CbBnylD4hQekDeGfmNRvITtL'
SECRET_KEY = 'a1mjmnGiaG9ywtLP0RvbjkyIASUylsL0'

client = AipNlp(APP_ID, API_KEY, SECRET_KEY)

text = "百度是一家高科技公司"

""" 调用词法分析 """
print(client.lexer(text))

options = {}
options["type"] = 13
print(client.commentTag(text, options))

print(client.sentimentClassify(text))

options = {}
options["scene"] = "talk"

""" 带参数调用对话情绪识别接口 """
print(client.emotion(text, options))