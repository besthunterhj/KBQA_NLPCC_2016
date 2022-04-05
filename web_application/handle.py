import hashlib
import web
import receive
import reply
import application.kbqa_application as kbqa
from transformers import BertForSequenceClassification, BertForTokenClassification, BertTokenizer
from text2vec import Word2Vec

# judge the data sent to the server
class Handle(object):
    def __init__(self):
        self.ner_tokenizer = BertTokenizer.from_pretrained("../../model/bert_ner_model")
        self.qa_tokenzier = BertTokenizer.from_pretrained("../../model/bert_qa_model")
        self.ner_model = BertForTokenClassification.from_pretrained("../../model/bert_ner_model")
        self.qa_model = BertForSequenceClassification.from_pretrained("../../model/bert_qa_model")
        self.w2v = Word2Vec("w2v-light-tencent-chinese")


    def GET(self):
        try:
            # get the data from input
            data = web.input()

            if len(data)  == 0:
                # get nothing from the reply
                return "Hello, this is handle view"

            # start to judge
            signature = data.signature
            timestamp = data.timestamp
            nonce = data.nonce
            echostr = data.echostr
            token = "kbqa4test"

            # init the list which be sent to judge
            para_list = [token, timestamp, nonce]
            print(para_list)
            para_list.sort()
            print(para_list)
            original_text = "".join(para_list)
            print(original_text)

            # get the hashcode
            hashcode = hashlib.sha1(original_text.encode("utf-8")).hexdigest()
            print("handle/GET func: hashcode, signature", hashcode, signature)
            if hashcode == signature:
                return echostr
            else:
                return ""

        except Exception as Argument:
            return Argument

    def POST(self):
        try:
            web_data = web.data()
            print("Handle Post webdata is ", str(web_data, "utf-8"))

            recMsg = receive.parse_xml(web_data)
            if isinstance(recMsg, receive.TextMsg) and recMsg.MsgType == "text":
                toUser = recMsg.FromUserName
                fromUser = recMsg.ToUserName
                question = str(recMsg.Content, "utf-8")
                # content = kbqa.automated_qa(question, , "../../data/test.kb")
                # replyMsg = reply.TextMsg(toUser, fromUser, content)
                # return replyMsg.send()

            else:
                print("Don't send immediately!")
                return "success"

        except Exception as Argument:
            return Argument

# if __name__ == "__main__":
#     test = kbqa.automated_qa("广东外语外贸大学的校庆是什么时候？", "../../model/bert_ner_model", "../../model/bert_qa_model", "../../data/kbqa_data.kb")
#     print(test)