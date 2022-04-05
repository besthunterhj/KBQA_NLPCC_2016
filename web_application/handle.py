import time
import hashlib
import web
import receive
import reply
from KBQA_NLPCC_2016.kbqa_application import automated_qa
from common.model import NER_TOKENIZER, NER_MODEL, QA_TOKENIZER, QA_MODEL, SENTENCEMODEL


# judge the data sent to the server
class Handle(object):
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
                content = automated_qa(question, NER_TOKENIZER, NER_MODEL, QA_TOKENIZER, QA_MODEL, SENTENCEMODEL, "../../data/test.kb")
                replyMsg = reply.TextMsg(toUser, fromUser, content)
                return replyMsg.send()

            else:
                print("Don't send immediately!")
                return "success"

        except Exception as Argument:
            return Argument

# if __name__ == "__main__":
#     start = time.time()
#     test = automated_qa("何俊毅的风格是什么？", NER_TOKENIZER, NER_MODEL, QA_TOKENIZER, QA_MODEL, SENTENCEMODEL,
#                         "../../data/test.kb")
#     end = time.time()
#     print(test)
#     print(end - start)