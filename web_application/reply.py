import time

class Msg(object):
    def __init__(self):
        pass

    def send(self):
        return "success"


class TextMsg(Msg):
    def __init__(self, toUserName, fromUserName, content):
        super().__init__()
        self.__dict = dict()
        self.__dict["ToUserName"] = toUserName
        self.__dict["FromUserName"] = fromUserName
        self.__dict["CreateTime"] = int(time.time())
        self.__dict["Content"] = content

    def send(self):
        XML_Form = """
            <xml>
                <ToUserName><![CDATA[{ToUserName}]]></ToUserName>
                <FromUserName><![CDATA[{FromUserName}]]></FromUserName>
                <CreateTime>{CreateTime}</CreateTime>
                <MsgType><![CDATA[text]]></MsgType>
                <Content><![CDATA[{Content}]]></Content>
            </xml>
        """
        return XML_Form.format(**self.__dict)
