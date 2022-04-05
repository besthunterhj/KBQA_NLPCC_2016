import xml.etree.ElementTree as ET


def parse_xml(web_data):
    if len(web_data) == 0:
        # get nothing from the server
        return None

    # parse the xml to string
    xml_data = ET.fromstring(web_data)
    msg_type = xml_data.find("MsgType").text

    return TextMsg(xml_data)


class Msg(object):
    def __init__(self, xml_data):
        self.ToUserName = xml_data.find("ToUserName").text
        self.FromUserName = xml_data.find("FromUserName").text
        self.CreateTime = xml_data.find("CreateTime").text
        self.MsgType = xml_data.find('MsgType').text
        self.MsgId = xml_data.find('MsgId').text


class TextMsg():
    def __init__(self, xml_data):
        self.ToUserName = xml_data.find("ToUserName").text
        self.FromUserName = xml_data.find("FromUserName").text
        self.CreateTime = xml_data.find("CreateTime").text
        self.MsgType = xml_data.find('MsgType').text
        self.MsgId = xml_data.find('MsgId').text
        self.Content = xml_data.find("Content").text.encode("utf-8")


class ImageMsg(Msg):
    def __init__(self, xml_data):
        super().__init__(xml_data)
        self.PicUrl = xml_data.find("PicUrl").text
        self.MediaId = xml_data.find("MediaId").text
