class GreekGetter(object):
    def __init__(self):
        self.trans=dict(dog='D',cat='C')
    def get(self,msgid):
        return self.trans.get(msgid,str(msgid))

def EnglishGetter(object):
    def get(self,msgid):
        return str(msgid)

#dict of factory
def get_localizer(language="English"):
    languages= dict(English=EnglishGetter, Greek=GreekGetter)
    return language[language]()

#get factory by product
e,g=get_localizer(language='English'), get_localizer(language='Greek')
for msgid in "dog parrot cat bear".split():
    print(e.get(msgid), g.get(msgid))
