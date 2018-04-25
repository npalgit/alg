class Borg(object):
    __shared_state={} #class level property
    def __init__(self):
        self.__dict__=self.__shared_state # copy currrent state dict when creating a new instance
        self.state='Init'
    def __str__(self):
        return self.state
class YourBorg(Borg):
    pass

rm1=Borg()
rm2=Borg()
rm1.state='Idle'
rm2.state='Running'
rm2.state='Zombie'
rm3=YourBorg()
