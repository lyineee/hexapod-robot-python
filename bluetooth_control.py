import bluetooth
import time

class Bluez(object):
    def __init__(self):
        self.bd_addr = "20:17:04:24:78:09"
        self.port = 1
        self.sock=bluetooth.BluetoothSocket( bluetooth.RFCOMM )

    def connect(self,retry):
        assert(retry>0)
        while retry>0:
            try:
                self.sock.connect((self.bd_addr, self.port))
            except:
                print('failed')
            else:
                return
            finally:
                retry-=1
                time.sleep(1)
        raise Exception('connection fail because try too many times')

    def send(self,data):
        self.sock.send(str(data))    
    
    def end(self):
        self.sock.close()

if __name__ == "__main__":
    bluez=Bluez()
    bluez.connect(3)
    bluez.end()
