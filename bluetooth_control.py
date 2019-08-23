import bluetooth
import time

class Bluez(object):
    def __init__(self):
        self.bd_addr = "20:17:04:24:78:09"
        self.port = 1
        self.is_bluetooth_connected = False

    def connect(self,retry):
        self.sock=bluetooth.BluetoothSocket( bluetooth.RFCOMM )
        self.sock.settimeout(3)
        assert(retry>0)
        while retry>0:
            try:
                self.sock.connect((self.bd_addr, self.port))
            except:
                self.sock.close()
                print('failed')
            else:
                self.is_bluetooth_connected=True
                return
            finally:
                retry-=1
                time.sleep(1)
        raise Exception('connection fail because try too many times')

    def send(self,data):
        if self.is_bluetooth_connected:
            try:
                self.sock.send(str(data))  
            except:
                self.is_bluetooth_connected = False
                self.sock.close()
                raise
        else:
            raise EnvironmentError('no connection')
          
    
    def end(self):
        self.sock.close()

if __name__ == "__main__":
    bluez=Bluez()
    bluez.connect(3)
    bluez.end()
