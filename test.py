from keycontrol import KeyControl
import threading

def main():
    global STATE 
    STATE=0

    th=threading.Thread(target=thread_loop)

    th.start()
    while True:
        print(STATE)


def thread_loop():
    global STATE
    key=KeyControl()
    key.start()
    while True:
        state=key.get_key()
        if state is not None:
            STATE=state

if __name__ == "__main__":
    main()