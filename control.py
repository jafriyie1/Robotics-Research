import RPi.GPIO as GPIO
#import get
import time
import termios, fcntl, sys, os


def run_nonblock():
    fd = sys.stdin.fileno()

    oldterm = termios.tcgetattr(fd)
    newattr = termios.tcgetattr(fd)
    newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
    termios.tcsetattr(fd, termios.TCSANOW, newattr)

    oldflags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, oldflags | os.O_NONBLOCK)

    command = ('y')
    try:
        while command != ('x'):
            try:
                command = sys.stdin.read(1)
                #print("Got acter", repr(command))

                if command == ('w'):
                    go_forward()
                elif command == ('a'):
                    go_left()
                elif command == ('d'):
                    go_right()
                elif command == ('s'):
                    go_back()
                time.sleep(0.1)
                stop()
            except IOError: pass
    finally:
        termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
        fcntl.fcntl(fd, fcntl.F_SETFL, oldflags)

def run():
    command = 'y'

    while command != 'x':
        command = input('command: ')
        command = command.rstrip('\n')

        if command == 'w':
            go_forward()
        elif command == 'a':
            go_left()
        elif command == 'd':
            go_right()
        elif command == 's':
            go_back()
        time.sleep(0.1)
        stop()

def stop():
    GPIO.output(7,0)
    GPIO.output(8,0)
    GPIO.output(9,0)
    GPIO.output(10,0)

def go_forward():
    '''turn right motor forward'''
    GPIO.output(9,0)
    GPIO.output(10,1)


    '''turn left motor forward'''
    GPIO.output(7,0)
    GPIO.output(8,1)

def go_back():
    GPIO.output(9,1)
    GPIO.output(10,0)
    GPIO.output(7,1)
    GPIO.output(8,0)

def go_right():
    GPIO.output(9,1)
    GPIO.output(10,0)
    GPIO.output(7,0)
    GPIO.output(8,1)

def go_left():
    GPIO.output(9,0)
    GPIO.output(10,1)
    GPIO.output(7,1)
    GPIO.output(8,0)

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(7, GPIO.OUT)
GPIO.setup(8, GPIO.OUT)
GPIO.setup(9, GPIO.OUT)
GPIO.setup(10, GPIO.OUT)

stop()



run_nonblock()

GPIO.cleanup()
