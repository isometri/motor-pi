#!/usr/bin/python

import wiringpi as wp
import time

wp.wiringPiSetupGpio()

#configure input and output pins
PWMPin = 18
Standby = 5
AIN1 = 23
AIN2 = 24


wp.pinMode(PWMPin, 2)
wp.pinMode(Standby, 1)
wp.pinMode(AIN1, 1)
wp.pinMode(AIN2, 1)

try:
   while True:
      direction = raw_input("Which direction? Type 'forward' or 'backward' \n")

      type(direction)

      if (direction == 'forward'):
         wp.digitalWrite(Standby, 1)
         wp.digitalWrite(AIN1, 0)
         wp.digitalWrite(AIN2, 1)
         for x in range(0, 800):
            wp.pwmWrite(PWMPin, x) #pwmWrite() takes input values in the range 0 - 1023
            time.sleep(.01)

         wp.digitalWrite(Standby, 0)
         wp.digitalWrite(AIN1, 0)
         wp.digitalWrite(AIN2, 0)
         time.sleep(5)

      if (direction == 'backward'):
         wp.digitalWrite(Standby, 1)
         wp.digitalWrite(AIN1, 1)
         wp.digitalWrite(AIN2, 0)
         for x in range(0, 800):
            wp.pwmWrite(PWMPin, x) #pwmWrite() takes input values in the range 0 - 1023
            time.sleep(.01)

         wp.digitalWrite(Standby, 0)
         wp.digitalWrite(AIN1, 0)
         wp.digitalWrite(AIN2, 0)
         time.sleep(5)
except:
   pass
finally:
   #clean up
   print("\nexiting")
   wp.pinMode(PWMPin, 0)
   exit(0)
