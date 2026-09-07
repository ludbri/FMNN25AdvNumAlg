

# This file contains tags used for inter-process communication
# using tags to communicate whether or not to stop.
# Idea suggested by copilot.


TAG_STOP = 0

# Sending data from main process to a room process
TAG_TO_ROOM = 1
# Sending data from a room process to the main process
TAG_FROM_ROOM = 2

# another tag message to tell a process to send its current value of u (but requires the receiver to know the right shape?).
TAG_SEND_U = 3
TAG_U_FROM_ROOM = 4