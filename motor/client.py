import socket

UDP_IP = "127.0.0.1"
UDP_PORT = 10000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP


def set_target_degree(degree):
    if not isinstance(degree, float):
        print('Invalid input for motor!')
        return
    if degree > 100 or degree < -100:
        print('Input degree is out of range:' + str(degree))

    sock.sendto(bytes('t ' + str(degree), "utf-8"), (UDP_IP, UDP_PORT))


def set_current_degree(degree):
    if not isinstance(degree, float):
        print('Invalid input for motor!')
        return
    sock.sendto(bytes('t ' + str(degree), "utf-8"), (UDP_IP, UDP_PORT))


if __name__ == '__main__':
    while True:
        inp = input('degree:')
        try:
            set_target_degree(float(inp))
        except Exception as e:
            print(e)
