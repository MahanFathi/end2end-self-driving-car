#include <fcntl.h>
#include <unistd.h>
#include <linux/parport.h>
#include <linux/ppdev.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/io.h>
#include <sys/types.h>

#include <iostream>
#include <thread>
#include <sstream>
#include <cmath>

#include "udp_client_server.h"

#define BASEPORT 0x378 /* lp1 */
using namespace std;
using namespace udp_client_server;

const double step = 360.0 / 6400.0;

double targetDir = 0;
double currentDir = 0;

void moveThread()
{
    int		PortFD;
    PortFD = open("/dev/parport0", O_RDWR);
    ioctl(PortFD, PPCLAIM);
    char data = 0;
    while(1){
        if (fabs(currentDir-targetDir) > 0.1) {
            if (currentDir < targetDir){
                data = 16;
            } else {
                data = 0;
            }
            data |= 4;
            ioctl(PortFD, PPWDATA, &data);
            usleep(500);
            data ^= 4;
            ioctl(PortFD, PPWDATA, &data);
            usleep(500);
            currentDir +=step;
            // cout << "##target dir = " << targetDir << "   current dir = " << currentDir << endl;
        }else{
            usleep(1000); 
        }
    }
}


int main() {
    udp_server *u = new udp_server("127.0.0.1", 10000);
    cout << u->get_addr();
    char buffer[100];
    thread t1(moveThread);
    while(1){
        int size = u->recv(buffer,100);
        buffer[size] = 0;
        cout << buffer << endl;
        if (size > 0)
        {
            istringstream sin(buffer);
            char command;
            sin >> command;
            switch(command)
            {
                case('t'):
                    sin >> targetDir;
                    break;
                case('c'):
                    sin >> currentDir;
                    break;
            }
            cout << "target dir = " << targetDir << "   current dir = " << currentDir << endl;
        }
    }
}