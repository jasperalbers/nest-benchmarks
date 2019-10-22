#!/usr/bin/env python
# encoding: utf8
from populations_nest3 import CreatePopulations
from conn_rules_nest3 import ConnectAll
import nest
import sys
import logging
log = logging.getLogger()
logging.basicConfig(level=logging.DEBUG, filename='mylog.log', filemode='w')

args = sys.argv

if len(args) > 1:
    scale = int(args[1])
    totVPs = int(args[2])
else:
    scale = 100
    totVPs = 4

print("scale: ", scale)
print("totVPs: ", totVPs)

if __name__ == '__main__':
    nest.SetKernelStatus({'total_num_virtual_procs': totVPs})
    pops = CreatePopulations(scale)
    ConnectAll(pops)
