#!/usr/bin/env python
# encoding: utf8
from populations import CreatePopulations
from conn_rules import ConnectAll
import nest
import logging
log = logging.getLogger()
logging.basicConfig(level=logging.DEBUG, filename='mylog.log', filemode='w')



if __name__ == '__main__':
    nest.SetKernelStatus({'local_num_threads': 2})
    pops = CreatePopulations()
    ConnectAll(pops)
