#!/usr/bin/env python
# encoding: utf8
from populations import CreatePopulations
from conn_rules import ConnectAll
import logging
log = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    pops = CreatePopulations()
    ConnectAll(pops)
