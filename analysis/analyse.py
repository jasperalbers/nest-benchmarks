#!/usr/bin/env python
"""
Usage: analyse [options] <file>

Options:
    -h --help      Display this text and exit.
    -v --verbose   Be more verbose. Shows debugging messages.
"""

import pandas as pd
import numpy as np
from docopt import docopt
import matplotlib.pyplot as plt
import logging
log = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def main(args):
    csv = pd.read_csv(args['<file>'])
    log.debug(csv)

    plot_columns = ['nodes', 'edges', 'init', 'sim', 'memory']
    num_rows = 2
    layout = (num_rows, int(np.ceil(len(plot_columns)/num_rows)))

    csv[plot_columns].plot(subplots=True, layout=layout,
                           kind='bar', legend=False, colormap='viridis')
    plt.show()


if __name__ == '__main__':
    args = docopt(__doc__)
    if args['--verbose']:
        log.setLevel(logging.DEBUG)
    log.debug(args)
    main(args)
