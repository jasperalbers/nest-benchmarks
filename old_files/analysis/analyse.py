#!/usr/bin/env python
"""
Usage: analyse [options] <file>...

Options:
    -h --help              Display this text and exit.
    -v --verbose           Be more verbose. Shows debugging messages.
    -p --plot              Plot results.
    -c COLS --cols=COLS    Only show specified columns. Comma-separated list.
    -o FILE --output=FILE  Save figure to file.
"""

import os
import pandas as pd
import numpy as np
from docopt import docopt
import matplotlib.pyplot as plt
import logging
log = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def print_columns(csv, columns):
    pd.set_option('display.max_columns', None)
    if columns:
        print(csv[columns])
    else:
        print(csv)


def plot_columns(csv, columns, output):
    if not columns:
        columns = csv.columns  # ['nodes', 'edges', 'init', 'sim', 'memory']
    num_rows = 2
    layout = (num_rows, int(np.ceil(len(columns)/num_rows)))

    ax = csv[columns].plot(subplots=True, layout=layout,
                      kind='bar', legend=False, colormap='viridis')

    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args['--output']:
        plt.savefig('../figures/{}'.format(args['--output']))

    return ax
    #plt.show()


def main(args):
    frames = []
    for bench in args['<file>']:
        csv = pd.read_csv(bench)
        csv = csv.set_index('SCALE')
        log.debug(csv)

        columns = args['--cols'].split(',') if args['--cols'] else None
        print_columns(csv, columns)
        if args['--plot']:
            frames.append(csv)
            csv = pd.concat(frames)
            plot_columns(csv, columns, args['--output'])

    plt.show()      
    # csv = pd.read_csv(args['<file>'])
    # csv = csv.set_index('SCALE')
    # log.debug(csv)

    # columns = args['--cols'].split(',') if args['--cols'] else None
    # print_columns(csv, columns)
    # if args['--plot']:
    #     plot_columns(csv, columns, args['--output'])


if __name__ == '__main__':
    args = docopt(__doc__)
    if args['--verbose']:
        log.setLevel(logging.DEBUG)
    log.debug(args)
    print(args['<file>'])
    main(args)
