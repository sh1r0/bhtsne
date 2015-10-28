#!/usr/bin/env python

import os
import numpy as np
from os.path import abspath, dirname, isfile, join as path_join
from shutil import rmtree
from struct import calcsize, pack, unpack
from subprocess import Popen
from sys import stderr, stdin, stdout
from tempfile import mkdtemp
import argparse

### Constants
BH_TSNE_BIN_PATH = path_join(dirname(__file__), 'bh_tsne')
assert isfile(BH_TSNE_BIN_PATH), ('Unable to find the bh_tsne binary in the '
    'same directory as this script, have you forgotten to compile it?: {}'
    ).format(BH_TSNE_BIN_PATH)
# Default hyper-parameter values from van der Maaten (2013)
DEFAULT_NO_DIMS = 2
DEFAULT_PERPLEXITY = 30.0
DEFAULT_THETA = 0.5
EMPTY_SEED = -1

###

class TmpDir:
    def __enter__(self):
        self._tmp_dir_path = mkdtemp()
        return self._tmp_dir_path

    def __exit__(self, type, value, traceback):
        rmtree(self._tmp_dir_path)


def _read_unpack(fmt, fh):
    return unpack(fmt, fh.read(calcsize(fmt)))

def bh_tsne(samples, no_dims=DEFAULT_NO_DIMS, perplexity=DEFAULT_PERPLEXITY,
        theta=DEFAULT_THETA, randseed=EMPTY_SEED, verbose=False):
    # Assume that the dimensionality of the first sample is representative for
    #   the whole batch
    sample_dim = len(samples[0])
    sample_count = len(samples)

    # bh_tsne works with fixed input and output paths, give it a temporary
    #   directory to work in so we don't clutter the filesystem
    with TmpDir() as tmp_dir_path:
        # Note: The binary format used by bh_tsne is roughly the same as for
        #   vanilla tsne
        with open(path_join(tmp_dir_path, 'data.dat'), 'wb') as data_file:
            # Write the bh_tsne header
            data_file.write(pack('iiddi', sample_count, sample_dim, theta, perplexity, no_dims))
            # Then write the data
            for sample in samples:
                data_file.write(pack('{}d'.format(len(sample)), *sample))
            # Write random seed if specified
            if randseed != EMPTY_SEED:
                data_file.write(pack('i', randseed))

        # Call bh_tsne and let it do its thing
        with open('/dev/null', 'w') as dev_null:
            bh_tsne_p = Popen((abspath(BH_TSNE_BIN_PATH), ), cwd=tmp_dir_path,
                    # bh_tsne is very noisy on stdout, tell it to use stderr
                    #   if it is to print any output
                    stdout=stderr if verbose else dev_null)
            bh_tsne_p.wait()
            assert not bh_tsne_p.returncode, ('ERROR: Call to bh_tsne exited '
                    'with a non-zero return code exit status, please ' +
                    ('enable verbose mode and ' if not verbose else '') +
                    'refer to the bh_tsne output for further details')

        # Read and pass on the results
        with open(path_join(tmp_dir_path, 'result.dat'), 'rb') as output_file:
            # The first two integers are just the number of samples and the
            #   dimensionality
            result_samples, result_dims = _read_unpack('ii', output_file)
            # Collect the results, but they may be out of order
            results = [_read_unpack('{}d'.format(result_dims), output_file)
                for _ in xrange(result_samples)]
            # Now collect the landmark data so that we can return the data in
            #   the order it arrived
            results = [(_read_unpack('i', output_file), e) for e in results]
            # Put the results in order and yield it
            results.sort()
            for _, result in results:
                yield result
            # The last piece of data is the cost for each sample, we ignore it
            #read_unpack('{}d'.format(sample_count), output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        help="Input features in npy format. "
    )
    parser.add_argument(
        "output_file",
        help="Output filename. "
    )
    parser.add_argument(
        '-d', '--no_dims',
        type=int,
        default=DEFAULT_NO_DIMS
    )
    parser.add_argument(
        '-p', '--perplexity',
        type=float,
        default=DEFAULT_PERPLEXITY
    )
    # 0.0 for theta is equivalent to vanilla t-SNE
    parser.add_argument(
        '-t', '--theta',
        type=float,
        default=DEFAULT_THETA
    )
    parser.add_argument(
        '-r', '--randseed',
        type=int,
        default=EMPTY_SEED
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true'
    )
    args = parser.parse_args()

    data = np.load(args.input)

    with open(args.output_file, 'w') as output:
        for result in bh_tsne(data, no_dims=args.no_dims,
                perplexity=args.perplexity, theta=args.theta,
                randseed=args.randseed, verbose=args.verbose):
            fmt = ''
            for i in range(1, len(result)):
                fmt = fmt + '{}\t'
            fmt = fmt + '{}\n'
            output.write(fmt.format(*result))

    # perform normalization
    d = np.loadtxt(args.output_file);
    d -= d.min(axis=0);
    d /= d.max(axis=0);
    np.savetxt(args.output_file, d, fmt="%.8f", delimiter="\t")


if __name__ == '__main__':
    exit(main())
