#!/usr/bin/env python

import numpy as np
import cv2
import argparse

COLORS = [(0, 0, 0xFF), (0, 0xFF, 0), (0xFF, 0xFF, 0), (0, 0xFF, 0xFF), (0xFF, 0, 0),
          (0xFF, 0, 0xFF), (0, 0, 0x99), (0, 0x99, 0x99), (0, 0x99, 0), (0x99, 0x99, 0)]

# form D3
D3_10_COLORS = [
    (0xb4, 0x77, 0x1f),
    (0x0e, 0x7f, 0xff),
    (0x2c, 0xa0, 0x2c),
    (0x28, 0x27, 0xd6),
    (0xbd, 0x67, 0x94),
    (0x4b, 0x56, 0x8c),
    (0xc2, 0x77, 0xe3),
    (0x7f, 0x7f, 0x7f),
    (0x22, 0xbd, 0xbc),
    (0xcf, 0xbe, 0x17),
]

D3_20_COLORS = [
    (0xb4, 0x77, 0x1f),
    (0xe8, 0xc7, 0xae),
    (0x0e, 0x7f, 0xff),
    (0x78, 0xbb, 0xff),
    (0x2c, 0xa0, 0x2c),
    (0x8a, 0xdf, 0x98),
    (0x28, 0x27, 0xd6),
    (0x96, 0x98, 0xff),
    (0xbd, 0x67, 0x94),
    (0xd5, 0xb0, 0xc5),
    (0x4b, 0x56, 0x8c),
    (0x94, 0x9c, 0xc4),
    (0xc2, 0x77, 0xe3),
    (0xd2, 0xb6, 0xf7),
    (0x7f, 0x7f, 0x7f),
    (0xc7, 0xc7, 0xc7),
    (0x22, 0xbd, 0xbc),
    (0x8d, 0xdb, 0xdb),
    (0xcf, 0xbe, 0x17),
    (0xe5, 0xda, 0x9e)]


def create_embed(imgs, x, labels, S=2000, s=50, d=None, colors=D3_20_COLORS):
    assert not (S % s)

    v = None
    if d is not None:
        thres_d = 100 # tbd
        h = d < thres_d
        v = np.sum(h, axis=1) # density score

    N = len(imgs)
    '''
    G = np.zeros((S, S, 3), dtype=np.uint8)
    '''
    G = np.empty((S, S, 3), dtype=np.uint8)
    G.fill(255)
    # '''
    border = 1 # border

    for i in xrange(0, N):
        if not ((i+1) % 100):
            print '%d/%d...' % (i+1, N)

        if labels[i] < 0:
            continue

        # location
        '''
        a = int((x[i, 0] * (S - s)) // s) * s
        b = int((x[i, 1] * (S - s)) // s) * s

        if np.count_nonzero(G[a, b]):
            continue # spot already filled
        '''
        a = int(x[i, 0] * (S - s))
        b = int(x[i, 1] * (S - s))

        I = cv2.imread(imgs[i], 1) # force image to be loaded as 24-bit
        I = cv2.resize(I, (s, s))

        cv2.rectangle(I, (0,0), (s-1,s-1), colors[labels[i]], border) # frame -- source indicator
        # cv2.putText(I, str(i), (1, s-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors[labels[i]], 1, cv2.CV_AA) # id
        if v is not None:
            cv2.putText(I, str(v[i]), (1, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors[labels[i]], 1, cv2.CV_AA)

        G[a:a+s, b:b+s] = I

    if v is not None:
        print np.mean(v)
        print np.std(v)
        print np.median(v)
        print np.max(v)

    return G


def create_full_embed(imgs, x, labels, S=2000, s=50, colors=D3_20_COLORS):
    assert not (S % s)

    N = len(imgs)
    G = np.zeros((S, S, 3), dtype=np.uint8)

    xnum = S/s
    ynum = S/s
    assert xnum * ynum <= N

    used = np.zeros(N, dtype=bool)

    qq = len(xrange(0, S, s))
    abes = np.zeros((qq**2, 2))
    i = 0
    for a in xrange(0, S, s):
        for b in xrange(0, S, s):
            abes[i] = [a, b]
            i += 1

    for i in xrange(0, abes.shape[0]):
        if not ((i+1) % 100):
            print '%d/%d...' % (i+1, abes.shape[0])

        if labels[di] < 0:
            continue

        a = abes[i, 0]
        b = abes[i, 1]
        xf = a / S
        yf = b / S
        dd = np.sum(np.power(x - [xf, yf], 2), axis=1)
        dd[used] = np.inf
        di = np.argmin(dd)

        used[di] = True
        I = cv2.imread(imgs[di], 1) # force image to be loaded as 24-bit
        I = cv2.resize(I, (s, s))

        cv2.rectangle(I, (0,0), (s-1,s-1), colors[labels[di]], 1)

        G[a:a+s, b:b+s] = I

    return G


def draw_dots(x, labels, S=2000, pad=24, radius=3, colors=D3_20_COLORS):
    canvas_size = S + pad * 2
    x = np.floor(x * S).astype(int) + pad
    '''
    G = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    '''
    G = np.empty((canvas_size, canvas_size, 3), dtype=np.uint8)
    G.fill(255)
    # '''
    for i in xrange(0, x.shape[0]):
        cv2.circle(G, (x[i, 1], x[i, 0]), radius, colors[labels[i]], -1)

    return G


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mapped_x",
        help="Projected coordinates. "
    )
    parser.add_argument(
        "labels",
        help="Corresponding labels. "
    )
    parser.add_argument(
        "-O", "--output_prefix",
        default='cnn',
        help="Filename prefix of output images. "
    )
    args = parser.parse_args()

    x = np.loadtxt(args.mapped_x)
    labels = np.load(args.labels).astype(int)

    S, s = 2000, 40
    dots = draw_dots(x, labels, S=1000, pad=12, radius=3, colors=D3_10_COLORS)
    # embed = create_embed(imgs, x, labels, S=S, s=s, d=d)
    # full_embed = create_full_embed(imgs, x, labels)

    cv2.imwrite('{0}_dots.png'.format(args.output_prefix), dots)
    # cv2.imwrite('{0}_embed_{1}.png'.format(args.output_prefix, S), embed)
    # cv2.imwrite('{0}_embed_full_{1}.png'.format(args.output_prefix, S), full_embed)


if __name__ == '__main__':
    main()
