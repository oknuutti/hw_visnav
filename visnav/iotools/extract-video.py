import os
import argparse
import cv2


def main():
    parser = argparse.ArgumentParser(description='Extract video frames into a given folder')
    parser.add_argument('--video', '-d', metavar='DATA', help='path to a video')
    parser.add_argument('--out', '-t', metavar='META', help='image output path')
    parser.add_argument('--skip', '-s', type=int, default=1, help='use only every xth frame (default: 1)')
    parser.add_argument('--gray', '-g', action='store_true', help='save as grayscale')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(args.video)

    f = 0
    while cap.isOpened():
        ret, img = cap.read()
        if args.gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        f += 1
        if f % args.skip == 0:
            cv2.imwrite(os.path.join(args.out, 'frame-%d.png' % f), img, (cv2.IMWRITE_PNG_COMPRESSION, 9))

    cap.release()


if __name__ == '__main__':
    if 1:
        main()
