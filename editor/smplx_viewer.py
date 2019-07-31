#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys
from PyQt5 import QtWidgets
from ui.main_window import Ui_MainWindow
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit_result', '-f', type=str, help='Path to a .pkl file with fitting result')
    parser.add_argument('--image_path', '-i', type=str, help='Path to source image, above which to render model')
    parser.add_argument('--gender', '-g', type=str, help='Model gender', choices=['male', 'female', 'neutral'],
                        default='neutral', required=False)
    # parser.add_argument('--model', '-m', type=str, help='Model name')
    # parser.add_argument('--camera_params', '-c', type=str, help='Path to camera params')
    # parser.add_argument('--texture_path', '-t', type=str, help='Path to the texture')
    parser.add_argument('--save_model_params', '-s', type=str, help='Path to save updated model params into. '
                                                                    'If None, rewrites params',
                        default=None, required=False)
    # parser.add_argument('--save_camera_params', '-p', type=str, help='Path to save updated camera params into')
    # parser.add_argument('--overlay_img_path', '-o', type=str, help='Path to save image with new model over photo')
    # parser.add_argument('--rotation_angle', '-r', type=float, default=0., help='Rotate model around y-axis')
    # parser.add_argument('--particular_pose', '-l', type=str, help='.npy with thetas of particulat pose')
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    form = Ui_MainWindow(args)
    form.show()
    form.raise_()
    sys.exit(app.exec_())
