#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys
import numpy as np
import cv2
import pickle
from os.path import join
# import ConfigParser
import trimesh
from PyQt5 import QtGui, QtWidgets
from opendr.camera import ProjectPoints, Rodrigues
from smplx.body_models import SMPLX
from trimesh import Trimesh

from ui.camera_widget import Ui_CameraWidget

sys.path.append('.')
from ui.gen.main_window import Ui_MainWindow as Ui_MainWindow_Base
from PyQt5 import QtCore
from pyrender.scene import Scene
from pyrender.camera import PerspectiveCamera, IntrinsicsCamera, CustomIntrinsicsCamera
from pyrender.mesh import Mesh
from pyrender import OffscreenRenderer, MetallicRoughnessMaterial, SpotLight


def load_model_params(params_path):
    with open(params_path, 'rb') as f:
        return pickle.load(f)


def load_camera_params(params_path):
    with open(params_path, 'rb') as f:
        return pickle.load(f)


def load_image(im_path):
    return cv2.imread(im_path)


def save_image(im_path, image):
    return cv2.imwrite(im_path, image)


_translate = QtCore.QCoreApplication.translate
MAX_CANVAS_HEIGHT = 2000
MAX_CANVAS_WIDTH = 1500
DATA_FOLDER = 'models/smplx'


class Ui_MainWindow(QtWidgets.QMainWindow, Ui_MainWindow_Base):
    def __init__(self, args):
        super(self.__class__, self).__init__()

        self.save_params_path = args.save_model_params if args.save_model_params is not None\
            else args.fit_result

        # Result retrieve
        with open(args.fit_result, 'rb') as f:
            fitting_results = pickle.load(f, encoding='latin1')
        self.cam_rot = fitting_results['camera_rotation'][0]
        self.cam_trans = fitting_results['camera_translation'][0]

        # Background image setup
        self.base_img = load_image(args.image_path) if args.image_path else None
        if self.base_img is not None:
            self.canvas_size = np.array(self.base_img.shape[:2][::-1], dtype=np.float32)
            if self.canvas_size[1] > MAX_CANVAS_HEIGHT:
                self.canvas_size *= (MAX_CANVAS_HEIGHT / float(self.canvas_size[1]))
            if self.canvas_size[0] > MAX_CANVAS_WIDTH:
                self.canvas_size *= (MAX_CANVAS_WIDTH / float(self.canvas_size[0]))
            self.canvas_size = tuple(self.canvas_size.astype(int))
        else:
            self.canvas_size = None

        # Model setup
        self.model = self._init_model(fitting_results, args.gender)

        # Scene setup
        self.scene = Scene(bg_color=[1.0, 1.0, 1.0])
        self.material = MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='BLEND',
            baseColorFactor=(5.0, 5.0, 5.0, 1.0))
        im_size = self.canvas_size if self.canvas_size is not None else (MAX_CANVAS_WIDTH, MAX_CANVAS_HEIGHT)
        # self.camera = PerspectiveCamera(yfov=np.pi/3.0,
        #                                 aspectRatio=float(self.canvas_size[0])/self.canvas_size[1])
        self.camera = IntrinsicsCamera(fx=5000., fy=5000.,
                                       cx=im_size[0]/2, cy=im_size[1]/2)
        # self.camera = CustomIntrinsicsCamera(fx=5000., fy=5000.,
        #                                cx=im_size[0]/2, cy=im_size[1]/2)

        self.camera_params = {'t': self.cam_trans,
                              'rt': cv2.Rodrigues(self.cam_rot)[0],
                              'c': np.array(im_size).astype(float) / 2,
                              'f': np.array((im_size[0],) * 2) * 1.0,
                              'k': np.zeros(5),
                              'frustum': {'near': self.camera.znear, 'far': self.camera.zfar,
                                          'width': im_size[0], 'height': im_size[1]}}
        camera_pos = np.eye(4)
        # print(self.cam_trans)
        # print(self.camera.get_projection_matrix(im_size[0], im_size[1]))
        # print(self.camera.cx, self.camera.cy)
        camera_pos[:3, :3] = self.cam_rot
        self.cam_trans[0] *= -1
        camera_pos[:3, 3] = self.cam_trans
        slight = SpotLight(color=[1.0, 1.0, 1.0], intensity=10.0)
        slight_pos = np.eye(4)
        slight_pos[:3, 3] = np.array([0, 0, 5])
        self.scene.add(self.camera, pose=camera_pos)
        self.scene.add(slight, pose=slight_pos)
        self.renderer = OffscreenRenderer(viewport_width=im_size[0], viewport_height=im_size[1], point_size=1.0)

        # Rest setup
        self.setupUi(self)
        self._moving = False
        self._rotating = False
        self._mouse_begin_pos = None
        self._loaded_gender = None
        self._update_canvas = False

        self.camera_widget = Ui_CameraWidget(self.camera, self.camera_params['frustum'], self.draw, self.camera_params)
        self._bind_actions()
        self._update_canvas = True

    def _init_model(self, params, gender):
        model = SMPLX(DATA_FOLDER, gender=gender, use_pca=False,
                      body_pose=params['body_pose'],
                      betas=params['betas'],
                      left_hand_pose=params['left_hand_pose'],
                      right_hand_pose=params['right_hand_pose'],
                      leye_pose=params['leye_pose'],
                      reye_pose=params['reye_pose'],
                      expression=params['expression']
                      )

        return model

    def _bind_actions(self):
        self.btn_camera.clicked.connect(lambda: self._show_camera_widget())

        for i, val in enumerate(self.model.betas[0]):
            self.__dict__['shape_{}'.format(i)].setProperty('value', int(10 * val + 50))

        for key, shape in self._shapes():
            shape.valueChanged[int].connect(lambda val, k=key: self._update_shape(k, val))
        self.set_body_layout()

        self.pos_0.valueChanged[float].connect(lambda val: self._update_position(0, val))
        self.pos_1.valueChanged[float].connect(lambda val: self._update_position(1, val))
        self.pos_2.valueChanged[float].connect(lambda val: self._update_position(2, val))

        self.reset_pose.clicked.connect(self._reset_pose)
        self.reset_shape.clicked.connect(self._reset_shape)
        self.reset_postion.clicked.connect(self._reset_position)

        self.to_body.clicked.connect(self.set_body_layout)
        self.to_left_hand.clicked.connect(self.set_left_layout)
        self.to_right_hand.clicked.connect(self.set_right_layout)
        self.save_params.clicked.connect(self._save_params)

        self.canvas.wheelEvent = self._zoom
        self.canvas.mousePressEvent = self._mouse_begin
        self.canvas.mouseMoveEvent = self._move
        self.canvas.mouseReleaseEvent = self._mouse_end

        self.view_joints.triggered.connect(self.draw)
        self.view_joint_ids.triggered.connect(self.draw)
        self.view_bones.triggered.connect(self.draw)

        self.pos_0.setValue(self.model.transl[0][0])
        self.pos_1.setValue(self.model.transl[0][1])
        self.pos_2.setValue(self.model.transl[0][2])

    def set_body_layout(self):
        for key, pose in self._poses():
            try:
                pose.valueChanged[int].disconnect()
            except Exception as e:
                pass

        for i, val in enumerate(self.model.global_orient.detach().numpy()[0]):
            value = int(50 * val / np.pi + 50)
            self.__dict__['pose_{}'.format(i)].setProperty('value', value)

        for i, val in enumerate(self.model.body_pose.detach().numpy()[0]):
            value = int(50 * val / np.pi + 50)
            self.__dict__['pose_{}'.format(i+3)].setProperty('value', value)

        for i, val in enumerate(self.model.jaw_pose.detach().numpy()[0]):
            value = int(50 * val / np.pi + 50)
            self.__dict__['pose_{}'.format(i+66)].setProperty('value', value)

        for key, pose in self._poses():
            pose.valueChanged[int].connect(lambda val, k=key: self._update_pose_body(k, val))

        self.label_3.setText(_translate("MainWindow", "Global Orientation"))
        self.label_4.setText(_translate("MainWindow", "Hip L"))
        self.label_5.setText(_translate("MainWindow", "Hip R"))
        self.label_6.setText(_translate("MainWindow", "Lower Spine"))
        self.label_7.setText(_translate("MainWindow", "Knee L"))
        self.label_8.setText(_translate("MainWindow", "Knee R"))
        self.label_9.setText(_translate("MainWindow", "Mid Spine"))
        self.label_10.setText(_translate("MainWindow", "Ankle L"))
        self.label_11.setText(_translate("MainWindow", "Ankle R"))
        self.label_12.setText(_translate("MainWindow", "Upper Spine"))
        self.label_13.setText(_translate("MainWindow", "Toes L"))
        self.label_14.setText(_translate("MainWindow", "Toes R"))
        self.label_15.setText(_translate("MainWindow", "Neck"))
        self.label_16.setText(_translate("MainWindow", "Clavicle L"))
        self.label_17.setText(_translate("MainWindow", "Clavicle R"))
        self.label_18.setText(_translate("MainWindow", "Head"))
        self.label_19.setText(_translate("MainWindow", "Shoulder L"))
        self.label_20.setText(_translate("MainWindow", "Shoulder R"))
        self.label_21.setText(_translate("MainWindow", "Elbow L"))
        self.label_22.setText(_translate("MainWindow", "Elbow R"))
        self.label_23.setText(_translate("MainWindow", "Wrist L"))
        self.label_24.setText(_translate("MainWindow", "Wrist R"))
        self.label_25.setText(_translate("MainWindow", "Jaw"))
        self.label_26.setText(_translate("MainWindow", "-"))

    def set_left_layout(self):
        for key, pose in self._poses():
            try:
                pose.valueChanged[int].disconnect()
            except Exception as e:
                pass
        for i, val in enumerate(self.model.left_hand_pose.detach().numpy()[0]):
            value = int(50 * val / np.pi + 50)
            self.__dict__['pose_{}'.format(i)].setProperty('value', value)

        for i, val in enumerate(self.model.leye_pose.detach().numpy()[0]):
            value = int(50 * val / np.pi + 50)
            self.__dict__['pose_{}'.format(i+45)].setProperty('value', value)

        for key, pose in self._poses():
            pose.valueChanged[int].connect(lambda val, k=key: self._update_pose_left(k, val))

        self.label_3.setText(_translate("MainWindow", "Index1 L"))
        self.label_4.setText(_translate("MainWindow", "Index2 L"))
        self.label_5.setText(_translate("MainWindow", "Index3 L"))
        self.label_6.setText(_translate("MainWindow", "Middle1 L"))
        self.label_7.setText(_translate("MainWindow", "Middle2 L"))
        self.label_8.setText(_translate("MainWindow", "Middle3 L"))
        self.label_9.setText(_translate("MainWindow", "Little1 L"))
        self.label_10.setText(_translate("MainWindow", "Little2 L"))
        self.label_11.setText(_translate("MainWindow", "Little3 L"))
        self.label_12.setText(_translate("MainWindow", "Ring1 L"))
        self.label_13.setText(_translate("MainWindow", "Ring2 L"))
        self.label_14.setText(_translate("MainWindow", "Ring3 L"))
        self.label_15.setText(_translate("MainWindow", "Thumb1 L"))
        self.label_16.setText(_translate("MainWindow", "Thumb2 L"))
        self.label_17.setText(_translate("MainWindow", "Thumb3 L"))
        self.label_18.setText(_translate("MainWindow", "Eye Left"))
        self.label_19.setText(_translate("MainWindow", "-"))
        self.label_20.setText(_translate("MainWindow", "-"))
        self.label_21.setText(_translate("MainWindow", "-"))
        self.label_22.setText(_translate("MainWindow", "-"))
        self.label_23.setText(_translate("MainWindow", "-"))
        self.label_24.setText(_translate("MainWindow", "-"))
        self.label_25.setText(_translate("MainWindow", "-"))
        self.label_26.setText(_translate("MainWindow", "-"))

    def set_right_layout(self):
        for key, pose in self._poses():
            try:
                pose.valueChanged[int].disconnect()
            except Exception as e:
                pass
        for i, val in enumerate(self.model.right_hand_pose.detach().numpy()[0]):
            value = int(50 * val / np.pi + 50)
            self.__dict__['pose_{}'.format(i)].setProperty('value', value)

        for i, val in enumerate(self.model.reye_pose.detach().numpy()[0]):
            value = int(50 * val / np.pi + 50)
            self.__dict__['pose_{}'.format(i+45)].setProperty('value', value)

        for key, pose in self._poses():
            pose.valueChanged[int].connect(lambda val, k=key: self._update_pose_right(k, val))

        self.label_3.setText(_translate("MainWindow", "Index1 R"))
        self.label_4.setText(_translate("MainWindow", "Index2 R"))
        self.label_5.setText(_translate("MainWindow", "Index3 R"))
        self.label_6.setText(_translate("MainWindow", "Middle1 R"))
        self.label_7.setText(_translate("MainWindow", "Middle2 R"))
        self.label_8.setText(_translate("MainWindow", "Middle3 R"))
        self.label_9.setText(_translate("MainWindow", "Little1 R"))
        self.label_10.setText(_translate("MainWindow", "Little2 R"))
        self.label_11.setText(_translate("MainWindow", "Little3 R"))
        self.label_12.setText(_translate("MainWindow", "Ring1 R"))
        self.label_13.setText(_translate("MainWindow", "Ring2 R"))
        self.label_14.setText(_translate("MainWindow", "Ring3 R"))
        self.label_15.setText(_translate("MainWindow", "Thumb1 R"))
        self.label_16.setText(_translate("MainWindow", "Thumb2 R"))
        self.label_17.setText(_translate("MainWindow", "Thumb3 R"))
        self.label_18.setText(_translate("MainWindow", "Eye Left"))
        self.label_19.setText(_translate("MainWindow", "-"))
        self.label_20.setText(_translate("MainWindow", "-"))
        self.label_21.setText(_translate("MainWindow", "-"))
        self.label_22.setText(_translate("MainWindow", "-"))
        self.label_23.setText(_translate("MainWindow", "-"))
        self.label_24.setText(_translate("MainWindow", "-"))
        self.label_25.setText(_translate("MainWindow", "-"))
        self.label_26.setText(_translate("MainWindow", "-"))

    def _save_params(self):
        params = {
            'camera_rotation': np.expand_dims(self.cam_rot, axis=0),
            'camera_translation': np.expand_dims(self.cam_trans, axis=0),
            'body_pose': self.model.body_pose.detach().numpy(),
            'left_hand_pose': self.model.left_hand_pose.detach().numpy(),
            'right_hand_pose': self.model.right_hand_pose.detach().numpy(),
            'jaw_pose': self.model.jaw_pose.detach().numpy(),
            'reye_pose': self.model.reye_pose.detach().numpy(),
            'leye_pose': self.model.leye_pose.detach().numpy(),
            'expression': self.model.expression.detach().numpy(),
            'global_orient': self.model.global_orient.detach().numpy(),
            'betas': self.model.betas.detach().numpy()
        }
        with open(self.save_params_path, 'wb') as f:
            pickle.dump(params, f, protocol=2)

    def showEvent(self, event):
        self._init_camera()
        super(self.__class__, self).showEvent(event)

    def resizeEvent(self, event):
        self._init_camera()
        super(self.__class__, self).resizeEvent(event)

    def closeEvent(self, event):
        self.camera_widget.close()
        super(self.__class__, self).closeEvent(event)

    def draw(self):
        if self._update_canvas:
            img = self._overlay_image()
            if self.canvas_size is not None:
                img = cv2.resize(img, self.canvas_size)
            self.canvas.setScaledContents(False)
            self.canvas.setPixmap(self._to_pixmap(img))

    def _init_camera(self):
        self.draw()

    def _overlay_image(self):
        for node in self.scene.get_nodes():
            if node.name == 'body_mesh':
                self.scene.remove_node(node)
                break
        rectified_orient = self.model.global_orient.detach().clone()
        rectified_orient[0][0] += np.pi
        model_obj = self.model(global_orient=rectified_orient)
        out_mesh = Trimesh(model_obj.vertices.detach().numpy().squeeze(), self.model.faces.squeeze())
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        out_mesh.apply_transform(rot)
        mesh = Mesh.from_trimesh(out_mesh, material=self.material)

        self.scene.add(mesh, name='body_mesh')
        rendered, _ = self.renderer.render(self.scene)

        if self.base_img is not None:
            img = self.base_img.copy()
        else:
            img = np.uint8(rendered)
        img_mask = (rendered != 255.0).sum(axis=2) == 3

        img[img_mask] = rendered[img_mask].astype(np.uint8)
        img = np.concatenate((img, img_mask[:, :, np.newaxis].astype(np.uint8)), axis=2)
        return img

    def _zoom(self, event):
        delta = -event.angleDelta().y() / 1200.0
        self.camera_widget.pos_2.setValue(self.camera_widget.pos_2.value() + delta)

    def _mouse_begin(self, event):
        if event.button() == 1:
            self._moving = True
        elif event.button() == 2:
            self._rotating = True
        self._mouse_begin_pos = event.pos()

    def _mouse_end(self, event):
        self._moving = False
        self._rotating = False

    def _move(self, event):
        if self._moving:
            delta = event.pos() - self._mouse_begin_pos
            self.camera_widget.pos_0.setValue(self.camera_widget.pos_0.value() + delta.x() / 500.)
            self.camera_widget.pos_1.setValue(self.camera_widget.pos_1.value() + delta.y() / 500.)
            self._mouse_begin_pos = event.pos()
        elif self._rotating:
            delta = event.pos() - self._mouse_begin_pos
            self.camera_widget.rot_0.setValue(self.camera_widget.rot_0.value() + delta.y() / 300.)
            self.camera_widget.rot_1.setValue(self.camera_widget.rot_1.value() - delta.x() / 300.)
            self._mouse_begin_pos = event.pos()

    def _show_camera_widget(self):
        self.camera_widget.show()
        self.camera_widget.raise_()

    def _update_shape(self, id, val):
        val = (val - 50) / 50.0 * 5.0
        self.model.betas[0][id] = val
        self.draw()

    def _reset_shape(self):
        self._update_canvas = False
        for key, shape in self._shapes():
            # value = self.model_params['betas'][key] / 5.0 * 50.0 + 50
            shape.setValue(50)
        self._update_canvas = True
        self.draw()

    def _update_pose_body(self, id, val):
        val = (val - 50) / 50.0 * np.pi

        if id < 3:
            self.model.global_orient[0, id] = val
        elif id > 65:
            if id < 69:
                self.model.jaw_pose[0, id-66] = val
        else:
            self.model.body_pose[0, id-3] = val
        self.draw()

    def _update_pose_left(self, id, val):
        val = (val - 50) / 50.0 * np.pi

        if id < 45:
            self.model.left_hand_pose[0][id] = val
        elif id > 44 and id < 48:
            self.model.leye_pose[0][id-45] = val
        self.draw()

    def _update_pose_right(self, id, val):
        val = (val - 50) / 50.0 * np.pi

        if id < 45:
            self.model.right_hand_pose[0][id] = val
        elif id > 44 and id < 48:
            self.model.reye_pose[0][id-45] = val
        self.draw()

    def _reset_pose(self):
        self._update_canvas = False
        for key, pose in self._poses():
            pose.setValue(50)
        self._update_canvas = True
        self.draw()

    def _update_position(self, id, val):
        self.model.transl[0][id] = val
        self.draw()

    def _reset_position(self):
        self._update_canvas = False
        self.pos_0.setValue(self.model.transl[0][0])
        self.pos_1.setValue(self.model.transl[0][1])
        self.pos_2.setValue(self.model.transl[0][2])
        self._update_canvas = True
        self.draw()

    def _poses(self):
        return enumerate([self.__dict__['pose_{}'.format(i)] for i in range(72)])

    def _shapes(self):
        return enumerate([self.__dict__['shape_{}'.format(i)] for i in range(10)])

    @staticmethod
    def _to_pixmap(im):
        if im.dtype == np.float32 or im.dtype == np.float64:
            im = np.uint8(im * 255)

        if len(im.shape) < 3 or im.shape[-1] == 1:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        qimg = QtGui.QImage(im, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888)

        return QtGui.QPixmap(qimg)
