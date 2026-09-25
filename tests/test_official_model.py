from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from panthera.config import ROOT
from panthera.mesh import stl_to_obj


def rotation_rpy(text):
    r, p, y = map(float, text.split())
    rx = np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
    ry = np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
    rz = np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]])
    return rz @ ry @ rx


def test_official_asset_hashes():
    folder = ROOT/'assets/panthera'
    manifest = json.loads((folder/'provenance.json').read_text())
    for file in manifest['files']:
        assert hashlib.sha256((folder/file['path']).read_bytes()).hexdigest() == file['sha256']


def test_mesh_conversion_preserves_every_triangle(tmp_path):
    source = ROOT/'assets/panthera/meshes/link1.STL'
    target = tmp_path/'link1.obj'
    stl_to_obj(source, target)
    dtype = np.dtype([('n','<f4',(3,)), ('v','<f4',(3,3)), ('a','<u2')])
    official = np.frombuffer(source.read_bytes(), dtype=dtype, offset=84)['v']
    vertices, faces = [], []
    for line in target.read_text().splitlines():
        if line.startswith('v '):
            vertices.append([float(x) for x in line.split()[1:]])
        elif line.startswith('f '):
            faces.append([int(x)-1 for x in line.split()[1:]])
    actual = np.asarray(vertices, dtype=np.float32)[np.asarray(faces)]
    np.testing.assert_array_equal(actual, official)


def test_frames_limits_and_inertias_match_original_urdf(env):
    tree = ET.parse(ROOT/'assets/panthera/panthera_6dof.urdf')
    rng = np.random.default_rng(923)
    for _ in range(6):
        q = rng.uniform(env.lower, env.upper)
        env.data.qpos[env.qids] = q
        mujoco.mj_forward(env.model, env.data)
        base = np.eye(4)
        base[:3,3] = env.cfg['robot']['base_position']
        frames = {'base_link': base}
        values = dict(zip(env.names, q))
        for joint in tree.findall('joint'):
            transform = np.eye(4)
            origin = joint.find('origin')
            transform[:3,3] = [float(x) for x in origin.get('xyz').split()]
            transform[:3,:3] = rotation_rpy(origin.get('rpy'))
            motion = np.eye(4)
            name = joint.get('name')
            if name in values:
                axis = np.array([float(x) for x in joint.find('axis').get('xyz').split()])
                if joint.get('type') == 'revolute':
                    x,y,z = axis
                    skew = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
                    angle = values[name]
                    motion[:3,:3] = np.eye(3)+np.sin(angle)*skew+(1-np.cos(angle))*(skew@skew)
                else:
                    motion[:3,3] = axis*values[name]
                index = env.model.joint(name).id
                limit = joint.find('limit')
                np.testing.assert_array_equal(env.model.jnt_range[index], [float(limit.get('lower')), float(limit.get('upper'))])
            child = joint.find('child').get('link')
            frames[child] = frames[joint.find('parent').get('link')] @ transform @ motion
        for name, frame in frames.items():
            bid = env.model.body(name).id
            np.testing.assert_allclose(env.data.xpos[bid], frame[:3,3], atol=1e-12)
            np.testing.assert_allclose(env.data.xmat[bid].reshape(3,3), frame[:3,:3], atol=1e-12)
    for link in tree.findall('link'):
        inertia = link.find('inertial')
        if inertia is None:
            continue
        bid = env.model.body(link.get('name')).id
        assert env.model.body_mass[bid] == float(inertia.find('mass').get('value'))
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, env.model.body_iquat[bid])
        actual = rot.reshape(3,3) @ np.diag(env.model.body_inertia[bid]) @ rot.reshape(3,3).T
        i = {k:float(v) for k,v in inertia.find('inertia').attrib.items()}
        expected = np.array([[i['ixx'],i['ixy'],i['ixz']],[i['ixy'],i['iyy'],i['iyz']],[i['ixz'],i['iyz'],i['izz']]])
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
