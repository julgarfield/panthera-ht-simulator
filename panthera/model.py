"""Import the official URDF, then add simulation-only scene and servo settings."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import yaml
from .config import resolve, ROOT
from .mesh import stl_to_obj


def numbers(values):
    return ' '.join(format(float(v), '.17g') for v in values)


def rpy_matrix(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])


def quaternion(matrix):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(matrix, dtype=float).ravel())
    return q


def camera_axes(forward, up):
    forward = np.asarray(forward, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-8:
        raise ValueError('Camera forward and up must not be parallel')
    right /= np.linalg.norm(right)
    return np.concatenate([right, np.cross(right, forward)])


def build_model(cfg):
    source = resolve(cfg, cfg['robot']['urdf_path'])
    if not source.is_file():
        raise FileNotFoundError(f'Missing official assets: {source}. Run tools/fetch_assets.py.')
    tree = ET.parse(source)
    arm_names = [f'joint{i}' for i in range(1, 7)]
    limits = {j.get('name'): dict(j.find('limit').attrib) for j in tree.findall('joint') if j.find('limit') is not None}
    if any(j not in limits for j in arm_names):
        raise ValueError('Expected the standard Panthera-HT six-axis URDF')
    build = ROOT / 'build'
    mesh_dir = build / 'meshes'
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for mesh in tree.findall('.//mesh'):
        src = source.parent / 'meshes' / Path(mesh.get('filename')).name
        digest = hashlib.sha256(src.read_bytes()).hexdigest()
        obj = mesh_dir / f'{src.stem}-{digest[:12]}.obj'
        if not obj.exists():
            stl_to_obj(src, obj)
        mesh.set('filename', obj.name)
    compiler = tree.find('mujoco/compiler')
    if compiler is None:
        extension = ET.SubElement(tree.getroot(), 'mujoco')
        compiler = ET.SubElement(extension, 'compiler')
    # Retain every link frame and every URDF inertia; no auto-balanced replacements.
    compiler.attrib.update(meshdir=str(mesh_dir), discardvisual='false', fusestatic='false', balanceinertia='false')
    normalized = build / 'normalized.urdf'
    tree.write(normalized, encoding='utf-8', xml_declaration=True)
    imported = mujoco.MjModel.from_xml_path(str(normalized))
    imported_xml = build / 'imported.xml'
    mujoco.mj_saveLastXML(str(imported_xml), imported)
    scene = ET.parse(imported_xml)
    root = scene.getroot()
    root.find('compiler').set('meshdir', str(mesh_dir))
    # mj_saveLastXML prints limited decimal precision. Restore source numbers,
    # including the complete inertia tensor, before the final compilation.
    links = {link.get('name'): link for link in tree.findall('link')}
    for body in root.findall('.//body'):
        link = links[body.get('name')]
        inertia = link.find('inertial')
        if inertia is not None:
            old = body.find('inertial')
            if old is not None:
                body.remove(old)
            i = inertia.find('inertia').attrib
            tensor = np.array([[float(i['ixx']), float(i['ixy']), float(i['ixz'])],
                               [float(i['ixy']), float(i['iyy']), float(i['iyz'])],
                               [float(i['ixz']), float(i['iyz']), float(i['izz'])]])
            origin = inertia.find('origin')
            rotation = rpy_matrix([float(x) for x in origin.get('rpy', '0 0 0').split()])
            tensor = rotation @ tensor @ rotation.T
            # Use a double-precision eigensolver; MuJoCo's fast fullinertia
            # diagonalization otherwise loses several digits on tiny cross terms.
            eigenvalues, eigenvectors = np.linalg.eigh(tensor)
            if np.linalg.det(eigenvectors) < 0:
                eigenvectors[:, 0] *= -1
            ET.SubElement(body, 'inertial', pos=origin.get('xyz'), mass=inertia.find('mass').get('value'),
                          diaginertia=numbers(eigenvalues), quat=numbers(quaternion(eigenvectors)))
        for geom, source_geom in zip(body.findall('geom'), [*link.findall('visual'), *link.findall('collision')]):
            origin = source_geom.find('origin')
            geom.set('pos', origin.get('xyz', '0 0 0'))
            geom.set('quat', numbers(quaternion(rpy_matrix([float(x) for x in origin.get('rpy', '0 0 0').split()]))))
            # Hide duplicate collision geometry in group 3 without disabling contact.
            if source_geom.tag == 'collision':
                geom.set('group', '3')
    for joint in tree.findall('joint'):
        body = root.find(f".//body[@name='{joint.find('child').get('link')}']")
        origin = joint.find('origin')
        body.set('pos', origin.get('xyz', '0 0 0'))
        body.set('quat', numbers(quaternion(rpy_matrix([float(x) for x in origin.get('rpy', '0 0 0').split()]))))
        mjjoint = body.find('joint')
        if mjjoint is not None:
            mjjoint.set('axis', joint.find('axis').get('xyz'))
            limit = joint.find('limit')
            mjjoint.set('range', f"{limit.get('lower')} {limit.get('upper')}")
    ET.SubElement(root, 'option', timestep=str(1/cfg['simulation']['physics_hz']),
                  gravity=numbers(cfg['simulation']['gravity']), integrator='implicitfast', iterations='80',
                  cone='elliptic', impratio='10', noslip_iterations='3')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth=str(max(cfg[n]['width'] for n in ('table_camera', 'wrist_camera'))),
                  offheight=str(max(cfg[n]['height'] for n in ('table_camera', 'wrist_camera'))))
    ET.SubElement(visual, 'quality', offsamples='0', shadowsize='1024')
    ET.SubElement(visual, 'headlight', ambient='0.35 0.35 0.35', diffuse='0.65 0.65 0.65')
    world = root.find('worldbody')
    base = world.find("body[@name='base_link']")
    base.set('pos', numbers(cfg['robot']['base_position']))
    base.set('quat', numbers(quaternion(rpy_matrix(cfg['robot']['base_rpy']))))
    for joint in root.findall('.//joint'):
        if joint.get('name') in limits:
            joint.set('damping', str(cfg['robot']['joint_damping']))
            joint.set('solreflimit', '0.004 1')
    for body in root.findall('.//body'):
        for geom in body.findall('geom'):
            if geom.get('contype', '1') != '0':
                geom.set('friction', f"{cfg['robot']['gripper_friction']} 0.01 0.001")
                geom.set('condim', '4')
    # Same exclusions as the official RoboTwin runtime, plus SRDF adjacency pairs.
    contact = ET.SubElement(root, 'contact')
    pairs = set()
    srdf = ET.parse(resolve(cfg, cfg['robot']['srdf_path']))
    pairs.update(tuple(sorted((e.get('link1'), e.get('link2')))) for e in srdf.findall('disable_collisions'))
    with resolve(cfg, cfg['robot']['collision_config']).open(encoding='utf-8') as stream:
        ignore = yaml.safe_load(stream)['robot_cfg']['kinematics']['self_collision_ignore']
    pairs.update(tuple(sorted((a, b))) for a, others in ignore.items() for b in others)
    for a, b in sorted(pairs):
        ET.SubElement(contact, 'exclude', body1=a, body2=b)
    equality = ET.SubElement(root, 'equality')
    for joint in tree.findall('joint'):
        mimic = joint.find('mimic')
        if mimic is not None:
            ET.SubElement(equality, 'joint', joint1=joint.get('name'), joint2=mimic.get('joint'),
                          polycoef=f"{mimic.get('offset', '0')} {mimic.get('multiplier', '1')} 0 0 0", solref='0.004 1')
    actuators = ET.SubElement(root, 'actuator')
    for index, name in enumerate(arm_names + ['gripper_L_joint', 'gripper_R_joint']):
        limit = limits[name]
        kp = cfg['robot']['kp'][index] if index < 6 else cfg['robot']['gripper_kp']
        kd = cfg['robot']['kd'][index] if index < 6 else cfg['robot']['gripper_kd']
        # MuJoCo integrates servo damping implicitly. A gravity offset to the
        # setpoint gives PD + bias compensation, with total effort still limited.
        ET.SubElement(actuators, 'position', name=f'{name}_servo', joint=name, kp=str(kp), kv=str(kd),
                      forcelimited='true', forcerange=f"-{limit['effort']} {limit['effort']}")
    ET.SubElement(world, 'light', pos='0 -1 3', dir='0 0 -1', diffuse='0.8 0.8 0.8', castshadow='false')
    ET.SubElement(world, 'light', pos='1 2 2', dir='-0.3 -0.4 -1', diffuse='0.4 0.4 0.4', castshadow='false')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='3 3 0.1', rgba='0.18 0.21 0.25 1')
    table = cfg['environment']
    center, half = np.array(table['table_position']), np.array(table['table_size']) / 2
    ET.SubElement(world, 'geom', name='table', type='box', pos=numbers(center), size=numbers(half), rgba='0.58 0.44 0.3 1', friction='1 0.01 0.001')
    # A visual site has no collision, mass, or effect on the official robot.
    from .task import DEFAULT_TASK
    target_half = cfg.get('task', {}).get('target_half_size', DEFAULT_TASK['target_half_size'])
    ET.SubElement(world, 'site', name='placement_target', type='box',
                  pos=numbers([.44, -.04, center[2]+half[2]+.0005]),
                  size=numbers([target_half, target_half, .0004]), rgba='0.1 0.9 0.25 0.55')
    for x in (-1, 1):
        for y in (-1, 1):
            h = (center[2]-half[2])/2
            ET.SubElement(world, 'geom', type='box', pos=numbers([center[0]+x*(half[0]-.06), center[1]+y*(half[1]-.06), h]),
                          size=numbers([.025,.025,h]), rgba='0.25 0.28 0.33 1')
    for obj in table['objects']:
        body = ET.SubElement(world, 'body', name=obj['name'], pos=numbers(obj['position']))
        ET.SubElement(body, 'freejoint', name=f"{obj['name']}_free")
        ET.SubElement(body, 'geom', name=f"{obj['name']}_geom", type=obj['type'], size=numbers(obj['size']),
                      mass=str(obj['mass']), rgba=numbers(obj['rgba']), friction='1.2 0.01 0.001', condim='4')
    cam = cfg['table_camera']
    forward = np.array(cam['look_at']) - cam['position']
    ET.SubElement(world, 'camera', name='table_camera', pos=numbers(cam['position']),
                  xyaxes=numbers(camera_axes(forward, cam['up'])), fovy=str(cam['fov']))
    cam = cfg['wrist_camera']
    parent = root.find(f".//body[@name='{cam['parent_link']}']")
    if parent is None:
        raise ValueError(f"Unknown wrist camera parent: {cam['parent_link']}")
    rotation = rpy_matrix(cam['orientation_offset_rpy'])
    ET.SubElement(parent, 'camera', name='wrist_camera', pos=numbers(cam['position_offset']),
                  xyaxes=numbers(camera_axes(rotation @ cam['forward'], rotation @ cam['up'])), fovy=str(cam['fov']))
    output = build / 'scene.xml'
    ET.indent(scene)
    scene.write(output, encoding='utf-8', xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(output))
    return model, limits
