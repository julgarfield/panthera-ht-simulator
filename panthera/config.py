from copy import deepcopy
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    path = Path(path or ROOT / 'config.yaml').resolve()
    with path.open(encoding='utf-8') as stream:
        cfg = yaml.safe_load(stream)
    cfg['_root'] = str(path.parent)
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    sim = cfg['simulation']
    for key in ('physics_hz', 'control_hz', 'display_hz'):
        if not isinstance(sim[key], int) or sim[key] <= 0:
            raise ValueError(f'simulation.{key} must be a positive integer')
    rates = [sim['control_hz'], cfg['table_camera']['fps'], cfg['wrist_camera']['fps']]
    if any(not isinstance(r, int) or r <= 0 or sim['physics_hz'] % r for r in rates):
        raise ValueError('Control and camera rates must divide physics_hz exactly')
    if rates[1] != rates[2] or rates[0] % rates[1]:
        raise ValueError('Cameras must have equal FPS, dividing control_hz for exact synchronization')
    if len(cfg['robot']['home_configuration']) != 6:
        raise ValueError('Exactly six arm home positions are required')
    for name in ('table_camera', 'wrist_camera'):
        camera = cfg[name]
        if min(camera['width'], camera['height']) < 16 or not 1 < camera['fov'] < 179:
            raise ValueError(f'Invalid {name} resolution or field of view')
    if cfg['logging']['format'] != 'hdf5':
        raise ValueError('Live recording uses hdf5; use --export for CSV and PNG')
    if cfg['logging']['queue_size'] < 1:
        raise ValueError('logging.queue_size must be positive')
    for k in ('joint_speed', 'cartesian_translation_speed', 'cartesian_rotation_speed', 'gripper_speed', 'ik_damping', 'max_target_lead'):
        if cfg['controls'][k] <= 0:
            raise ValueError(f'controls.{k} must be positive')


def resolve(cfg, path):
    p = Path(path)
    return p if p.is_absolute() else Path(cfg['_root']) / p


def portable_config(cfg):
    copy = deepcopy(cfg)
    copy.pop('_root', None)
    return copy
