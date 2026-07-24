#!/usr/bin/python
"""Replay a pretrained MANO grasping policy on a chosen GRAB object.

Loads a trained right-hand grasp policy (``-w/--weight``) and a config file
(``-c/--cfg``), spawns one RaiSim environment per selected object, generates an initial
hand/object pose from the object's affordance mesh, then rolls the policy out for
``trail_steps + lift_step`` steps across 25 repeated updates, optionally lifting the
object after the grasp phase. Each rollout's hand and object trajectory is recorded to
``data_all/diverse_seq_npy/<object>.npy`` and the object mesh is copied alongside to
``data_all/diverse_seq_obj/<object>.obj``.

Rebuild the environment after making changes to Environment.hpp or RaisimGymVecEnvOther.py.

To rebuild the environment, run::

    bash scripts/setup_graspxl.sh

Viewer: this script always runs with ``visualize=True`` (see below), which starts a
RaiSim server that a RaiSimUnity client can connect to over TCP -- the sim runs either
way (and still writes the recorded trajectory), but without a connected viewer you
won't see anything. Start RaiSimUnity *before* running this script::

    bash scripts/launch_raisim_unity.sh   # from the GraspXL/ directory

This opens the RaiSimUnity app and prints the two things to set once in its sidebar:
the resource directory (``GraspXL/rsc``) and Auto-connect (port 8080 by default).
Leave Auto-connect on and it reconnects automatically on every subsequent run. See
the top-level README's "Adding Custom Objects" / Demo sections for more detail.

Usage examples (run from anywhere; paths resolve relative to this script)::

    # Random object from rsc/mixed_train, default mass
    python runner.py

    # A specific object by name, then lift it after the grasp/trail phase
    python runner.py --object SodaCan_3703ada8cc31df4337b00c4c2fbe82aa --lift

    # Interactively list and pick an object by name or index
    python runner.py --select-object

    # Same object, 3x heavier (mass and inertia both scaled by 3.0)
    python runner.py --object SodaCan_3703ada8cc31df4337b00c4c2fbe82aa --mass-scale 3.0

    # Randomize the mass scale once per run, uniformly sampled from [0.5, 3.0]
    python runner.py --random-mass-scale 0.5 3.0

    # Lift the object and pull the table out from under it right before lift-off
    python runner.py --lift --no-support

    # Reproducible run: same object, same seed -> identical sequence of sampled grasp
    # points and hand/object initial poses across all 25 rollouts, every time
    python runner.py --object SodaCan_3703ada8cc31df4337b00c4c2fbe82aa --seed 42

    # Test the policy on a GRAB object instead of the default mixed_train (PartNet) set
    python runner.py --cat-name grab --object mug

CLI flags:
    -c, --cfg              Config YAML under ``cfgs/`` (default: ``cfg_reg.yaml``).
    -d, --logdir            Directory to write logs/checkpoints under (default: repo root).
    -e, --exp_name          Experiment name, used in the log directory path.
    -w, --weight            Policy checkpoint filename to load.
    -sd, --storedir         Subdirectory under ``<logdir>/raisimGymTorch/`` for run data.
    -seed, --seed           Seeds random/np.random/torch/RaiSim so that, for a fixed
                             --object, the sampled grasp points and resulting hand/object
                             initial poses across all 25 rollouts are reproducible.
    -cn, --cat-name          Object set to load from, i.e. the rsc/<cat_name>/ subdirectory
                             (default: mixed_train). E.g. grab for GRAB objects.
    -o, --object            Load a specific object by name instead of a random one.
    -oi, --object-index     Load a specific object by its 0-based sorted index.
    --select-object         Prompt interactively to pick an object (name or index).
    --lift                  Lift the object once the grasp/trail phase ends.
    --no-support             Remove the table's physical support at the same grasp/trail
                             -> lift transition (independent of --lift; combine or use
                             alone). See ``env.switch_table_support`` below.
    --mass-scale            Uniform multiplier on the loaded object's mass and inertia
                             (default: 1.0, i.e. unscaled). See ``object_mass.py``.
    --random-mass-scale     MIN MAX: sample the mass scale once, uniformly, from this
                             range at startup, overriding --mass-scale.

Mass is varied by rewriting the object's URDF's ``<inertial>`` tags before it is loaded
(see ``object_mass.scale_urdf_mass``) — the compiled RaiSim environment used here has no
Python-exposed ``setMass``/``setObjectMass`` API, unlike ``model_eval/run_scene.py``'s pure
Python RaiSim scenes, which can mutate mass on live bodies directly. The resolved mass
scale and each object's resulting total mass are printed to the command line once at
load time and again at the start of every rollout; RaiSim's viewer in this build has no
text-overlay API (only geometric visual objects and C++-only numeric charts), so mass is
not drawn into the 3D view itself.

``--no-support`` calls ``env.switch_table_support(True)``, a new per-env virtual method
(``RaisimGymEnv.hpp`` / ``VectorizedEnvironment.hpp`` / ``raisim_gym.cpp``, mirroring the
existing ``switch_root_guidance``/``--lift`` plumbing) that ``mano_mass_exp/Environment.hpp``
overrides to teleport the (invisible) table collision box far below the scene, out of
collision range. Sibling envs that don't override it get the shared no-op default, so this
required rebuilding the compiled ``mano_mass_exp`` module (see ``scripts/setup_graspxl.sh``);
it is not a pure-Python change like the mass-scale feature.

``--seed`` (default 1) seeds Python's ``random``, ``numpy``'s global RNG, and
``torch.manual_seed`` right after argument parsing, before anything else can draw from
them. This matters because every source of randomness in the per-rollout initial-pose
pipeline -- ``helper/initial_pose_final.get_initial_pose``'s grasp-direction/rotation/
target-point sampling, and the affordance/non-affordance point-cloud sampling
``RaisimGymVecEnvOther.RaisimGymVecEnvTest.__init__`` does once at ``VecEnv(...)``
construction time -- draws from that same global ``numpy.random`` state via
``trimesh.sample.sample_surface``/``np.random.uniform``/``numpy.random.randint`` (no call
site passes its own seed). So for a fixed ``--object``, seeding once up front is enough to
make the entire sequence of sampled grasp points and resulting hand/object initial poses
across all 25 rollouts exactly reproducible -- no per-call seeding needed. (The policy's
action selection itself, ``actor_r.architecture.architecture(obs)``, is already
deterministic -- it's a plain MLP forward pass, not the stochastic ``NormalSampler`` path
used only during training -- so it doesn't need seeding.)
"""

from ruamel.yaml import YAML, dump, RoundTripDumper
# The compiled module name always matches this env's directory name (mano_mass_exp),
# per raisimGymTorch's CMakeLists.txt (one pybind11 module per env/envs/<subdir>,
# named after <subdir>). Must be rebuilt (see scripts/setup_graspxl.sh) after editing
# Environment.hpp -- e.g. after adding switch_table_support for --no-support below.
from raisimGymTorch.env.bin import mano_mass_exp as mano
from raisimGymTorch.env.RaisimGymVecEnvOther import RaisimGymVecEnvTest as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param
from raisimGymTorch.env.bin.mano_mass_exp import NormalSampler
from raisimGymTorch.helper.initial_pose_final import get_initial_pose, get_initial_pose_set
import random
from random import choice, uniform
import shutil
import object_mass

import os
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import argparse
from raisimGymTorch.helper import rotations

import torch


data_version = "chiral_220223"
ref_r = [0.09566994, 0.00638343, 0.0061863]

# resolve paths relative to this script's location so it can be run from any cwd
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = os.path.normpath(os.path.join(task_path, "..", "..", "..", "..", ".."))
raisimgymtorch_path = os.path.normpath(os.path.join(task_path, "..", "..", "..", ".."))

path_mean_r = os.path.join(home_path, "rsc", "mano_double", "right_pose_mean.txt")
pose_mean_r = np.loadtxt(path_mean_r)


exp_name = "floating_mixed"

weight_saved = 'full_5600_r.pt'


record=True

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cfg', help='config file', type=str, default='cfg_reg.yaml')
parser.add_argument('-d', '--logdir', help='set dir for storing data', type=str, default=None)
parser.add_argument('-e', '--exp_name', help='exp_name', type=str, default=exp_name)
parser.add_argument('-w', '--weight', type=str, default=weight_saved)
parser.add_argument('-sd', '--storedir', type=str, default='data_all')
parser.add_argument('-seed', '--seed', type=int, default=1,
                     help='Seeds random/np.random/torch (and RaiSim\'s own RNG) so that, for '
                          'a fixed --object, the sequence of sampled grasp points and the '
                          'resulting hand/object initial poses across all 25 rollouts is '
                          'exactly reproducible run-to-run.')
parser.add_argument('-cn', '--cat-name', type=str, default='mixed_train',
                     help='Object set to load from, i.e. the subdirectory name under rsc/ '
                          '(default: mixed_train). E.g. --cat-name grab to load GRAB objects.')
parser.add_argument('-o', '--object', type=str, default=None,
                     help='Specific object name to load (default: random). See --select-object to pick interactively.')
parser.add_argument('-oi', '--object-index', type=int, default=None,
                     help='Select object by index instead of name (0-based, matches the --select-object listing).')
parser.add_argument('--select-object', action='store_true',
                     help='If --object/--object-index are not given, list available objects and prompt to pick one (by name or index) instead of choosing randomly.')
parser.add_argument('--lift', action='store_true',
                     help='If --lift is given, the object will be lifted after the grasp.')
parser.add_argument('--no-support', action='store_true',
                     help='If given, the table\'s physical support is removed once the hand '
                          'has a firm grasp, right before lift-off (same step as --lift\'s '
                          'guidance switch), so the object is held by the hand alone from '
                          'that point on. Independent of --lift: can be combined with it or '
                          'used on its own.')
parser.add_argument('--mass-scale', type=float, default=1.0,
                     help='Uniform multiplier on the loaded object\'s mass and inertia '
                          '(default: 1.0, unscaled). Applied by rewriting the object URDF '
                          'before it is loaded; see object_mass.py.')
parser.add_argument('--random-mass-scale', type=float, nargs=2, default=None, metavar=('MIN', 'MAX'),
                     help='Sample the mass scale once, uniformly, from [MIN, MAX] at startup, '
                          'overriding --mass-scale. Mass is fixed for the whole run since the '
                          'object is only loaded once (before the update loop).')

args = parser.parse_args()
weight_path = args.weight
cfg_grasp = args.cfg

# Seed every RNG that feeds the initial-pose sampling pipeline (get_initial_pose's
# grasp-direction/rotation/target-point draws in helper/initial_pose_final.py, the
# affordance/non-affordance point-cloud sampling VecEnv does at construction time in
# RaisimGymVecEnvOther.py, and --random-mass-scale below) so that, for a fixed object,
# the sequence of sampled grasp points and resulting hand/object initial poses across
# the 25 rollouts is exactly reproducible run-to-run. This must happen before VecEnv(...)
# is constructed below, since that constructor already draws from np.random. All of
# get_initial_pose's sampling (trimesh.sample.sample_surface, np.random.uniform,
# numpy.random.randint) draws from this same global numpy RNG, so seeding it once here
# is sufficient -- no per-call seed threading is needed.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# resolve the effective object mass-scale: a random draw (if requested) takes priority
# over the fixed --mass-scale value; both are applied identically downstream.
if args.random_mass_scale is not None:
    mass_scale_min, mass_scale_max = args.random_mass_scale
    if mass_scale_min <= 0 or mass_scale_max <= 0 or mass_scale_min > mass_scale_max:
        parser.error(
            f"--random-mass-scale MIN MAX must satisfy 0 < MIN <= MAX "
            f"(got {mass_scale_min} {mass_scale_max})."
        )
    mass_scale = uniform(mass_scale_min, mass_scale_max)
    print(f"Randomized mass scale: {mass_scale:.3f}x (sampled from [{mass_scale_min}, {mass_scale_max}])")
else:
    mass_scale = args.mass_scale

print(f"Configuration file: \"{args.cfg}\"")
print(f"Experiment name: \"{args.exp_name}\"")

# task specification
task_name = args.exp_name
# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.logdir is None:
    exp_path = home_path
else:
    exp_path = args.logdir

# config
cfg = YAML().load(open(task_path + '/cfgs/' + args.cfg, 'r'))

if args.seed != 1:
    cfg['seed'] = args.seed

cfg['environment']['visualize'] = True


cat_name = args.cat_name

cfg['environment']['load_set'] = cat_name
directory_path = home_path + f"/rsc/{cat_name}/"
print(directory_path)

items = os.listdir(directory_path)

# # Filter out only the folders (directories) from the list of items
folder_names = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]

# obj_ori_list / sorted_obj_list: the full catalog of loadable objects (one per
# subdirectory of rsc/mixed_train). obj_path_list is filled in below once the actual
# per-run URDF (original or mass-scaled) is known for each selected object.
obj_path_list = []
obj_ori_list = folder_names
sorted_obj_list = sorted(obj_ori_list)


def _resolve_object(text):
    """Resolve user input as either an object name or a 0-based index into sorted_obj_list."""
    if text in obj_ori_list:
        return text
    if text.isdigit() and 0 <= int(text) < len(sorted_obj_list):
        return sorted_obj_list[int(text)]
    return None


# Resolve which object(s) this run will grasp: an explicit name/index takes priority,
# then interactive selection, then a uniform random pick. num_envs (below) always ends
# up equal to len(obj_list), so this currently means exactly one parallel environment.
if args.object is not None:
    if args.object not in obj_ori_list:
        parser.error(
            f"Unknown --object '{args.object}'. Available objects:\n  "
            + "\n  ".join(sorted_obj_list)
        )
    obj_list = [args.object]
elif args.object_index is not None:
    if not (0 <= args.object_index < len(sorted_obj_list)):
        parser.error(f"--object-index {args.object_index} out of range (0-{len(sorted_obj_list) - 1}).")
    obj_list = [sorted_obj_list[args.object_index]]
elif args.select_object:
    print("Available objects:")
    for i, name in enumerate(sorted_obj_list):
        print(f"  [{i}] {name}")
    selected = _resolve_object(input("Select an object (name or index): ").strip())
    while selected is None:
        selected = _resolve_object(input("Unknown object. Select an object (name or index): ").strip())
    obj_list = [selected]
else:
    obj_list = [choice(obj_ori_list)]
print(obj_list)

num_envs = len(obj_list)

activations = nn.LeakyReLU

cfg['environment']['num_envs'] = num_envs
print('num envs', num_envs)

# Environment definition
env = VecEnv(obj_list, mano.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
             cfg['environment'], cat_name=cat_name)
# Also seed RaiSim's own per-thread C++ RNG for completeness/future-proofing; nothing in
# mano_mass_exp/Environment.hpp currently draws from it, so this has no effect today, but
# keeps the run reproducible if that ever changes.
env.seed(args.seed)

print("initialization finished")

# Build the list of URDF paths (relative to directory_path/, matching resourceDir_
# + load_set on the C++ side) that RaiSim will load one per environment. When a
# mass scale other than 1.0 is in effect, load a scaled sibling URDF instead of the
# original so the object is heavier/lighter without touching the base dataset file.
# object_masses records the resulting per-object total mass so it can be re-printed
# at the start of every rollout below (RaiSim's viewer has no text-overlay API in
# this build -- only geometric visual objects and C++-only numeric charts -- so mass
# is reported to the command line only, not drawn into the 3D view).
object_masses = {}
for obj_item in obj_list:
    base_urdf_path = os.path.join(directory_path, obj_item, f"{obj_item}.urdf")
    load_urdf_path = object_mass.scale_urdf_mass(base_urdf_path, mass_scale)
    object_masses[obj_item] = object_mass.total_mass(load_urdf_path)
    if load_urdf_path != base_urdf_path:
        print(
            f"{obj_item}: mass scale {mass_scale:.3f}x -> "
            f"{object_masses[obj_item]:.4f} kg "
            f"(base {object_mass.total_mass(base_urdf_path):.4f} kg)"
        )
    obj_path_list.append(os.path.join(obj_item, os.path.basename(load_urdf_path)))
env.load_multi_articulated(obj_path_list)

# ob_dim_r/act_dim must match the trained policy's network shape exactly (304-dim
# observation, 51-dim action = 6 wrist DoF + 45 finger DoF).
ob_dim_r = 304
act_dim = 51
print('ob dim', ob_dim_r)
print('act dim', act_dim)

# Training
# n_steps_r: length of one rollout (grasp approach + trail hold + optional lift phase).
# This script only runs inference (load_param below), so "Training" here just means
# the same fixed-length-rollout config the policy was trained/evaluated with.
trail_steps = 60
reward_clip = -2.0
grasp_steps = 0
lift_step = 80
n_steps_r = grasp_steps + trail_steps + lift_step
total_steps_r = n_steps_r * env.num_envs

# RL network
# Actor/critic architectures must mirror the checkpoint being loaded (load_param below
# restores their weights in place); PPO itself is only instantiated to host the
# optimizer state that load_param expects, no learning updates happen in this script.

actor_r = ppo_module.Actor(
    ppo_module.MLP(cfg['architecture']['policy_net'], activations, ob_dim_r, act_dim),
    ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, num_envs, 1.0, NormalSampler(act_dim)), device)

critic_r = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], activations, ob_dim_r, 1), device)


test_dir = True

saver = ConfigurationSaver(log_dir=exp_path + "/raisimGymTorch/" + args.storedir + "/" + task_name,
                           save_items=[], test_dir=test_dir)


ppo_r = PPO.PPO(actor=actor_r,
                critic=critic_r,
                num_envs=num_envs,
                num_transitions_per_env=n_steps_r,
                num_learning_epochs=4,
                gamma=0.996,
                lam=0.95,
                num_mini_batches=4,
                device=device,
                log_dir=saver.data_dir,
                shuffle_batch=False
                )
# Loads the pretrained weights (args.weight) into actor_r/critic_r/ppo_r.optimizer in place.
load_param(saver.data_dir.split('eval')[0]+weight_path, env, actor_r, critic_r, ppo_r.optimizer, saver.data_dir, cfg_grasp)

success_rate = 0.0
ori_error = 0.0
direction_error = 0.0
rotation_error = 0.0
center_error = 0.0
contact_ratio = 0.0
suc_ori_error = 0.0
suc_direction_error = 0.0
suc_rotation_error = 0.0
suc_center_error = 0.0
suc_contact_ratio = 0.0

lifted_num = 0

# Each iteration of this loop is one full independent rollout (a fresh initial pose is
# sampled, the sim is reset, then the policy runs for n_steps_r steps and the
# trajectory is optionally recorded). 25 repeats give 25 recordings per selected object.
for update in range(25):
    start = time.time()

    # Report the mass scale and each object's resulting total mass to the command
    # line at the start of every rollout (RaiSim's viewer here has no on-screen text
    # API, so this is command-line-only -- see the object_masses comment above).
    print(f"[sim {update}/25] mass_scale={mass_scale:.3f}x  " + ", ".join(
        f"{obj_item}={object_masses[obj_item]:.4f}kg" for obj_item in obj_list
    ))

    qpos_reset_r = np.zeros((num_envs, 51), dtype='float32')
    qpos_reset_l = np.zeros((num_envs, 51), dtype='float32')
    obj_pose_reset = np.zeros((num_envs, 8), dtype='float32')

    target_center = np.zeros_like(env.affordance_center)
    object_center = np.zeros_like(env.affordance_center)
    fake_non_aff_center = [0.346408, 0.346408, 0.346408]
    contain_non_aff = np.zeros((num_envs, 1), dtype='float32')

    # Per-object initial-pose generation: place the object at its recorded resting
    # height, seed the hand near its mean pose, then sample a hand approach
    # pose/target relative to the object's affordance region (get_initial_pose).
    for i in range(num_envs):
        lowest_point = 0.
        txt_file_path = os.path.join(directory_path, obj_list[i]) + "/lowest_point_new.txt"
        with open(txt_file_path, 'r') as txt_file:
            lowest_point = float(txt_file.read())

        obj_pose_reset[i, :] = [1., -0., 0.502, 1., -0., -0., 0., 0.]
        obj_pose_reset[i, 2] -= lowest_point

        qpos_reset_r[i, 6:] = pose_mean_r.copy() / 5
        qpos_reset_r[i, -9:-6] *= 5
        qpos_reset_r[i, -8] += 0.4
        qpos_reset_r[i, -7] += 0.6

        # fake_non_aff_center is the sentinel the env uses when an object has no
        # non-affordance part (e.g. no "bottom" mount); a centroid this close to it
        # means treat the object as single-part for pose generation/reward purposes.
        if np.linalg.norm(env.non_aff_mesh[i].centroid - fake_non_aff_center) < 0.01:
            non_aff_mesh = None
        else:
            non_aff_mesh = env.non_aff_mesh[i]
            contain_non_aff[i, 0] = 1.

        # direction = np.array([[0, 1, 0]])
        # rotation = 3.14
        # point = np.array([0.0, -0.0, 0.00])
        # rot, pos, bias = get_initial_pose_set(env.aff_mesh[i], non_aff_mesh, direction, rotation, point)
        rot, pos, bias = get_initial_pose(env.aff_mesh[i], non_aff_mesh)

        obj_mat = rotations.quat2mat(obj_pose_reset[i, 3:7])
        wrist_pose_obj = rotations.axisangle2euler(rot.reshape(-1, 3)).reshape(1, -1)
        wrist_mat = rotations.euler2mat(wrist_pose_obj)
        wrist_in_world = np.matmul(obj_mat, wrist_mat)
        wrist_pose = rotations.mat2euler(wrist_in_world)
        qpos_reset_r[i, :3] = obj_pose_reset[i, :3] + np.matmul(obj_mat, pos[0, :])
        qpos_reset_r[i, 3:6] = wrist_pose[0, :]

        target_center[i, :] = bias[:]
        object_center[i, :] = env.affordance_center[i]
    print("complete initial pose generation")

    # Push the sampled initial hand/object poses into the sim (zero velocity reset).
    env.reset_state(qpos_reset_r,
                    qpos_reset_l,
                    np.zeros((num_envs, 51), 'float32'),
                    np.zeros((num_envs, 51), 'float32'),
                    obj_pose_reset,
                    )

    # Set the reward targets for this rollout; only target_center/object_center are
    # populated above, the remaining scalar goal channels are unused by this policy
    # (kept zero) but required by the fixed set_goals signature.
    env.set_goals(target_center,
                  object_center,
                  np.zeros((num_envs, 1), 'float32'),
                  np.zeros((num_envs, 1), 'float32'),
                  np.zeros((num_envs, 1), 'float32'),
                  np.zeros((num_envs, 1), 'float32'),
                  np.zeros((num_envs, 1), 'float32'),
                  np.zeros((num_envs, 1), 'float32'),
                  np.zeros((num_envs, 1), 'float32'),
                  np.zeros((num_envs, 1), 'float32'),
                  )

    obs_new_r, _ = env.observe(contain_non_aff, partial_obs=False)
    if record:
        # Per-step trajectory buffers for this rollout: right-hand wrist
        # translation/rotation, 15-joint finger pose (axis-angle, mean-pose-relative),
        # and object translation/rotation/joint-angle. Written to disk after the loop.
        trans_r = np.zeros((n_steps_r, 3))
        rot_r = np.zeros((n_steps_r, 3))
        pose_r = np.zeros((n_steps_r, 45))
        trans_obj = np.zeros((n_steps_r, 3))
        rot_obj = np.zeros((n_steps_r, 3))
        angle_obj = np.zeros((n_steps_r, 1))
        points = np.zeros((n_steps_r, 2, 21, 3))
    # Policy rollout: one action per physics step, real-time-paced via wait_time so
    # playback (this script always runs with visualize=True) doesn't run faster than
    # the configured control_dt.
    for step in range(n_steps_r):
        obs_r = obs_new_r
        obs_r = obs_r[:, :].astype('float32')

        # lift the object after the grasp and trail steps
        if args.lift and step == (grasp_steps + trail_steps):
            env.switch_root_guidance(True)

        # remove the table's support at the same grasp->lift transition, once the
        # hand has held the object through the trail phase (a proxy for "firm grasp")
        if args.no_support and step == (grasp_steps + trail_steps):
            env.switch_table_support(True)

        action_r = actor_r.architecture.architecture(torch.from_numpy(obs_r.astype('float32')).to(device))
        action_r = action_r.cpu().detach().numpy()
        action_l = np.zeros_like(action_r)

        frame_start = time.time()

        reward_r, _, dones = env.step(action_r.astype('float32'), action_l.astype('float32'))

        obs_new_r, dis_info = env.observe(contain_non_aff, partial_obs=False)
        obs_new_r = obs_new_r[:].astype('float32')

        global_state = env.get_global_state().reshape(-1,).astype('float32')

        # global_state layout (fixed offsets into the flattened per-env state vector):
        #   [136:139]  right-wrist world translation      [139:143]  wrist quaternion
        #   [143:188]  15 finger joints, euler (45 floats) [188:191]  object translation
        #   [191:195]  object quaternion
        if record:
            trans_r[step, :] = global_state[136:139] - ref_r
            axis, angle = rotations.quat2axisangle(global_state[139:143].reshape(1,4))
            rot_r[step, :] = axis * angle

            temp_pose = global_state[143:188].reshape(15, 3)
            for j in range(15):
                pose_r[step, 3*j:3*j+3] = rotations.euler2axisangle(temp_pose[j].reshape(1,3))

            pose_r[step, :] = pose_r[step, :] - pose_mean_r

            trans_obj[step, :] = global_state[188:191]
            axis, angle = rotations.quat2axisangle(global_state[191:195].reshape(1,4))
            rot_obj[step, :] = axis * angle

        # Sleep out the remainder of control_dt so playback runs at real-time speed.
        frame_end = time.time()
        wait_time = cfg['environment']['control_dt'] - (frame_end - frame_start)
        if wait_time > 0.:
            time.sleep(wait_time)

    # Persist this rollout's trajectory: one .npy dict per object (hand + object
    # trans/rot/angle arrays) plus a copy of the object mesh for downstream tools that
    # expect the mesh alongside the trajectory.
    if record:
        for i in range(num_envs):
            obj_item = obj_list[i]
            data = {}
            data['right_hand'] = {}
            data[obj_item] = {}
            data['right_hand']['trans'] = np.float32(trans_r[:])
            data['right_hand']['rot'] = np.float32(rot_r[:])
            data['right_hand']['pose'] = np.float32(pose_r[:])

            data[obj_item]['trans'] = np.float32(trans_obj[:])
            data[obj_item]['rot'] = np.float32(rot_obj[:])
            data[obj_item]['angle'] = np.float32(angle_obj[:])

            npy_dir = os.path.join(raisimgymtorch_path, "data_all", "diverse_seq_npy")
            obj_dir = os.path.join(raisimgymtorch_path, "data_all", "diverse_seq_obj")
            os.makedirs(npy_dir, exist_ok=True)
            os.makedirs(obj_dir, exist_ok=True)

            np.save(os.path.join(npy_dir, f"{obj_item}.npy"), data)

            obj_file = directory_path + obj_item + "/top_watertight_tiny.obj"
            shutil.copy(obj_file, os.path.join(obj_dir, f"{obj_item}.obj"))

    print("end")
