# GraspXL: Generating Grasping Motions for Diverse Objects at Scale

<p align="center">
  <a href="https://arxiv.org/pdf/2403.19649.pdf">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white">
  </a>
  <a href="https://eth-ait.github.io/graspxl/">
    <img alt="Project Page" src="https://img.shields.io/badge/Project%20Page-Website-2f80ed?style=for-the-badge&logo=googlechrome&logoColor=white">
  </a>
  <a href="https://youtu.be/0-dRbxmX2PI">
    <img alt="Video" src="https://img.shields.io/badge/Video-YouTube-ff0000?style=for-the-badge&logo=youtube&logoColor=white">
  </a>
  <a href="https://huggingface.co/datasets/ethHuiZhang/GraspXL">
    <img alt="Dataset" src="https://img.shields.io/badge/Dataset-Hugging%20Face-ffcc4d?style=for-the-badge&logo=huggingface&logoColor=black">
  </a>
  <a href="https://github.com/zdchan/GraspXL">
    <img alt="Code" src="https://img.shields.io/badge/Code-GitHub-24292f?style=for-the-badge&logo=github&logoColor=white">
  </a>
  <a href="https://github.com/zdchan/GraspXL_visualization">
    <img alt="Visualizer" src="https://img.shields.io/badge/Visualizer-GitHub-6f42c1?style=for-the-badge&logo=github&logoColor=white">
  </a>
</p>

<img src="/tease_more.jpg" /> 

### Contents

1. [News](#News)
2. [Dataset](#Dataset)
3. [Code](#Code)
4. [Installation](#installation)
5. [Demo](#Demo)
6. [Adding Custom Objects (e.g. GRAB)](#adding-custom-objects-eg-grab)
7. [Citation](#citation)
8. [License](#license)

## News
[2026.05] **We uploaded the GraspXL dataset to [Hugging Face](https://huggingface.co/datasets/ethHuiZhang/GraspXL). We also updated data for the new tabletop grasping setting of MANO, Allegro, and Shadow hands. For MANO, we additionally fixed the unnatural little-finger motions.**

[2026.05] **We also updated the visualizer [GraspXL_visualization](https://github.com/zdchan/GraspXL_visualization) for the new tabletop grasping setting.**

[2024.11] We released the grasping motion data for the LEAP hand model on a subset of our objects!

[2024.10] We released the script in [URDF_gen_from_obj](./raisimGymTorch/raisimGymTorch/helper/URDF_gen_from_obj) to pre-process new objects. Put the .obj files you want to grasp (make sure they have meaningful sizes for grasping) under [URDF_gen_from_obj/temp](./raisimGymTorch/raisimGymTorch/helper/URDF_gen_from_obj/) and run [urdf_gen.py](./raisimGymTorch/raisimGymTorch/helper/URDF_gen_from_obj/urdf_gen.py), then it will generate a folder with the processed objects and the urdf files in [rsc](./rsc), which you can further utilize to generate grasping motions with any environment scripts.

[2024.08] Data example & viewer released!

[2024.08] Code released!

[2024.07] The large-scale generated motions for 500k+ objects, each with diverse objectives and currently MANO and Allegro hand models, are ready to download! If you are interested, ~~just fill [this form](https://forms.gle/dNwaGvtb4ppi1HZt5) to get access!~~ just download it from [Hugging Face](https://huggingface.co/datasets/ethHuiZhang/GraspXL).

We will continuously enrich the dataset (e.g., motions generated with more hand models, more grasping motions generated with different objectives, etc) and keep you updated!

[2024.03] **~~The code will be released soon.~~ Please fill out [this form](https://forms.gle/dNwaGvtb4ppi1HZt5) if you want to get the notification for any update!**



## Dataset
The dataset has been released, including the grasping motion sequences of different robot hands for 500k+ objects (for both diverse approaching direction setting and tabletop setting). Check [Hugging Face](https://huggingface.co/datasets/ethHuiZhang/GraspXL) for details and instructions.

For an easier trial of the dataset, we give some examples (30 objects) of the data in the [dataset_example](./dataset_example) subfolder.

We also provide a viewer for the grasping motions. Check [GraspXL_visualization](https://github.com/zdchan/GraspXL_visualization) for more details.

**For texture**. We use decimated and texture-free Objaverse meshes in our dataset for smaller space consumption. However, the original Objaverse object ids are still included in the dataset (<object_id>). You can download the original Objaverse objects with textures according to their official download tutorial. The only thing to notice is the meshes we use in our dataset are scaled from original meshes, so you should calculate the scaling factor of each object by the bounding box size, and scale the downloaded original Objaverse mesh accordingly while keeping the textures. It should be quite convenient to be done with a Python script using trimesh or/and pymeshlab. After this, you can replace the objects in the dataset with the textured meshes for visualization.

**Note** The MANO hand poses in our dataset align with the original MANO model. Note that [manopth](https://github.com/hassony2/manopth) and [manotorch](https://github.com/lixiny/manotorch) have different joint orders. For more details, check [manotorch](https://github.com/lixiny/manotorch).



## Code

The repository comes with all the features of the [RaiSim](https://raisim.com/) physics simulation, as GraspXL is integrated into RaiSim.

The GraspXL related code can be found in the [raisimGymTorch](./raisimGymTorch) subfolder. There are 12 environments (see [envs](./raisimGymTorch/raisimGymTorch/env/envs/)) for Allegro Hand ("allegro\_"), Mano Hand ("ours\_") and Shadow Hand ("shadow\_"), 4 for each. "\_fixed" and "\_floating" represent the environments for the first and second training phase respectively. "\_test" represents the test environments and contain different test scripts for different test sets (PartNet, ShapeNet, Objaverse, and Generated/Reconstructed objects). "\_demo" represents the visualization environments which also record the generated motions.



## Installation


For good practice for Python package management, it is recommended to use virtual environments (e.g., `virtualenv` or `conda`) to ensure packages from different projects do not interfere with each other. The code is tested under Python 3.8.10.

### RaiSim setup

GraspXL is based on RaiSim simulation. For the installation of RaiSim, see and follow our documentation under [docs/INSTALLATION.md](./docs/INSTALLATION.md). Note that you need to get a valid, free license for the RaiSim physics simulation and an activation key (run any script and follow the instruction). 

### GraspXL setup

After setting up RaiSim, the last part is to set up the GraspXL environments.

```
$ cd raisimGymTorch 
$ python setup.py develop
```

All the environments are run from this raisimGymTorch folder. 

Note that every time you change the environment.hpp, you need to run `python setup.py develop` again to build the environments.

Then install pytorch with (Check your CUDA version and make sure they match)

```
$ pip3 install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
```

Install required packages

```
$ pip install scipy
$ pip install scikit-learn scipy matplotlib
```

### Other alternative requirements

1. (Only for Mano policy training) GraspXL uses [manotorch](https://github.com/lixiny/manotorch) [Anatomy Loss](https://github.com/lixiny/manotorch#anatomy-loss) during the training (for Mano Hand only), so if you want to train Mano Hand policies (run [ours_fixed/runner.py](./raisimGymTorch/raisimGymTorch/env/envs/ours_fixed/runner.py) or [ours_floating/runner.py](./raisimGymTorch/raisimGymTorch/env/envs/ours_floating/runner.py)), you need to install manotorch. Please follow the official guideline in [manotorch](https://github.com/lixiny/manotorch).

​	After installation, replace the mano_assets_root in [mano_amano.py](https://github.com/zdchan/GraspXL/blob/1e239242082ec2bae9b9eddb4895f9f4f1d640af/raisimGymTorch/raisimGymTorch/helper/mano_amano.py#L10-L13) to your own path.

2. (Only for ShapeNet test set) If you want to generate motions for the objects from the ShapeNet test set, download [ShapeNet.zip](https://1drv.ms/u/s!ArIwHmrYW4HkoO0tm1D48rVudC4Bnw?e=DyEtsL), upzip and put the folder named large_scale_obj in [rsc](./rsc) (The original object meshes are from [ShapeNet](https://www.shapenet.org/))
3. (Only for 500k+ Objaverse test set) If you want to generate motions for the 500k+ objaverse objects, fill [this form](https://forms.gle/dNwaGvtb4ppi1HZt5) to get access to objaverse_urdf.zip. Unzip it and put the subset you want in [rsc](./rsc).

You should be all set now. Try to run the demo!



## Demo

We provide some pre-trained models to view the output of our method. They are stored in [this folder](./raisimGymTorch/data_all/). 

+ For interactive visualizations, start the RaiSimUnity viewer *before* running a demo/runner script (the RaiSim server it connects to keeps running either way, but you won't see anything without a connected viewer). The platform-specific binary lives under [raisimUnity](./raisimUnity) (`linux/raisimUnity.x86_64`, `mac/RaiSimUnity.app`, `m1/RaiSimUnity.app`, or `win32/RaiSimUnity.exe`) and needs the [rsc](./rsc) folder added as a resource directory with Auto-connect enabled (port 8080 by default) — do this once, it's remembered afterwards.

  This repo provides a script that launches the right binary for your platform and prints exactly what to set:

  ```Shell
  bash scripts/launch_raisim_unity.sh   # from the GraspXL/ directory
  ```

+ To randomly choose an object and visualize the generated sequences in simulation (use Mano Hand as an example), run

  ```Shell
  python raisimGymTorch/env/envs/ours_demo/demo.py
  ```

You can indicate the objects or the objectives of the generated motions in the visualization environments

+ The object is by default a random object from the training set, which you can change to a specified object. You can specify the object set by the variable cat_name (e.g., for [ours_demo](https://github.com/zdchan/GraspXL/blob/1e239242082ec2bae9b9eddb4895f9f4f1d640af/raisimGymTorch/raisimGymTorch/env/envs/ours_demo/demo.py#L76)), and choose a specific object by the variable obj_list (e.g., for [ours_demo](https://github.com/zdchan/GraspXL/blob/1e239242082ec2bae9b9eddb4895f9f4f1d640af/raisimGymTorch/raisimGymTorch/env/envs/ours_demo/demo.py#L90)). 

  The object sets include [mixed_train](./rsc/mixed_train) (the training set from [PartNet](https://partnet.cs.stanford.edu/)), [affordance_level](./rsc/affordance_level) (the PartNet test set), [large_scale_obj](./rsc/large_scale_obj) (the ShapeNet test set which you can download with [ShapeNet.zip](https://1drv.ms/u/s!ArIwHmrYW4HkoO0tm1D48rVudC4Bnw?e=DyEtsL)),  [YCB](./rsc/YCB) (reconstructed YCB objects), [gt](./rsc/gt) (groundtruth of the reconstructed YCB objects), [wild](./rsc/wild) (reconstructed in-the-wild objects), [gen](./rsc/gen) (objects generated with [DreamFusion](https://dreamfusion3d.github.io/))

+ The objectives are by default randomly sampled with the function get_initial_pose. You can also specify a desired objective with the function get_initial_pose_set.  [ours_demo](https://github.com/zdchan/GraspXL/blob/1e239242082ec2bae9b9eddb4895f9f4f1d640af/raisimGymTorch/raisimGymTorch/env/envs/ours_demo/demo.py#L198-L201) shows an example.



## Adding Custom Objects (e.g. GRAB)

Every environment loads objects from a folder under [rsc](./rsc) (the `cat_name`/`load_set` variable in each env's runner/demo script -- see [Demo](#demo) above). Each object subfolder needs:

- `<name>.urdf` (+ `_fixed_base.urdf`)
- `top_watertight_tiny.obj/.stl` -- the object mesh, loaded as the graspable "affordance" region
- `bottom_watertight_tiny.obj/.stl` -- a second mesh for a non-graspable region. Objects with no such split (most rigid objects, including all of GRAB) get a placeholder `cube.obj` here instead; `RaisimGymVecEnvOther.py` recognizes that cube's fixed centroid and skips the non-affordance branch entirely, so the whole object is treated as graspable.
- `lowest_point_new.txt` -- the object's resting-height offset

To regenerate this from a directory of raw object meshes (`.obj`/`.ply`, one file per object -- e.g. a downloaded [GRAB](https://grab.is.tue.mpg.de/) dataset's `tools/object_meshes/contact_meshes/`):

```bash
cd raisimGymTorch/raisimGymTorch/helper/URDF_gen_from_obj

# 1. Stage the source meshes as temp/<category>/<name>.obj (converts .ply -> .obj)
python stage_meshes.py /path/to/grab/tools/object_meshes/contact_meshes --dest-category grab

# 2. Convert every staged object into a URDF + top/bottom meshes
python urdf_gen.py
# -> writes rsc/formated_temp_obj_urdf/<name>/ for every staged object

# 3. Rename to the object-set name you'll pass as --cat-name / cat_name
mv ../../../../rsc/formated_temp_obj_urdf ../../../../rsc/grab

# 4. Generate lowest_point_new.txt for every object in the new set
cd ../../../../rsc
python -c "from get_lowest_point import process_objects; process_objects('grab/')"
```

Then point an env at the new set. `mano_mass_exp/runner.py` exposes this as a flag:

```bash
python raisimGymTorch/env/envs/mano_mass_exp/runner.py --cat-name grab --object mug
```

Other envs' runner/demo scripts still hardcode `cat_name` (see the [Demo](#demo) section above) -- edit that variable directly, or port the same `--cat-name` argparse option over from `mano_mass_exp/runner.py`.

`urdf_gen.py`'s mesh-processing helpers can optionally use `pymeshlab` and `smplx`, but only in code paths this object-only pipeline doesn't exercise (mano-hand URDF export, an alternate pymeshlab-based mesh backend) -- those imports are lazy, so the steps above work without either package installed.



## BibTeX Citation

To cite us, please use the following:

```bibtex
@inProceedings{zhang2024graspxl,
  title={{GraspXL}: Generating Grasping Motions for Diverse Objects at Scale},
  author={Zhang, Hui and Christen, Sammy and Fan, Zicong and Hilliges, Otmar and Song, Jie},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```



## License

This work and the dataset are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
