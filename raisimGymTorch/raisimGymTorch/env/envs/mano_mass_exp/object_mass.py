"""Utilities for varying an object's simulated mass in ``mano_mass_exp``.

``mano_mass_exp``'s environment (``Environment.hpp``) is compiled ahead of time into a
pybind11 shared object (``raisimGymTorch.env.bin.ours_floating_no_support``). It only
*reads* an object's mass (``arctic->getTotalMass()``, used for gravity-compensation and
non-affordance bookkeeping) — there is no ``setMass``/``setObjectMass`` exposed to Python,
unlike ``model_eval/run_scene.py`` which builds its RaiSim world directly in Python and can
call ``body.setMass()``/``body.setInertia()`` on live objects.

The lever available here instead is the URDF the object is loaded from: each GRAB object
directory (e.g. ``rsc/mixed_train/SodaCan_.../SodaCan_....urdf``) has explicit
``<inertial><mass value="..."/><inertia ixx=".." .../></inertial>`` tags per link. This
module rewrites those tags by a uniform scale factor and writes the result to a sibling
file next to the original, so relative ``<mesh>`` references in the URDF keep resolving
without needing to copy or symlink any mesh files. ``runner.py`` then loads that generated
URDF instead of the original.

The scaling models "same geometry, denser/lighter material": the center of mass is left
untouched and inertia is scaled by the same factor as mass, which is exact for uniform
density scaling of a fixed shape.
"""

import os
import xml.etree.ElementTree as ET

_INERTIA_ATTRS = ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")


def _scale_tag(mass_scale: float) -> str:
    """Turn a float like 2.5 into a filesystem-safe tag like "2p5" (or "0p5neg" for negatives)."""
    text = f"{mass_scale:g}"
    return text.replace(".", "p").replace("-", "neg")


def scale_urdf_mass(urdf_path: str, mass_scale: float) -> str:
    """Return the path to a mass-scaled sibling copy of ``urdf_path``.

    Multiplies every ``<inertial><mass>`` value and every attribute on the paired
    ``<inertia>`` tag by ``mass_scale``, for every link in the URDF. Writes the result
    to ``<same directory>/<base>_mass<tag>.urdf`` (e.g. ``SodaCan_..._mass2p5.urdf``),
    overwriting it on each call so repeated runs stay in sync with the source URDF and
    the chosen scale.

    Args:
        urdf_path: Absolute or relative path to the original object URDF.
        mass_scale: Uniform multiplier applied to mass and inertia. ``1.0`` is a no-op
            that returns ``urdf_path`` unchanged (no file is written).

    Returns:
        Path to the URDF to actually load: either the untouched ``urdf_path`` (when
        ``mass_scale == 1.0``) or the newly written scaled sibling file.
    """
    if mass_scale == 1.0:
        return urdf_path

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for inertial in root.iter("inertial"):
        mass_el = inertial.find("mass")
        if mass_el is not None and "value" in mass_el.attrib:
            mass_el.set("value", str(float(mass_el.get("value")) * mass_scale))

        inertia_el = inertial.find("inertia")
        if inertia_el is not None:
            for attr in _INERTIA_ATTRS:
                if attr in inertia_el.attrib:
                    inertia_el.set(attr, str(float(inertia_el.get(attr)) * mass_scale))

    out_dir = os.path.dirname(urdf_path)
    base = os.path.splitext(os.path.basename(urdf_path))[0]
    out_path = os.path.join(out_dir, f"{base}_mass{_scale_tag(mass_scale)}.urdf")
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


def total_mass(urdf_path: str) -> float:
    """Sum the ``<inertial><mass value="..."/>`` across every link in ``urdf_path``.

    Used only for logging (e.g. reporting the resulting object mass after scaling);
    not consumed by the simulator, which reads mass directly from the loaded URDF.
    """
    root = ET.parse(urdf_path).getroot()
    return sum(
        float(mass_el.get("value"))
        for mass_el in root.iter("mass")
        if "value" in mass_el.attrib
    )
