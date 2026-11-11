# %%
"""
Create a layer model for geophysics modelling in late 2026.

Install required packages:

"""

import logging
import pathlib

import nlmod
from nhflodata.get_paths import get_abs_data_path
from nhflotools.nhi_chloride import get_nhi_chloride_concentration
from nhflotools.pwnlayers.layers import get_top_from_ahn
from nhflotools.pwnlayers3.layers import get_pwn_layer_model

logging.basicConfig(level=logging.INFO)

# layer and grid options
transport = True  # Not tested without transport
extent = [94000, 110000, 498000, 527000]  # hydraulica calibration and geophysics modelling extent.
delr = delc = 100  # or 5x5m as long as it connects to REGIS=100x100m

# %% create folder structure
ws = pathlib.Path(__file__).parent.absolute() / "model_ws"
_, cachedir = nlmod.util.get_model_dirs(ws)

data_path_2024 = get_abs_data_path(name="bodemlagen_pwn_2024", version="latest", location="get_from_env")
data_path_chloride = get_abs_data_path(name="nhi_chloride_concentration", version="latest", location="get_from_env")

# %% Set up grid and retrieve REGIS
layer_model_regis_struc = nlmod.read.regis.get_combined_layer_models(
    extent,
    use_regis=True,
    use_geotop=False,
    remove_nan_layers=False,
    cachedir=cachedir,
    cachename="layer_model",
)

ds_regis = nlmod.to_model_ds(
    ds=layer_model_regis_struc,
    model_name="pwn",
    model_ws=ws,
    delr=delr,
    delc=delc,
    fill_nan=True,
    extrapolate=True,
    remove_nan_layers=False,
    transport=transport,
    # fill_value_kh/kv fill both-NaN REGIS-gap cells; nlmod's defaults (kh=1.0/kv=0.1) give a
    # 10x too-low aquitard kv. anisotropy mirrors get_pwn_layer_model (#121).
    anisotropy=10.0,
    fill_value_kh=5.0,
    fill_value_kv=1.0,
)

ds_regis = nlmod.grid.refine(
    ds_regis,
    model_ws=ws,
    remove_nan_layers=False,
    refinement_features=[],
)

# Use ds with refined grid to get AHN, which comes on a much finer grid
ahn = nlmod.read.ahn.download_ahn5(
    extent=ds_regis.extent,
    identifier="AHN5 maaiveldmodel (DTM) 5m",
    cachedir=cachedir,
    cachename="ahn",
)
ds_regis["ahn"] = nlmod.dims.resample.structured_da_to_ds(ahn, ds_regis, method="average")
top = get_top_from_ahn(
    ds_regis,
    replace_surface_water_with_peil=True,
    replace_northsea_with_constant=0.0,
    method_elsewhere="nearest",
    cachedir=cachedir,
)

# %% Get PWN layer model
fname_koppeltabel = os.path.join(data_path_2024, "bodemlagenvertaaltabel.csv")

ds = get_pwn_layer_model(
    ds_regis=ds_regis,
    data_path_2024=data_path_2024,
    fname_koppeltabel=fname_koppeltabel,
    top=top,
    anisotropy=10.0,
    distance_transition=250.0,
    fix_min_layer_thickness=True,
    fill_value_kh=5.0,
    fill_value_kv=1.0,
    return_diagnostics=True,  # required by plot_diagnostic_cross_sections (cat_botm etc.)
    cachedir=cachedir,
    cachename="pwn_layer_model",
)
ds["ahn"] = ds_regis["ahn"]

if ds.transport:
    ds["chloride"] = get_nhi_chloride_concentration(ds, data_path_chloride, cachedir=cachedir, cachename="chloride")
    ds = nlmod.gwt.prepare.set_default_transport_parameters(ds, transport_type="chloride")
