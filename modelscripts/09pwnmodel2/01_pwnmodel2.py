"""
Modelscript used to create a groundwater model for the PWN area.

The model is based on the REGIS layer model and
contains the following features:
- Recharge from KNMI data
- Major surface water bodies as CHD and GHB
- Chloride transport
- PWN extraction wells
- Polder drains
- Infiltration from panden
- TATA steel wells
- Sea level rise
- Starting head based on AHN.
"""

import logging
import os
import pathlib

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import nlmod
import numpy as np
import pandas as pd
import xarray as xr
from nhflodata.get_paths import get_abs_data_path
from nhflotools.major_surface_waters import (
    chd_ghb_from_major_surface_waters,
    get_chd_ghb_data_from_major_surface_waters,
)
from nhflotools.nhi_chloride import get_nhi_chloride_concentration
from nhflotools.panden import riv_from_oppervlakte_pwn
from nhflotools.polder import drn_from_waterboard_data
from nhflotools.pwnlayers.layers import get_pwn_layer_model, get_top_from_ahn
from nhflotools.well import get_wells_pwn_dataframe

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if not nlmod.util.check_presence_mfbinaries():
    nlmod.download_mfbinaries()


# scenario
model_name = "reference"
sealvl_ref = 0.04  # reference level relative to NAP
sealvl_2050 = sealvl_ref
sealvl_2100 = sealvl_ref
transport = True  # Not tested without transport
panden = True
polders_in_detail = False  # Not implemented
use_knmi_recharge = False  # else use constant recharge
tata = True

# layer and grid options
extent = (99000, 103000, 500000, 505000)
delr = delc = 500.0

rechs = [0.0007, 0.0009, 0.0011, 0.0013, 0.0015]

# %% time settings
# Single steady state time step afterwhich many transient time steps follow
perlen_tr = 90.0  # Duration transient time steps
perlen_ss = 365.0  # Duration steady state time step
tstart_tr = pd.Timestamp("2023-01-01")  # start transient period
tend = pd.Timestamp("2100-01-02")  # approximate end transient period

# %% create folder structure
try:
    ws = os.path.join(pathlib.Path(__file__).parent.absolute(), "model_ws")
except NameError:
    # When running via pytest
    ws = pathlib.Path("model_ws").absolute()

figdir, cachedir = nlmod.util.get_model_dirs(ws)

# %% Setup time and space grid and retreive REGIS
"""
Order:
- Get REGIS ds using nlmod.read.regis.get_combined_layer_models() and nlmod.to_model_ds()
- Refine grid with surface water polygons and areas of interest with nlmod.grid.refine()
- Get AHN with nlmod.read.ahn.get_ahn4() and resample to model grid with nlmod.dims.resample.structured_da_to_ds()
- Get PWN layer model with nlmod.read.pwn.get_pwn_layer_model()
"""
layer_model_regis_struc = nlmod.read.regis.get_combined_layer_models(
    extent,
    use_regis=True,
    use_geotop=False,
    remove_nan_layers=False,
    cachedir=cachedir,
    cachename="layer_model",
)

ds_regis = nlmod.to_model_ds(
    layer_model_regis_struc,
    "pwn",
    ws,
    delr=delr,
    delc=delc,
    remove_nan_layers=False,
    transport=transport,
)

# Now refine the grid
ds_regis = nlmod.grid.refine(ds_regis, model_ws=ws, remove_nan_layers=False)

# Use ds with refined grid to get AHN
# `ahn` contains has a much finer grid than ds
ahn = nlmod.read.ahn.get_ahn4(extent=ds_regis.extent, cachedir=cachedir, cachename="ahn")
ds_regis["ahn"] = nlmod.dims.resample.structured_da_to_ds(ahn, ds_regis, method="average")
top = get_top_from_ahn(
    ds_regis,
    replace_surface_water_with_peil=True,
    replace_northsea_with_constant=0.0,
    method_elsewhere="nearest",
    cachedir=cachedir,
)

# %% Get PWN layer model
# Use ds with refined grid to get PWN layer model
data_path_mensink = get_abs_data_path(name="bodemlagen_pwn_nhdz", version="latest", location="get_from_env")
data_path_panden = get_abs_data_path(
    name="oppervlaktewater_pwn_shapes_panden", version="latest", location="get_from_env"
)
data_path_bergen = get_abs_data_path(name="bodemlagen_pwn_bergen", version="latest", location="get_from_env")
data_path_chloride = get_abs_data_path(name="nhi_chloride_concentration", version="latest", location="get_from_env")
data_path_wells_pwn = get_abs_data_path(name="wells_pwn", version="latest", location="get_from_env")
data_path_wells_tata = get_abs_data_path(name="wells_tata", version="latest", location="get_from_env")
data_path_koppeltabel = get_abs_data_path(
    name="bodemlagen_pwn_regis_koppeltabel", version="latest", location="get_from_env"
)
fname_koppeltabel = os.path.join(data_path_koppeltabel, "bodemlagenvertaaltabelv2.csv")

ds = get_pwn_layer_model(
    ds_regis=ds_regis,
    data_path_mensink=data_path_mensink,
    data_path_bergen=data_path_bergen,
    fname_koppeltabel=fname_koppeltabel,
    top=top,
    length_transition=100.0,
    cachedir=cachedir,
    cachename="pwn_layer_model",
)
ds["ahn"] = ds_regis["ahn"]

# %% Time settings
tstart_ss = tstart_tr - pd.Timedelta(perlen_ss, "D")
model_period_tr = (tend - tstart_tr).days
nper_tr = int(model_period_tr / perlen_tr)
perlens = np.ones(nper_tr + 1, dtype=float) * perlen_tr
perlens[0] = perlen_ss
steady = np.zeros(nper_tr + 1, dtype=bool)
steady[0] = True

ds = nlmod.time.set_ds_time(
    ds,
    start=tstart_ss,
    steady=steady,
    perlen=perlens,
)

# %% Add recharge
if use_knmi_recharge:
    # download knmi recharge data
    knmi_ds = nlmod.read.knmi.get_recharge(ds, cachedir=cachedir, cachename="knmi")
    ds.update(knmi_ds)
else:
    ds["recharge"] = xr.DataArray(
        dims=("time", "icell2d"),
        data=np.ones(shape=(ds.sizes["time"], ds.sizes["icell2d"])) * 0.0007,
        attrs={"units": "m/d"},
    )

# %% add major surface water bodies
ds.update(get_chd_ghb_data_from_major_surface_waters(ds, da_name="rws_oppwater", cachedir=None))

# %% Add Chloride
# Requires major surface water bodies to be added prior.
if ds.transport == 1:
    # load chloride data
    ds["chloride"] = get_nhi_chloride_concentration(ds, data_path_chloride, cachedir=cachedir, cachename="chloride")

    # add transport parameters to model
    ds = nlmod.gwt.prepare.set_default_transport_parameters(ds, transport_type="chloride")

# %% PWN Onttrekkingen
weltype = "wel"  # or "maw"
wdf = get_wells_pwn_dataframe(data_path_wells_pwn, flow_product="median")

# %% Add wells TATA steel
if tata:
    # Add well TATA steel zout
    gdf_tata_zout = gpd.read_file(os.path.join(data_path_wells_tata, "tata_zoutwaterbronnen.geojson"), driver="GeoJSON")
    gdf_tata_zout["x"] = gdf_tata_zout["geometry"].x
    gdf_tata_zout["y"] = gdf_tata_zout["geometry"].y
    gdf_tata_zout["Q"] = -gdf_tata_zout["Q_m3/d"] / len(gdf_tata_zout)

    # Add concentration for infiltration
    gdf_tata_zout["CONCENTRATION"] = 0.0

    ds["thickness"] = nlmod.dims.calculate_thickness(ds)
    kd = ds["kh"] * ds["thickness"]

    # Add well TATA steel zoet
    kd_zoet_layer = 100.0  # m2/d
    cl_max_zoet_layer = 1000  # mg/l
    gdf_tata_zoet = gpd.read_file(os.path.join(data_path_wells_tata, "tata_zoetwaterbronnen.geojson"), driver="GeoJSON")
    gdf_tata_zoet["x"] = gdf_tata_zoet["geometry"].x
    gdf_tata_zoet["y"] = gdf_tata_zoet["geometry"].y
    gdf_tata_zoet["Q"] = -gdf_tata_zoet["Q_m3/d"] / len(gdf_tata_zoet)

    # Vertically align the zoetwaterbronnen with the fresh aquifer
    gdf_tata_zoet["CONCENTRATION"] = 0.0

    gdf_tata_zoet["botm"] = np.nan
    gdf_tata_zoet["top"] = np.nan

    for name, row in gdf_tata_zoet.iterrows():
        inearest = np.sqrt((ds.x - row["x"]) ** 2 + (ds.y - row["y"]) ** 2).argmin(dim="icell2d")
        lay = np.where(kd.isel(icell2d=inearest) > kd_zoet_layer)[0][0]
        cl = ds["chloride"].isel(icell2d=inearest)[lay]

        if cl > cl_max_zoet_layer:
            msg = "Unable to place zoetwaterbron in fresh aquifer. => Placing TATA zoetwaterbron in saline aquifer."
            msg += f" Cl: {cl:.2f} mg/l, kd: {kd.isel(icell2d=inearest)[lay]:.2f} m2/d, ilayer: {lay}, "
            msg += f"layername: {ds.layer[lay].item()}, xwell: {row['x']}, ywell: {row['y']}, modelextent: {ds.extent}"
            logging.warning(msg)

        gdf_tata_zoet.loc[name, "botm"] = ds["botm"].isel(icell2d=inearest)[lay] + 0.001
        gdf_tata_zoet.loc[name, "top"] = ds["botm"].isel(icell2d=inearest)[lay - 1] - 0.001

# %% Add starting head
# Set starting head with a maximum of 5 m NAP
large_number = 1e5
starting_head = xr.where(ds["northsea"] == 1, 0.0, ds["ahn"])
starting_head = xr.where(starting_head > large_number, -0.4, starting_head)
starting_head = np.clip(starting_head, a_max=5.0, a_min=-np.inf)
ds["starting_head"] = xr.zeros_like(ds["botm"])
for ilay in range(ds.sizes["layer"]):
    ds["starting_head"].values[ilay] = starting_head.fillna(0.0)

if ds.transport:
    # Convert starting head to equivalent freshwater head given chloride distribution
    ds["starting_head"] = nlmod.gwt.output.freshwater_head(ds, ds["starting_head"], ds["chloride"])

# %% Configure Flopy
# Save dataset to netCDF file
ds.to_netcdf(os.path.join(cachedir, "model_ds.nc"))

# Create simulation
sim = nlmod.sim.sim(ds)

# Create time discretisation
tdis = nlmod.sim.tdis(ds, sim)

# Create ims
ims_gwf = nlmod.sim.ims(sim)

# Create groundwater flow model
gwf = nlmod.gwf.gwf(ds, sim)

# Create discretization
dis_gwf = nlmod.gwf.disv(ds, gwf)

# create node property flow
npf = nlmod.gwf.npf(ds, gwf, save_flows=True)

# Create storage
sto = nlmod.gwf.sto(ds, gwf, sy=0.2, ss=1e-5, save_flows=True)

# Create the initial conditions
ic_gwf = nlmod.gwf.ic(ds, gwf, starting_head="starting_head")

# Create output control package
oc_gwf = nlmod.gwf.oc(ds, gwf)

# Build recharge package
rch = nlmod.gwf.rch(ds, gwf, mask=ds["northsea"] == 0)

# %% Sea level rise
dt2050 = (pd.Timestamp("2050-01-01") - tstart_ss).days
dt2100 = (pd.Timestamp("2100-01-01") - tstart_ss).days + 0.1  # for interpolation on bound
sea_lvl_ts = [
    (0.0, sealvl_ref),
    (perlen_ss, sealvl_ref),
    (dt2050, sealvl_2050),
    (dt2100, sealvl_2100),
]

chd, ghb, ts_sea = chd_ghb_from_major_surface_waters(ds, gwf, sea_stage=sea_lvl_ts)

# Build drain package
if polders_in_detail:
    msg = "Polders in detail not implemented"
    raise NotImplementedError(msg)
    fname_bgt = os.path.join(cachedir, "bgt.gpkg")

    if not os.path.isfile(fname_bgt):
        bgt = nlmod.read.bgt.get_bgt(extent, make_valid=True)
        bgt = nlmod.gwf.add_min_ahn_to_gdf(bgt, ahn, buffer=5.0, column="ahn_min")
        la = nlmod.gwf.surface_water.download_level_areas(bgt, extent=extent, raise_exceptions=False)
        bgt = nlmod.gwf.surface_water.add_stages_from_waterboards(bgt, la=la)
        bgt.to_file(fname_bgt)

    else:
        bgt = gpd.read_file(fname_bgt)
else:
    drn = drn_from_waterboard_data(ds=ds, gwf=gwf, wb="Hollands Noorderkwartier", cbot=1.0)


# Build infiltration panden river package
if panden:
    riv = riv_from_oppervlakte_pwn(ds, gwf, data_path_panden)

# Build well/maw package
if weltype == "wel":
    wel = nlmod.gwf.wells.wel_from_df(
        wdf,
        gwf,
        top="bovenkant_filter",
        botm="onderkant_filter",
        Q="Q",
        aux="CONCENTRATION",
        boundnames="locatie",
        save_flows=True,
    )
    if wel.stress_period_data.data is None:
        gwf.remove_package(wel.name[0])
else:
    maw = nlmod.gwf.wells.maw_from_df(
        wdf,
        gwf,
        top="bovenkant_filter",
        botm="onderkant_filter",
        Q="Q",
        boundnames="locatie",
    )
    if maw.stress_period_data.data is None:
        gwf.remove_package(maw.name[0])

if tata:
    # Add well tata
    weltatazout = nlmod.gwf.wells.wel_from_df(
        gdf_tata_zout,
        gwf,
        aux="CONCENTRATION",
        boundnames="Name",
        save_flows=True,
        pname="zout_tata",
    )
    weltatazoet = nlmod.gwf.wells.wel_from_df(
        gdf_tata_zoet,
        gwf,
        aux="CONCENTRATION",
        boundnames="Name",
        save_flows=True,
        pname="zoet_tata",
    )
    if weltatazout.stress_period_data.data is None:
        gwf.remove_package(weltatazout.name[0])
    if weltatazoet.stress_period_data.data is None:
        gwf.remove_package(weltatazoet.name[0])

# Create groundwater transport model
if ds.transport:
    buy = nlmod.gwf.buy(ds, gwf)
    gwt = nlmod.gwt.gwt(ds, sim)
    ims_gwt = nlmod.sim.ims(sim, pname="ims_gwt", filename=f"{gwt.name}.ims")
    nlmod.sim.register_ims_package(sim, gwt, ims_gwt)
    dis_gwt = nlmod.gwt.disv(ds, gwt)
    ic_gwt = nlmod.gwt.ic(ds, gwt, "chloride")
    adv = nlmod.gwt.adv(ds, gwt)
    dsp = nlmod.gwt.dsp(ds, gwt)
    mst = nlmod.gwt.mst(ds, gwt)
    ssm = nlmod.gwt.ssm(ds, gwt)
    cnc = nlmod.gwt.cnc(
        ds,
        gwt,
        da_mask=(ds["northsea"] == 1).expand_dims({"layer": [ds.layer.data[0]]}),
        da_conc=ds["chloride"].isel(layer=slice(0, 1)),
    )
    if cnc.stress_period_data.data is None:
        gwt.remove_package(cnc.name[0])
    oc_gwt = nlmod.gwt.oc(ds, gwt)
    gwfgwt = nlmod.gwt.gwfgwt(ds, sim)


# %% Verify modellayers
def compare_layer_models(
    ds1,
    line,
    colors,
    ds2=None,
    zmin=-200.0,
    zmax=10.0,
    min_label_area=1000,
    title1="REGIS original",
    title2="Modified layers",
    xlabel="Distance along x-sec (m)",
    ylabel="m NAP",
):
    """
    Compare two layer models along a line.

    Parameters
    ----------
    ds1 : xarray.Dataset
        Dataset with original layer model
    line : list
        List with two tuples with x and y coordinates of the line
    colors : list
        List with colors for the layers
    ds2 : xarray.Dataset, optional
        Dataset with modified layer model, by default None
    zmin : float, optional
        Minimum z value, by default -200.
    zmax : float, optional
        Maximum z value, by default 10.
    min_label_area : int, optional
        Minimum area for layer labels, by default 1000
    title1 : str, optional
        Title for first plot, by default "REGIS original"
    title2 : str, optional
        Title for second plot, by default "Modified layers"
    xlabel : str, optional
        Label for x-axis, by default "Distance along x-sec (m)"
    ylabel : str, optional
        Label for y-axis, by default "m NAP"
    """
    if ds2 is None:
        _fig, ax1 = plt.subplots(1, 1, figsize=(14, 6))
    else:
        _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

    dcs1 = nlmod.plot.DatasetCrossSection(ds1, line=line, ax=ax1, zmin=zmin, zmax=zmax)
    dcs1.plot_layers(colors=colors, min_label_area=min_label_area)
    dcs1.plot_grid(linewidth=0.5, vertical=False)
    ax1.set_ylabel(ylabel)

    if ds2 is not None:
        ax1.set_title(title1)
        dcs2 = nlmod.plot.DatasetCrossSection(ds2, line=line, ax=ax2, zmin=zmin, zmax=zmax)
        dcs2.plot_layers(colors=colors, min_label_area=min_label_area)
        dcs2.plot_grid(linewidth=0.5, vertical=False)
        ax2.set_ylabel(ylabel)
        ax2.set_xlabel(xlabel)
        ax2.set_title(title2)
    else:
        ax1.set_xlabel(xlabel)


ym = (ds.attrs["extent"][2] + ds.attrs["extent"][3]) / 2
line = [(ds.attrs["extent"][0], ym), (ds.attrs["extent"][1], ym)]
# %% Write and run the model
nlmod.sim.write_and_run(sim, ds)

# %% Postprocessing
# Get heads
ds["head"] = nlmod.gwf.output.get_heads_da(ds, fname=os.path.join(ws, f"{ds.attrs['model_name']}.hds"))
ds["head_filled"] = ds["head"].bfill(dim="layer")

# load concentration results
conc = nlmod.gwt.output.get_concentration_da(ds, fname=os.path.join(ws, f"{ds.attrs['model_name']}_gwt.ucn"))
ctop = nlmod.gwt.output.get_concentration_at_gw_surface(conc)

ds["concentration"] = conc
ds["conc_filled"] = ds["concentration"].bfill(dim="layer")

# fresh water head
ds["freshwater_head"] = nlmod.gwt.output.freshwater_head(ds, ds["head_filled"], ds["conc_filled"], denseref=1000.0)

# calculate grensvlak
threshold_fresh = 1000
threshold_brakkish = 8000

conbool_fresh = ds["concentration"] > threshold_fresh
ds["ilay_grensvlak"] = conbool_fresh.argmax(axis=1) - 1
grensvlak = ds["botm"].isel(layer=ds["ilay_grensvlak"])
grensvlak = xr.where(ds["ilay_grensvlak"] == -1, ds["top"], grensvlak)
grensvlak.attrs["threshold"] = threshold_fresh
ds["grensvlak_zoet"] = grensvlak

conbool_brakkish = ds["concentration"] > threshold_brakkish
ds["ilay_grensvlak"] = conbool_brakkish.argmax(axis=1) - 1
grensvlak = ds["botm"].isel(layer=ds["ilay_grensvlak"])
grensvlak = xr.where(ds["ilay_grensvlak"] == -1, ds["top"], grensvlak)
grensvlak.attrs["threshold"] = threshold_brakkish
ds["grensvlak_brak"] = grensvlak

# %% verandering gewogen gemiddelde concentratie
ds["thickness"] = nlmod.dims.calculate_thickness(ds)
concentration_thick = ds["concentration"] * ds["thickness"]
concentration_mean = concentration_thick.sum(dim="layer") / ds["thickness"].sum(dim="layer")
dconcentration_meant0 = concentration_mean - concentration_mean[0]

ds["concentration_mean"] = concentration_mean
ds["dconcentration_mean"] = dconcentration_meant0


#  %% Plot
fig, ax = nlmod.plot.get_map(ds.extent, base=1e4)
nlmod.plot.modelgrid(ds, ax=ax, color="k", lw=0.5, alpha=0.5)
nlmod.plot.add_background_map(ax, map_provider="nlmaps.water", alpha=0.8)
fig.savefig(os.path.join(figdir, "doorsnedelijnen.png"), bbox_inches="tight", dpi=150)

if "drn_elev" in ds:
    f, ax = nlmod.plot.get_map(extent, base=1e4)
    ax.set_aspect("equal", adjustable="box")
    pc = nlmod.plot.data_array(ds["drn_elev"], ds=ds)
    nlmod.plot.modelgrid(ds, ax=ax, lw=0.25, alpha=0.5, color="k")
    ax.set_title("Oppervlaktewater, infiltratie en onttrekkingen")
    f.savefig(os.path.join(figdir, "oppervlaktewater.png"), bbox_inches="tight", dpi=150)

ilay = 0
iper = gwf.nper - 1

f, ax = nlmod.plot.get_map(extent, base=1e4)
ax.set_aspect("equal", adjustable="box")
pc = nlmod.plot.data_array(
    ds["freshwater_head"].isel(time=iper, layer=ilay),
    ds=ds,
    norm=mpl.colors.Normalize(-5, 5),
    cmap="Spectral_r",
)
nlmod.plot.modelgrid(ds, ax=ax, lw=0.25, alpha=0.5, color="k")
t = pd.Timestamp(ds.time.isel(time=iper).values[()])
ax.set_title(f"$h_f$, layer={ilay}, t={t.year}-{t.month:02g}")
cbar = f.colorbar(pc, ax=ax, shrink=0.8)
cbar.set_label("freshwater head [m NAP]")
ax.set_xlabel("X [km RD]")
ax.set_ylabel("Y [km RD]")
f.savefig(
    os.path.join(figdir, f"map_head_L{ilay}_t{iper}.png"),
    bbox_inches="tight",
    dpi=150,
)

f, ax = nlmod.plot.get_map(extent, base=1e4)
ax.set_aspect("equal", adjustable="box")
pc = nlmod.plot.data_array(
    ctop.isel(time=iper),
    ds=ds,
    norm=mpl.colors.Normalize(0, 5_000.0),
    cmap="RdYlGn_r",
)
nlmod.plot.modelgrid(ds, ax=ax, lw=0.25, alpha=0.5, color="k")
t = pd.Timestamp(ds.time.isel(time=iper).values[()])
ax.set_title(f"concentration, layer={ilay}, t={t.year}-{t.month:02g}")
cbar = f.colorbar(pc, ax=ax, shrink=0.8)
cbar.set_label("concentration [mg Cl-/L]")
# bgt.plot(ax=ax, edgecolor="k", facecolor="none")
ax.set_xlabel("X [km RD]")
ax.set_ylabel("Y [km RD]")
f.savefig(os.path.join(figdir, f"map_conc_L{ilay}_t{iper}.png"), bbox_inches="tight", dpi=150)

for gv in ["zoet", "brak"]:
    if gv == "zoet":
        thresh = threshold_fresh
    elif gv == "brak":
        thresh = threshold_brakkish

    f, ax = nlmod.plot.get_map(extent, base=1e4)
    ax.set_aspect("equal", adjustable="box")
    pc = nlmod.plot.data_array(
        ds[f"grensvlak_{gv}"].isel(time=iper),
        ds=ds,
        norm=mpl.colors.Normalize(-200, 20.0),
        cmap="RdYlBu_r",
    )
    nlmod.plot.modelgrid(ds, ax=ax, lw=0.25, alpha=0.5, color="k")
    t = pd.Timestamp(ds.time.isel(time=iper).values[()])
    ax.set_title(f"grensvlak {gv} (cl={thresh:.0f}mg/l), t={t.year}-{t.month:02g}")
    cbar = f.colorbar(pc, ax=ax, shrink=0.8)
    cbar.set_label("grensvlak [m NAP]")
    # bgt.plot(ax=ax, edgecolor="k", facecolor="none")
    ax.set_xlabel("X [km RD]")
    ax.set_ylabel("Y [km RD]")
    f.savefig(
        os.path.join(figdir, f"grensvlak_{gv}_t{iper}.png"),
        bbox_inches="tight",
        dpi=150,
    )
