from frb.surveys import survey_utils as su
from frb.surveys import catalog_utils as cu

import os
import tqdm
from typing import Dict, Optional, Sequence, Tuple

from importlib_resources import files

from astropy import visualization as vis
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy import units as u

import numpy as np
import matplotlib.pyplot as plt

from sparcl.client import SparclClient

from scipy.signal import medfilt
from scipy.interpolate import interp1d, UnivariateSpline

import legacystamps


DEFAULT_NEBULAR_LINES = {
    "[OII]3726": 3726.03,
    "[OII]3729": 3728.82,
    "Hb": 4861.33,
    "[OIII]4959": 4958.91,
    "[OIII]5007": 5006.84,
    "Ha": 6562.80,
    "[NII]6548": 6548.05,
    "[NII]6583": 6583.45,
    "[SII]6716": 6716.44,
    "[SII]6731": 6730.82,
}


def _empty_spectrum_table() -> Table:
    """Return an empty spectrum table with the canonical column schema."""
    return Table(
        names=["wavelength", "flux", "error", "flux_norm", "error_norm", "flux_clip", "error_clip"],
        dtype=[float] * 7,
    )

def get_desi_qso_table(query=None, overwrite=False, output_file=None):
    """
    Get the DESI QSO catalog as an astropy Table.

    Args:
        query (str, optional): Custom SQL query to execute. If None, a default query 
            will be used to retrieve all QSOs with good redshifts (zwarn=0).
        overwrite (bool, optional): If True, overwrite the output file if it already exists.
        output_file (str, optional): Path to save the output catalog. If None, it
            will be saved to "data/desi_qsos.csv" within the package.
    Returns:
        astropy.table.Table: Catalog of DESI QSOs.
    """
    if output_file is None:
            output_file = files('gotoq').joinpath("data/tables/desi_qsos.csv")

    if os.path.exists(output_file) and not overwrite:
        desi_qsos = Table.read(output_file, format="csv")
    else:
        if query is None:
            query_str ="""
                SELECT
                    targetid, mean_fiber_ra AS ra, mean_fiber_dec AS dec, z, zerr, survey, zcat_primary as DESI_zcat_primary, zwarn
                FROM
                    desi_dr1.zpix
                WHERE
                    spectype = 'QSO' AND
                    zwarn = 0
                """
        else:
            query_str = query

        coord = SkyCoord(ra=0*u.degree, dec=0*u.degree, frame='icrs')
        survey = su.load_survey_by_name("DESI", coord, radius=180*u.degree) # The coord and radius are not important here since we are doing a custom query
        desi_qsos = survey.get_catalog(query = query_str, zcat_primary_only=False)

        # Save the catalog to a file        
        desi_qsos.write(output_file, format="csv", overwrite=True)
    
    return desi_qsos

def make_cutout_png(img, hdr, save_path: str, title: Optional[str] = None, cmap: str = "gray") -> None:
    """Render and save a WCS cutout PNG with sigma-clipped stretch."""
    wcs = WCS(hdr)
    _, med, std = sigma_clipped_stats(img, sigma=3.0)
    norm = vis.ImageNormalize(vmin=med - std, vmax=med + 100 * std, stretch=vis.LogStretch())

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": wcs})
    ax.imshow(img, origin="lower", cmap=cmap, norm=norm)
    ax.set_title(title if title is not None else "DESI QSO Cutout")
    ax.set_xlabel("R.A.")
    ax.set_ylabel("Dec.")
    fig.savefig(save_path)
    plt.close(fig)

def download_legacy_stamp(
    ra: float,
    dec: float,
    output_dir: str,
    output_name: Optional[str] = None,
    bands: str = "grz",
    size: u.Quantity = 60 * u.arcsec,
    mode: str = "jpeg",
    overwrite: bool = True,
) -> str:
    """Download a Legacy Survey stamp and optionally rename it.

    Returns:
        str: Path to the downloaded (or renamed) file.
    """
    os.makedirs(output_dir, exist_ok=True)
    size_deg = float(size.to(u.deg).value) if hasattr(size, "to") else float(size)
    download_path = legacystamps.download(
        ra=ra,
        dec=dec,
        mode=mode,
        bands=bands,
        size=size_deg,
        ddir=os.path.abspath(output_dir),
    )

    if output_name is None:
        return download_path

    renamed_path = os.path.join(output_dir, output_name)
    if os.path.exists(renamed_path):
        if overwrite:
            os.remove(renamed_path)
        else:
            return renamed_path
    os.rename(download_path, renamed_path)
    return renamed_path


def batch_download_legacy_stamps(
    table: Table,
    ra_col: str = "ra",
    dec_col: str = "dec",
    id_col: str = "DESI_ID",
    output_dir: str = "desi_qso_cutouts",
    suffix: str = "_cutout.jpeg",
    bands: str = "grz",
    size: u.Quantity = 60 * u.arcsec,
    skip_existing: bool = True,
) -> Dict[str, str]:
    """Batch download Legacy Survey cutouts keyed by object id.

    Returns:
        dict: Mapping from object ID to file path.
    """
    file_map: Dict[str, str] = {}
    os.makedirs(output_dir, exist_ok=True)
    for row in tqdm.tqdm(table):
        object_id = str(row[id_col])
        out_name = f"{object_id}{suffix}"
        out_path = os.path.join(output_dir, out_name)
        if skip_existing and os.path.isfile(out_path):
            file_map[object_id] = out_path
            continue
        file_map[object_id] = download_legacy_stamp(
            ra=float(row[ra_col]),
            dec=float(row[dec_col]),
            output_dir=output_dir,
            output_name=out_name,
            bands=bands,
            size=size,
            mode="jpeg",
            overwrite=True,
        )
    return file_map


def clip_qso_features(
    wave: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    z: float,
    lines: Optional[Dict[str, float]] = None,
    dv_clip: float = 400,
    absorption_threshold: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mask strong QSO line regions and deep absorption regions in normalized spectra."""
    if lines is None:
        lines = DEFAULT_NEBULAR_LINES

    flux_clip = flux.copy()
    err_clip = err.copy()

    for lam0 in lines.values():
        lam_obs = lam0 * (1 + z)
        dlam = lam_obs * dv_clip / 3e5
        mask = np.abs(wave - lam_obs) < dlam
        flux_clip[mask] = np.nan
        err_clip[mask] = np.nan

    absorption_mask = flux < absorption_threshold
    flux_clip[absorption_mask] = np.nan
    err_clip[absorption_mask] = np.nan
    return flux_clip, err_clip


def continuum_normalize(
    wave: np.ndarray,
    flux: np.ndarray,
    error: np.ndarray,
    kernel_win: float = 5,
    node_spacing: float = 2.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Continuum-normalize a spectrum with median filtering and spline fitting."""
    dw = float(np.median(np.diff(wave)))
    kernel_size = int(kernel_win // dw)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(kernel_size, 3)
    if kernel_size >= len(wave):
        kernel_size = len(wave) - 1 if len(wave) % 2 == 0 else len(wave)
        kernel_size = max(kernel_size, 3)

    flux_med = medfilt(flux, kernel_size)
    residuals = flux - flux_med

    idx = (
        (error > 0)
        & np.isfinite(error)
        & np.isfinite(flux)
        & (residuals / error > -1.5)
        & (residuals / error < 2.0)
    )
    if np.sum(idx) < 10:
        raise ValueError("Too few continuum points")

    wav_cont_init = wave[idx]
    cont_init = flux[idx]
    cont_med = medfilt(cont_init, kernel_size)
    cont_interp = interp1d(
        wav_cont_init,
        cont_med,
        bounds_error=False,
        fill_value=(cont_med[0], cont_med[-1]),
    )

    wave_nodes = np.arange(wave[0], wave[-1] + node_spacing, node_spacing)
    cont_nodes = cont_interp(wave_nodes)
    cont_spline = UnivariateSpline(wave_nodes, cont_nodes, s=0)
    continuum = cont_spline(wave)

    flux_norm = flux / continuum
    error_norm = error / continuum
    return flux_norm, error_norm


def download_desi_spectrum_raw(
    target_id: int,
    spec_client: Optional[SparclClient] = None,
):
    """Retrieve raw DESI spectrum arrays from SPARCL for a target id.

    Returns:
        tuple: (wavelength, flux, ivar) or (None, None, None) when unavailable.
    """
    client = spec_client if spec_client is not None else SparclClient(announcement=False)
    results = client.retrieve_by_specid(
        specid_list=[int(target_id)],
        include=["specid", "wavelength", "flux", "ivar"],
    ).records
    if len(results) == 0:
        return None, None, None
    return results[0].wavelength, results[0].flux, results[0].ivar


def process_desi_spectrum(
    wave: np.ndarray,
    flux: np.ndarray,
    ivar: Optional[np.ndarray] = None,
    error: Optional[np.ndarray] = None,
    z: Optional[float] = None,
    lines: Optional[Dict[str, float]] = None,
    dv_clip: float = 400,
    normalize: bool = True,
) -> Table:
    """Process raw DESI spectrum arrays into standardized table columns."""
    if error is None:
        if ivar is None:
            raise ValueError("Either 'ivar' or 'error' must be provided")
        with np.errstate(divide="ignore", invalid="ignore"):
            error = np.where(ivar > 0, 1 / np.sqrt(ivar), np.nan)

    if normalize:
        flux_norm, err_norm = continuum_normalize(wave, flux, error)
    else:
        flux_norm = np.asarray(flux, dtype=float)
        err_norm = np.asarray(error, dtype=float)

    if z is not None:
        flux_clip, err_clip = clip_qso_features(wave, flux_norm, err_norm, z, lines=lines, dv_clip=dv_clip)
    else:
        flux_clip = np.asarray(flux_norm, dtype=float)
        err_clip = np.asarray(err_norm, dtype=float)

    return Table(
        [wave, flux, error, flux_norm, err_norm, flux_clip, err_clip],
        names=["wavelength", "flux", "error", "flux_norm", "error_norm", "flux_clip", "error_clip"],
    )


def download_desi_spectrum(
    target_id: int,
    output_file: Optional[str] = None,
    z: Optional[float] = None,
    spec_client: Optional[SparclClient] = None,
    lines: Optional[Dict[str, float]] = None,
    dv_clip: float = 400,
    normalize: bool = True,
    overwrite: bool = True,
) -> Table:
    """Download, process, and optionally persist a DESI spectrum table."""
    try:
        wave, flux, ivar = download_desi_spectrum_raw(target_id, spec_client=spec_client)
        if wave is None:
            return _empty_spectrum_table()

        spec_tab = process_desi_spectrum(
            np.asarray(wave),
            np.asarray(flux),
            ivar=np.asarray(ivar),
            z=z,
            lines=lines,
            dv_clip=dv_clip,
            normalize=normalize,
        )
        if output_file is not None:
            spec_tab.write(output_file, format="ascii.commented_header", overwrite=overwrite)
        return spec_tab
    except Exception as exc:
        print(f"Error retrieving spectrum for target ID {target_id}: {exc}")
        return _empty_spectrum_table()

