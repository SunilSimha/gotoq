from frb.surveys import survey_utils as su
from frb.surveys import catalog_utils as cu

import os
import tqdm

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

