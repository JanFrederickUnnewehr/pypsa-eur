# coding: utf-8
"""
Retrieves renewable powerplant capacities and locations from ..., assigns these to buses and creates a ``.csv`` file. It is possible to amend the powerplant database with custom entries provided in ``data/custom_renewable_powerplants.csv``.

Relevant Settings
-----------------

.. code:: yaml

    electricity:
      powerplants_filter:
      custom_powerplants:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`electricity`

Inputs
------

- ``networks/base.nc``: confer :ref:`base`.
- ``data/custom_powerplants.csv``: custom powerplants in the same format as `powerplantmatching <https://github.com/FRESNA/powerplantmatching>`_ provides

Outputs
-------

- ``resource/powerplants.csv``: A list of conventional power plants (i.e. neither wind nor solar) with fields for name, fuel type, technology, country, capacity in MW, duration, commissioning year, retrofit year, latitude, longitude, and dam information as documented in the `powerplantmatching README <https://github.com/FRESNA/powerplantmatching/blob/master/README.md>`_; additionally it includes information on the closest substation/bus in ``networks/base.nc``.

    .. image:: ../img/powerplantmatching.png
        :scale: 30 %

    **Source:** `powerplantmatching on GitHub <https://github.com/FRESNA/powerplantmatching>`_

Description
-----------

The configuration options ``electricity: powerplants_filter`` and ``electricity: custom_powerplants`` can be used to control whether data should be retrieved from the original powerplants database or from custom amendmends. These specify `pandas.query <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html>`_ commands.

1. Adding all powerplants from custom:

    .. code:: yaml

        powerplants_filter: false
        custom_powerplants: true

2. Replacing powerplants in e.g. Germany by custom data:

    .. code:: yaml

        powerplants_filter: Country not in ['Germany']
        custom_powerplants: true

    or

    .. code:: yaml

        powerplants_filter: Country not in ['Germany']
        custom_powerplants: Country in ['Germany']


3. Adding additional built year constraints:

    .. code:: yaml

        powerplants_filter: Country not in ['Germany'] and YearCommissioned <= 2015
        custom_powerplants: YearCommissioned <= 2015

"""

import logging
logger = logging.getLogger(__name__)
from _helpers import configure_logging

from scipy.spatial import cKDTree as KDTree

import pypsa
import powerplantmatching as pm
import pandas as pd
import numpy as np
import glob

# def add_custom_re_powerplants(re_ppl):
#     custom_ppl_query = snakemake.config['electricity']['custom_powerplants']
#     if not custom_ppl_query:
#         return re_ppl
#     add_ppls = pd.read_csv(snakemake.input.custom_powerplants, index_col=0,
#                            dtype={'bus': 'str'})
#     if isinstance(custom_ppl_query, str):
#         add_ppls.query(custom_ppl_query, inplace=True)
#     return re_ppl.append(add_ppls, sort=False, ignore_index=True, verify_integrity=True)


if __name__ == "__main__":

    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_renewable_powerplants')
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.base_network)
    countries = n.buses.country.unique()
    
    

    
    re_ppl = pd.read_csv(snakemake.input.installed_renewable_capacities, usecols=['commissioning_date', 'decommissioning_date',
                                                                                 'technology', 'electrical_capacity', 'federal_state',
                                                                                 'postcode', 'municipality_code', 'municipality', 
                                                                                 'address', 'lat', 'lon', 'data_source', 'comment'],
                         parse_dates=['commissioning_date', 'decommissioning_date'], encoding='utf-8')
    
    tech_dict = {'Photovoltaics ground': 'solar', 'Photovoltaics': 'solar', 'Onshore':'onshore', 'Offshore': 'offshore'}
    
    re_ppl = re_ppl.replace({'technology': tech_dict})
    
    re_ppl = re_ppl.rename(columns={'electrical_capacity': 'capacity'})
    
    re_ppl = re_ppl.query('technology in ["onshore", "offshore", "solar"]')
    
    re_ppl = re_ppl[re_ppl.commissioning_date < n.snapshots[-1]]
    
    re_ppl = re_ppl[~(re_ppl.decommissioning_date < n.snapshots[-1])]
    
    
    

    
    
    #re_ppl = re_ppl[re_ppl.commissioning_date.loc[n.snapshots[-1]]]

    # #re_ppl = pd.read_csv(snakemake.input.installed_renewable_capacities, index_col=2)
    
    # re_ppl['Country'] = 'DE'
    # re_ppl['Carrier'] = 'onwind'


    # # ppl_query = snakemake.config['electricity']['powerplants_filter']
    # # if isinstance(ppl_query, str):
    # #     ppl.query(ppl_query, inplace=True)

    # # ppl = add_custom_powerplants(ppl) # add carriers from own powerplant files

    # cntries_without_re_ppl = [c for c in countries if c not in re_ppl.Country.unique()]

    # for c in countries:
    #     substation_i = n.buses.query('substation_lv and country == @c').index
    #     kdtree = KDTree(n.buses.loc[substation_i, ['x','y']].values)
    #     re_ppl_i = re_ppl.query('Country == @c').index

    #     tree_i = kdtree.query(re_ppl.loc[re_ppl_i, ['lon','lat']].values)[1]
    #     re_ppl.loc[re_ppl_i, 'bus'] = substation_i.append(pd.Index([np.nan]))[tree_i]

    # if cntries_without_re_ppl:
    #     logging.warning(f"No renewable powerplants known in: {', '.join(cntries_without_re_ppl)}")

    # bus_null_b = re_ppl["bus"].isnull()
    # if bus_null_b.any():
    #     logging.warning(f"Couldn't find close bus for {bus_null_b.sum()} renewable powerplants")

    # re_ppl.to_csv(snakemake.output[0])
