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
    
       
    re_ppl = pd.read_csv(snakemake.input.installed_renewable_capacities_DE, sep=',', usecols=['commissioning_date', 'decommissioning_date',
                                                                                 'technology', 'electrical_capacity', 'federal_state',
                                                                                 'postcode', 'municipality', 
                                                                                 'address', 'lat', 'lon'],
                                                                                 parse_dates=['commissioning_date', 'decommissioning_date'], encoding='utf-8')
    
    tech_dict = {'Photovoltaics ground': 'solar', 'Photovoltaics': 'solar', 'Onshore':'onshore', 'Offshore': 'offshore'}
    
    re_ppl = re_ppl.replace({'technology': tech_dict})
    
    re_ppl = re_ppl.rename(columns={'electrical_capacity': 'capacity'})
    
    re_ppl = re_ppl.query('technology in ["solar"]') # "offshore", "onshore"]
    
    re_ppl = re_ppl[re_ppl.commissioning_date < n.snapshots[-1]]
    
    re_ppl = re_ppl[~(re_ppl.decommissioning_date < n.snapshots[-1])]
    
     
    # find coordinates for some solar entries
    
    re_ppl_solar = re_ppl.query('technology == "solar"').copy()
     
    solar_isna_index = re_ppl_solar[re_ppl_solar.lat.isna()].index
    
    for i in solar_isna_index:
        re_ppl_solar_i = re_ppl_solar.loc[i]
        re_ppl_solar_i_loc = pm.utils.parse_Geoposition(location=re_ppl_solar_i[['federal_state', 'municipality', 'address']].to_list(), country=['Germany'])
        try:
            re_ppl_solar.at[i, 'lat'] = re_ppl_solar_i_loc.lat
            re_ppl_solar.at[i, 'lon'] = re_ppl_solar_i_loc.lon
        except AttributeError:
            re_ppl_solar_i_loc = pm.utils.parse_Geoposition(location=re_ppl_solar_i['federal_state'], country=['Germany'])
            try:
                re_ppl_solar.at[i, 'lat'] = re_ppl_solar_i_loc.lat
                re_ppl_solar.at[i, 'lon'] = re_ppl_solar_i_loc.lon
            except AttributeError:
                pass
            
    
            
    re_ppl_solar['Country'] = 'DE'
    re_ppl_solar['Carrier'] = 'solar'
    
    del re_ppl

    # # bei vielen offshore windparls fehlen die Koordinaten.
    # # erster Ansatz diese zu erstellen ist zu afuwenig. Siehe follgenden Ansatz
    
    #re_ppl_offshore = re_ppl.query('technology == "offshore"').copy()
    
    # # find index where lat is nan and capacity is 6.263999999999999 this turbines belomgs to Gode Wind 1 & 2 (in totla 97 turbines)
    # # https://en.wikipedia.org/wiki/List_of_offshore_wind_farms_in_Germany
    
    # Gode_Wind_i = re_ppl_offshore[re_ppl_offshore.lat.isna() & (re_ppl_offshore.capacity == 6.263999999999999)].index
    
    # re_ppl_offshore.at[Gode_Wind_i, 'lat'] = 54.05
    # re_ppl_offshore.at[Gode_Wind_i, 'lon'] = 7.016667
    
    
    # # find index where lat is nan and capacity is 6.15 and commissioning_date.dt.year <= 2015 this turbines belomgs to Nordsee_Ost_i (in totla 48 turbines)
    # # https://en.wikipedia.org/wiki/List_of_offshore_wind_farms_in_Germany
    
    # Nordsee_Ost_i = re_ppl_offshore[re_ppl_offshore.lat.isna() & (re_ppl_offshore.capacity == 6.15) & (re_ppl_offshore.commissioning_date.dt.year <= 2015)].index
    
    # re_ppl_offshore.at[Nordsee_Ost_i, 'lat'] = 54.44
    # re_ppl_offshore.at[Nordsee_Ost_i, 'lon'] = 7.68
    
    
    
    # # find index where lat is nan and capacity is 6.15 and commissioning_date.dt.year > 2015 and the re_ppl_offshore.federal_state == 'Ausschließliche Wirtschaftszone'
    # # this turbines belomgs to Nordsee_One (in totla 54 turbines)
    # # https://en.wikipedia.org/wiki/List_of_offshore_wind_farms_in_Germany
    # Nordsee_One_i = re_ppl_offshore[re_ppl_offshore.lat.isna() & (re_ppl_offshore.capacity == 6.15) & (re_ppl_offshore.commissioning_date.dt.year > 2015) &  (re_ppl_offshore.federal_state == 'Ausschließliche Wirtschaftszone')].index

    # re_ppl_offshore.at[Nordsee_One_i, 'lat'] = 53.978889
    # re_ppl_offshore.at[Nordsee_One_i, 'lon'] = 6.813889    
    

    # # find index where lat is nan and capacity is 6.15 and commissioning_date.dt.year > 2015 and the re_ppl_offshore.federal_state == 'Ausschließliche Wirtschaftszone'
    # # this turbines belomgs to Nordergründe (in totla 18 turbines)
    # #https://de.wikipedia.org/wiki/Offshore-Windpark_Nordergr%C3%BCnde
    # Norder_gruende_i = re_ppl_offshore[re_ppl_offshore.lat.isna() & (re_ppl_offshore.capacity == 6.15) & (re_ppl_offshore.commissioning_date.dt.year > 2015) &  (re_ppl_offshore.federal_state == 'Niedersachsen')].index

    # re_ppl_offshore.at[Norder_gruende_i, 'lat'] = 53.844319
    # re_ppl_offshore.at[Norder_gruende_i, 'lon'] = 8.163276  
    
    
    
    # alpha_ventus_i = re_ppl_offshore[re_ppl_offshore.lat.isna() & (re_ppl_offshore.commissioning_date.dt.year <= 2010) & (re_ppl_offshore.federal_state == 'Niedersachsen')].index

    # re_ppl_offshore.at[alpha_ventus_i, 'lat'] = 54.008333
    # re_ppl_offshore.at[alpha_ventus_i, 'lon'] = 6.598333 
    
    # Riffgat_i = re_ppl_offshore[re_ppl_offshore.lat.isna() & (re_ppl_offshore.capacity == 3.78) &  (re_ppl_offshore.federal_state == 'Niedersachsen')].index

    # re_ppl_offshore.at[Riffgat_i, 'lat'] = 53.69
    # re_ppl_offshore.at[Riffgat_i, 'lon'] = 6.48


    # Nordergruende = re_ppl_offshore[re_ppl_offshore.lat.isna() & (re_ppl_offshore.capacity == 3.78) &  (re_ppl_offshore.federal_state == 'Niedersachsen')].index

    # re_ppl_offshore.at[Riffgat_i, 'lat'] = 53.69
    # re_ppl_offshore.at[Riffgat_i, 'lon'] = 6.48    
    
        
    

    
    
    #re_ppl = re_ppl[re_ppl.commissioning_date.loc[n.snapshots[-1]]]

    re_ppl_onwind = pd.read_csv(snakemake.input.installed_renewable_capacities_DE_onwind, index_col=2)
    
    re_ppl_onwind['Country'] = 'DE'
    re_ppl_onwind['Carrier'] = 'onwind'
    re_ppl_onwind = re_ppl_onwind.rename(columns={'Capacity': 'capacity'})



    offwind_DE = pd.read_html('https://en.wikipedia.org/wiki/List_of_offshore_wind_farms_in_Germany', header=0, match="Riffgat")[0]

    offwind_DE = pd.read_html('https://de.wikipedia.org/wiki/Liste_der_deutschen_Offshore-Windparks', header=0, match="Riffgat")[0]
    offwind_DE = offwind_DE[~offwind_DE['Leistung(MW)'].str.contains('Nordsee')]
    offwind_DE = offwind_DE[~offwind_DE['Leistung(MW)'].str.contains('Ostsee')]
    offwind_DE = offwind_DE[offwind_DE['Status'].str.contains('in Betrieb')]
    offwind_DE = offwind_DE.astype({'Leistung(MW)': 'int32', 'Inbetrieb­nahme(Jahr)': 'int32', 'AnzahlWKAs' : 'int32'})
    

    def extract_coordinates(string):
        characters = ['°',',','″','′']
        for character in characters:
            string = string.replace(character,'')
        string = string.replace(' ','\xa0')
        string=string.split('\xa0')
        e = string
        sign = {'N':1,'S':-1,'O':1,'W':-1}
        lat = (float(e[0]) + (float(e[1]) + float(e[2])/60.)/60.)*sign[e[3]]
        lon = (float(e[4]) + (float(e[5]) + float(e[6])/60.)/60.)*sign[e[3]]
        return lon, lat

    offwind_DE['Koordinaten'] = offwind_DE['Koordinaten'].apply(extract_coordinates)
    offwind_DE[['lon', 'lat']] = pd.DataFrame(offwind_DE['Koordinaten'].tolist(), index=offwind_DE.index) 
    offwind_DE['Country'] = 'DE'
    offwind_DE['Carrier'] = 'offwind'
    offwind_DE.rename(columns={'Leistung(MW)': 'capacity'}, inplace=True)
    offwind_DE = offwind_DE[offwind_DE['Inbetrieb­nahme(Jahr)'] < (n.snapshots[-1].year+1)]
    
    # re_ppl_offwind = pd.read_csv(snakemake.input.installed_renewable_capacities_DE_offwind, usecols=['Capacity','YearCommissioned','lat', 'lon'], parse_dates=['YearCommissioned'])

    # re_ppl_offwind['Country'] = 'DE'
    # re_ppl_offwind['Carrier'] = 'offwind'
    # re_ppl_offwind = re_ppl_offwind.rename(columns={'Capacity': 'capacity'})
    # re_ppl_offwind = re_ppl_offwind[re_ppl_offwind.YearCommissioned < n.snapshots[-1]]
    # #convert to MW
    # re_ppl_offwind.capacity = re_ppl_offwind.capacity / 1000
    
    # re_ppl = pd.concat([re_ppl_solar, re_ppl_onwind, re_ppl_offwind], ignore_index=True)

    re_ppl = pd.concat([re_ppl_solar, re_ppl_onwind, offwind_DE], ignore_index=True)

    cntries_without_re_ppl = [c for c in countries if c not in re_ppl.Country.unique()]
    
    # find bus for solar and onwind

    for c in countries:
        substation_i = n.buses.query('substation_lv and country == @c').index
        kdtree = KDTree(n.buses.loc[substation_i, ['x','y']].values)
        re_ppl_i = re_ppl.query('Country == @c and Carrier in ["onwind", "solar"]').index

        tree_i = kdtree.query(re_ppl.loc[re_ppl_i, ['lon','lat']].values)[1]
        re_ppl.loc[re_ppl_i, 'bus'] = substation_i.append(pd.Index([np.nan]))[tree_i]

    # find bus for offshore
    import geokit as gk
    
    regions = gk.vector.extractFeatures(snakemake.input.regions_offshore, onlyAttr=True)
    regions.set_index('name', inplace=True)

    for c in countries:
        #substation_i = n.buses.query('substation_off and country == @c').index
        substation_i = regions.query('country == @c').index
        kdtree = KDTree(n.buses.loc[substation_i, ['x','y']].values)
        re_ppl_i = re_ppl.query('Country == @c and Carrier in ["offwind"]').index

        tree_i = kdtree.query(re_ppl.loc[re_ppl_i, ['lon','lat']].values)[1]
        re_ppl.loc[re_ppl_i, 'bus'] = substation_i.append(pd.Index([np.nan]))[tree_i]

    if cntries_without_re_ppl:
        logging.warning(f"No renewable powerplants with location known in: {', '.join(cntries_without_re_ppl)}")

    bus_null_b = re_ppl["bus"].isnull()
    if bus_null_b.any():
        logging.warning(f"Couldn't find close bus for {bus_null_b.sum()} renewable powerplants")

    re_ppl.to_csv(snakemake.output[0])