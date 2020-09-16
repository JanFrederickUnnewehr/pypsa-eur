# coding: utf-8
"""
Adds todays electrical generators and existing hydro storage units to a base network.

Relevant Settings
-----------------

.. code:: yaml

    costs:
        year:
        USD2013_to_EUR2013:
        dicountrate:
        emission_prices:

    electricity:
        max_hours:
        marginal_cost:
        capital_cost:
        conventional_carriers:
        co2limit:
        extendable_carriers:
            Generator:
        estimate_renewable_capacities_from_capacity_stats:

    load:
        scaling_factor:

    renewable: (keys)
        hydro:
            carriers:
            hydro_max_hours:
            hydro_capital_cost:

    lines:
        length_factor:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at :ref:`costs_cf`,
    :ref:`electricity_cf`, :ref:`load_cf`, :ref:`renewable_cf`, :ref:`lines_cf`

Inputs
------

- ``data/costs.csv``: The database of cost assumptions for all included technologies for specific years from various sources; e.g. discount rate, lifetime, investment (CAPEX), fixed operation and maintenance (FOM), variable operation and maintenance (VOM), fuel costs, efficiency, carbon-dioxide intensity.
- ``data/bundle/hydro_capacities.csv``: Hydropower plant store/discharge power capacities, energy storage capacity, and average hourly inflow by country.

    .. image:: ../img/hydrocapacities.png
        :scale: 34 %

- ``data/geth2015_hydro_capacities.csv``: alternative to capacities above; NOT CURRENTLY USED!
- ``data/bundle/time_series_60min_singleindex_filtered.csv``: Hourly per-country load profiles since 2010 from the `ENTSO-E statistical database <https://www.entsoe.eu/data/power-stats/hourly_load/>`_

    .. image:: ../img/load-box.png
        :scale: 33 %

    .. image:: ../img/load-ts.png
        :scale: 33 %

- ``resources/regions_onshore.geojson``: confer :ref:`busregions`
- ``resources/nuts3_shapes.geojson``: confer :ref:`shapes`
- ``resources/powerplants.csv``: confer :ref:`powerplants`
- ``resources/profile_{}.nc``: all technologies in ``config["renewables"].keys()``, confer :ref:`renewableprofiles`.
- ``networks/base.nc``: confer :ref:`base`

Outputs
-------

- ``networks/elec.nc``:

    .. image:: ../img/elec.png
            :scale: 33 %

Description
-----------

The rule :mod:`add_electricity` ties all the different data inputs from the preceding rules together into a detailed PyPSA network that is stored in ``networks/elec.nc``. It includes:

- today's transmission topology and transfer capacities (optionally including lines which are under construction according to the config settings ``lines: under_construction`` and ``links: under_construction``),
- today's thermal and hydro power generation capacities (for the technologies listed in the config setting ``electricity: conventional_carriers``), and
- today's load time-series (upsampled in a top-down approach according to population and gross domestic product)

It further adds extendable ``generators`` with **zero** capacity for

- photovoltaic, onshore and AC- as well as DC-connected offshore wind installations with today's locational, hourly wind and solar capacity factors (but **no** current capacities),
- additional open- and combined-cycle gas turbines (if ``OCGT`` and/or ``CCGT`` is listed in the config setting ``electricity: extendable_carriers``)
"""

from vresutils.costdata import annuity
#from vresutils.load import timeseries_opsd
from vresutils import transfer as vtransfer

import logging
logger = logging.getLogger(__name__)
from _helpers import configure_logging

import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
import pypsa
import powerplantmatching as ppm

idx = pd.IndexSlice


def normed(s): return s/s.sum()

def _add_missing_carriers_from_costs(n, costs, carriers):
    missing_carriers = pd.Index(carriers).difference(n.carriers.index)
    if missing_carriers.empty: return

    emissions_cols = costs.columns.to_series()\
                           .loc[lambda s: s.str.endswith('_emissions')].values
    suptechs = missing_carriers.str.split('-').str[0]
    emissions = costs.loc[suptechs, emissions_cols].fillna(0.)
    emissions.index = missing_carriers
    n.import_components_from_dataframe(emissions, 'Carrier')

def load_costs(Nyears=1., tech_costs=None, config=None, elec_config=None):
    if tech_costs is None:
        tech_costs = snakemake.input.tech_costs

    if config is None:
        config = snakemake.config['costs']

    # set all asset costs and other parameters
    costs = pd.read_csv(tech_costs, index_col=list(range(3))).sort_index()

    # correct units to MW and EUR
    costs.loc[costs.unit.str.contains("/kW"),"value"] *= 1e3
    costs.loc[costs.unit.str.contains("USD"),"value"] *= config['USD2013_to_EUR2013']

    costs = (costs.loc[idx[:,config['year'],:], "value"]
             .unstack(level=2).groupby("technology").sum(min_count=1))

    costs = costs.fillna({"CO2 intensity" : 0,
                          "FOM" : 0,
                          "VOM" : 0,
                          "discount rate" : config['discountrate'],
                          "efficiency" : 1,
                          "fuel" : 0,
                          "investment" : 0,
                          "lifetime" : 25})

    costs["capital_cost"] = ((annuity(costs["lifetime"], costs["discount rate"]) +
                             costs["FOM"]/100.) *
                             costs["investment"] * Nyears)

    costs.at['OCGT', 'fuel'] = costs.at['gas', 'fuel']
    costs.at['CCGT', 'fuel'] = costs.at['gas', 'fuel']

    costs['marginal_cost'] = costs['VOM'] + costs['fuel'] / costs['efficiency']

    costs = costs.rename(columns={"CO2 intensity": "co2_emissions"})

    costs.at['OCGT', 'co2_emissions'] = costs.at['gas', 'co2_emissions']
    costs.at['CCGT', 'co2_emissions'] = costs.at['gas', 'co2_emissions']

    costs.at['solar', 'capital_cost'] = 0.5*(costs.at['solar-rooftop', 'capital_cost'] +
                                             costs.at['solar-utility', 'capital_cost'])

    def costs_for_storage(store, link1, link2=None, max_hours=1.):
        capital_cost = link1['capital_cost'] + max_hours * store['capital_cost']
        efficiency = link1['efficiency']**0.5
        if link2 is not None:
            capital_cost += link2['capital_cost']
            efficiency *= link2['efficiency']**0.5
        return pd.Series(dict(capital_cost=capital_cost,
                              marginal_cost=0.,
                              efficiency=efficiency,
                              co2_emissions=0.))

    if elec_config is None:
        elec_config = snakemake.config['electricity']
    max_hours = elec_config['max_hours']
    costs.loc["battery"] = \
        costs_for_storage(costs.loc["battery storage"], costs.loc["battery inverter"],
                          max_hours=max_hours['battery'])
    costs.loc["H2"] = \
        costs_for_storage(costs.loc["hydrogen storage"], costs.loc["fuel cell"],
                          costs.loc["electrolysis"], max_hours=max_hours['H2'])

    for attr in ('marginal_cost', 'capital_cost'):
        overwrites = config.get(attr)
        if overwrites is not None:
            overwrites = pd.Series(overwrites)
            costs.loc[overwrites.index, attr] = overwrites

    return costs

def load_powerplants(ppl_fn=None):
    if ppl_fn is None:
        ppl_fn = snakemake.input.powerplants
    carrier_dict = {'ocgt': 'OCGT', 'ccgt': 'CCGT', 'bioenergy':'biomass',
                    'ccgt, thermal': 'CCGT', 'hard coal': 'coal'}
    return (pd.read_csv(ppl_fn, index_col=0, dtype={'bus': 'str'})
            .powerplant.to_pypsa_names()
            .rename(columns=str.lower).drop(columns=['efficiency'])
            .replace({'carrier': carrier_dict}))

def load_renewable_powerplants(re_ppl_fn=None):
    if re_ppl_fn is None:
        re_ppl_fn = snakemake.input.renewable_powerplants
    return (pd.read_csv(re_ppl_fn, usecols=['capacity', 'Country', 'Carrier', 'bus'], dtype={'bus': 'str'})
            .rename(columns=str.lower).rename(columns={'capacity': 'p_nom'}))

# =============================================================================
# Attach components
# =============================================================================

# ### Load

def attach_load(n):
    substation_lv_i = n.buses.index[n.buses['substation_lv']]
    regions = (gpd.read_file(snakemake.input.regions).set_index('name')
               .reindex(substation_lv_i))
    opsd_load = (pd.read_csv(snakemake.input.load, index_col=0, parse_dates=True)
                .filter(items=snakemake.config['countries']))
    
    # Scalling data according to scalling factor in config.yaml
    logger.info(f"Load data scalled with scalling factor {snakemake.config['load']['scaling_factor']}.")
    opsd_load = opsd_load * snakemake.config.get('load', {}).get('scaling_factor', 1.0)

    # Convert to naive UTC (has to be explicit since pandas 0.24)
    opsd_load.index = opsd_load.index.tz_localize(None)

    nuts3 = gpd.read_file(snakemake.input.nuts3_shapes).set_index('index')

    def normed(x): return x.divide(x.sum())

    def upsample(cntry, group):
        l = opsd_load[cntry]
        if len(group) == 1:
            return pd.DataFrame({group.index[0]: l})
        else:
            nuts3_cntry = nuts3.loc[nuts3.country == cntry]
            transfer = vtransfer.Shapes2Shapes(group, nuts3_cntry.geometry,
                                               normed=False).T.tocsr()
            gdp_n = pd.Series(transfer.dot(nuts3_cntry['gdp'].fillna(1.).values),
                              index=group.index)
            pop_n = pd.Series(transfer.dot(nuts3_cntry['pop'].fillna(1.).values),
                              index=group.index)

            # relative factors 0.6 and 0.4 have been determined from a linear
            # regression on the country to continent load data (refer to vresutils.load._upsampling_weights)
            factors = normed(0.6 * normed(gdp_n) + 0.4 * normed(pop_n))
            return pd.DataFrame(factors.values * l.values[:,np.newaxis],
                                index=l.index, columns=factors.index)

    load = pd.concat([upsample(cntry, group) for cntry, group
                      in regions.geometry.groupby(regions.country)], axis=1)

    n.madd("Load", substation_lv_i, bus=substation_lv_i, p_set=load)

### Set line costs

def update_transmission_costs(n, costs, length_factor=1.0, simple_hvdc_costs=False):
    n.lines['capital_cost'] = (n.lines['length'] * length_factor *
                               costs.at['HVAC overhead', 'capital_cost'])

    if n.links.empty: return

    dc_b = n.links.carrier == 'DC'
    if simple_hvdc_costs:
        costs = (n.links.loc[dc_b, 'length'] * length_factor *
                 costs.at['HVDC overhead', 'capital_cost'])
    else:
        costs = (n.links.loc[dc_b, 'length'] * length_factor *
                ((1. - n.links.loc[dc_b, 'underwater_fraction']) *
                costs.at['HVDC overhead', 'capital_cost'] +
                n.links.loc[dc_b, 'underwater_fraction'] *
                costs.at['HVDC submarine', 'capital_cost']) +
                costs.at['HVDC inverter pair', 'capital_cost'])
    n.links.loc[dc_b, 'capital_cost'] = costs

### Generators

def attach_wind_and_solar(n, costs, re_cap_country, re_ppl):
    re_cap_len = {}
    for tech in snakemake.config['renewable']:
        if tech == 'hydro': continue

        
        re_cap_len[tech] = 0
        if tech not in n.carriers.index:
            n.add("Carrier", name=tech)
        with xr.open_dataset(getattr(snakemake.input, 'profile_' + tech)) as ds:
            
            if ds.indexes['bus'].empty:
                continue

#            suptech = tech.split('-', 2)[0]
#            if suptech == 'offwind':
#                underwater_fraction = ds['underwater_fraction'].to_pandas()
#                connection_cost = (snakemake.config['lines']['length_factor'] *
#                                   ds['average_distance'].to_pandas() *
#                                   (underwater_fraction *
#                                    costs.at[tech + '-connection-submarine', 'capital_cost'] +
#                                    (1. - underwater_fraction) *
#                                    costs.at[tech + '-connection-underground', 'capital_cost']))
#                capital_cost = (costs.at['offwind', 'capital_cost'] +
#                                costs.at[tech + '-station', 'capital_cost'] +
#                                connection_cost)
#                logger.info("Added connection cost of {:0.0f}-{:0.0f} Eur/MW/a to {}"
#                            .format(connection_cost.min(), connection_cost.max(), tech))
#            else:
#                capital_cost = costs.at[tech, 'capital_cost']
                
            
            #distribution of todays renewable capacities according to the installed capacity in each country

            def normed(x): return x.divide(x.sum())
            
            countries = list(n.buses.country.unique())
            
            #all countrys in the network without renewable powerplants with locations
            countries_with_re = list(re_ppl[re_ppl.carrier == tech].country.unique())
            
            for i in countries_with_re:
                countries.remove(i)
            
            re_cap_bus=pd.DataFrame()
            
            
            # filter ds to countries without renewbales
            ds = ds.where(ds.bus.isin(n.buses.query("country in @countries").index))
            
            for country in countries:
                
                #print(tech)
                #print(country)
                
                #filter all busses (index) in one country
                index_country = n.buses.query("country == @country").index

                #CF for each bus in country
                CF_index_country = ds['profile'].to_pandas().query('index in @index_country').mean(axis=1).to_frame()
                
                #land availability for each bus in country
                potential_index_country = ds['p_nom_max'].to_pandas().to_frame().query('index in @index_country')

                #distribution -> 0.5 x CP + 0.5 x potential 
                #re_cap_bus = re_cap_bus.append(re_cap_country.loc[country][suptech] * normed(0.5*normed(ds['profile'].to_pandas().query('index in @index_country').mean(axis=1).to_frame())+0.5*normed(ds['p_nom_max'].to_pandas().to_frame().query('index in @index_country'))))

                # distribution -> CF^x * potential^y.
                
                x=2
                y=0.5
                
                re_cap_index_country = re_cap_country.loc[country][tech] * normed(CF_index_country.pow(x) * potential_index_country.pow(y))

                re_cap_bus = re_cap_bus.append(re_cap_index_country,verify_integrity=True)


            # add renewable capacities to the network
            
            logger.info('Adding {} generators with capacities\n{}'
                .format(tech, re_cap_bus.sum()[0]))

            
            n.madd("Generator", ds.indexes['bus'], ' ' + tech,
                   bus=ds.indexes['bus'],
                   carrier=tech,
                   p_nom=re_cap_bus[0],
                   weight=ds['weight'].to_pandas(),
                   marginal_cost=costs.at[tech, 'marginal_cost'],
                   efficiency=costs.at[tech, 'efficiency'],
                   p_max_pu=ds['profile'].transpose('time', 'bus').to_pandas())
            
            
#                   p_nom_extendable=False,
#                   p_nom_max=ds['p_nom_max'].to_pandas(),
#                   
#                   capital_cost=capital_cost,
# def attach_wind_and_solar(n, costs):
#     for tech in snakemake.config['renewable']:
#         if tech == 'hydro': continue

#         n.add("Carrier", name=tech)
#         with xr.open_dataset(getattr(snakemake.input, 'profile_' + tech)) as ds:
#             if ds.indexes['bus'].empty: continue

#             suptech = tech.split('-', 2)[0]
#             if suptech == 'offwind':
#                 underwater_fraction = ds['underwater_fraction'].to_pandas()
#                 connection_cost = (snakemake.config['lines']['length_factor'] *
#                                    ds['average_distance'].to_pandas() *
#                                    (underwater_fraction *
#                                     costs.at[tech + '-connection-submarine', 'capital_cost'] +
#                                     (1. - underwater_fraction) *
#                                     costs.at[tech + '-connection-underground', 'capital_cost']))
#                 capital_cost = (costs.at['offwind', 'capital_cost'] +
#                                 costs.at[tech + '-station', 'capital_cost'] +
#                                 connection_cost)
#                 logger.info("Added connection cost of {:0.0f}-{:0.0f} Eur/MW/a to {}"
#                             .format(connection_cost.min(), connection_cost.max(), tech))
#             elif suptech == 'onwind':
#                 capital_cost = (costs.at['onwind', 'capital_cost'] +
#                                 costs.at['onwind-landcosts', 'capital_cost'])
#             else:
#                 capital_cost = costs.at[tech, 'capital_cost']

#             n.madd("Generator", ds.indexes['bus'], ' ' + tech,
#                    bus=ds.indexes['bus'],
#                    carrier=tech,
#                    p_nom_extendable=True,
#                    p_nom_max=ds['p_nom_max'].to_pandas(),
#                    weight=ds['weight'].to_pandas(),
#                    marginal_cost=costs.at[suptech, 'marginal_cost'],
#                    capital_cost=capital_cost,
#                    efficiency=costs.at[suptech, 'efficiency'],
#                    p_max_pu=ds['profile'].transpose('time', 'bus').to_pandas())


def attach_wind_and_solar_with_locations(n, costs, re_ppl):
    
    for tech in snakemake.config['renewable']:
        if tech == 'hydro': continue

        if tech not in n.carriers.index:
            n.add("Carrier", name=tech)
        with xr.open_dataset(getattr(snakemake.input, 'profile_' + tech)) as ds:
            
            # if ds.indexes['bus'].empty:
            #     continue

            #add todays renewable capacities with locations to the network
            
            #filter re_ppl 
            re_ppl_tech = re_ppl.query('carrier == @tech')
            
            # agg. because dataframe is to large for merge function maybe for all techs
            
            re_ppl_tech = re_ppl_tech.groupby(['bus'], as_index=False).agg({'p_nom': 'sum', 'country': 'first', 'carrier': 'first'})
            
            #filter all busses in the network with renewbale powerplants           
            busses_re_ppl = pd.DataFrame()           
            busses_re_ppl = re_ppl_tech.bus.unique()

            #CF for each bus in the network
            busses_CF = ds['profile'].to_pandas().query('index in @busses_re_ppl')
            #reset index for merge function
            re_ppl_tech.reset_index(inplace=True)
            re_ppl_CF = re_ppl_tech[['index','bus']]

            re_ppl_CF = re_ppl_CF.merge(busses_CF, left_on='bus', right_on=busses_CF.index)
            
            re_ppl_CF.set_index('index', inplace=True)

            # add renewable capacities to the network
            
            logger.info('Adding {} generators with locations and capacities of\n{}'
                    .format(tech, re_ppl_tech.p_nom.sum()))
    
            n.madd("Generator", re_ppl_tech.index, ' ' + tech,
                       bus=re_ppl_tech.bus,
                       carrier=tech,
                       p_nom=re_ppl_tech.p_nom,
                       marginal_cost=costs.at[tech, 'marginal_cost'],
                       efficiency=costs.at[tech, 'efficiency'],
                       p_max_pu=re_ppl_CF.transpose())            



def attach_conventional_generators(n, costs, ppl):
    carriers = snakemake.config['electricity']['conventional_carriers']
    _add_missing_carriers_from_costs(n, costs, carriers)
    ppl = (ppl.query('carrier in @carriers').join(costs, on='carrier')
           .rename(index=lambda s: 'C' + str(s)))

    logger.info('Adding {} generators with capacities\n{}'
                .format(len(ppl), ppl.groupby('carrier').p_nom.sum()))
    n.madd("Generator", ppl.index,
           suffix='_C_Gen',
           carrier=ppl.carrier,
           bus=ppl.bus,
           p_nom=ppl.p_nom,
           efficiency=ppl.efficiency,
           marginal_cost=ppl.marginal_cost,
           capital_cost=0)
    logger.warning(f'Capital costs for conventional generators put to 0 EUR/MW.')

def attach_conventional_generator_profiles(n, ppl):

    carriers = snakemake.config['electricity']['conventional_carriers']
    _add_missing_carriers_from_costs(n, costs, carriers)
    
    profile_pp_data = xr.open_dataset(snakemake.input.profile_pp)

    profile_pp = profile_pp_data['profile'].to_pandas()
    
    pp_cap = pd.DataFrame.from_dict(profile_pp_data.profile.attrs, orient='index', columns=['capacity'])
    
    #pp_cap = pp_cap.rename(index=lambda s: 'C_p' + str(s))

    ppl = (ppl.join(costs, on='carrier'))#.rename(index=lambda s: 'C_p' + str(s)))
    
    ppl.carrier.loc[266] = 'CCGT' # small correction

    profile_pp = profile_pp.transpose()
    
    #profile_pp = profile_pp.rename(index=lambda s: 'C_p' + str(s))  
    
    ppl_index = profile_pp.index.tolist()
    
    ppl = ppl.query('index in @ppl_index')
    
    ppl.p_nom.update(other=pp_cap.capacity)

    logger.info('Adding {} generators and their profiles with capacities\n{}'
                .format(len(ppl), ppl.groupby('carrier').p_nom.sum()))
    
    n.madd("Generator", ppl.index,
           suffix='_C_Gen_prof',
           carrier=ppl.carrier,
           bus=ppl.bus,
           p_nom=ppl.p_nom,
           efficiency=1,
           marginal_cost=0,
           capital_cost=0,
           p_max_pu=profile_pp.transpose(),
           p_min_pu=(profile_pp.transpose()-0.000001))
    
    return ppl_index

def attach_hydro(n, costs, ppl):
    if 'hydro' not in snakemake.config['renewable']: return
    c = snakemake.config['renewable']['hydro']
    carriers = c.get('carriers', ['ror', 'PHS', 'hydro'])

    _add_missing_carriers_from_costs(n, costs, carriers)

    ppl = ppl.query('carrier == "hydro"').reset_index(drop=True)\
             .rename(index=lambda s: str(s) + ' hydro')
    ror = ppl.query('technology == "Run-Of-River"')
    phs = ppl.query('technology == "Pumped Storage"')
    hydro = ppl.query('technology == "Reservoir"')

    country = ppl['bus'].map(n.buses.country).rename("country")
    
    # get indices from ror and hydro for inflow of water (combine both indices of dataframe)
    inflow_idx = ror.index | hydro.index
    if not inflow_idx.empty:
        dist_key = ppl.loc[inflow_idx, 'p_nom'].groupby(country).transform(normed)

        with xr.open_dataarray(snakemake.input.profile_hydro) as inflow:
            inflow_countries = pd.Index(country[inflow_idx])
            missing_c = (inflow_countries.unique()
                         .difference(inflow.indexes['countries']))
            assert missing_c.empty, (f"'{snakemake.input.profile_hydro}' is missing "
                f"inflow time-series for at least one country: {', '.join(missing_c)}")
                       
            # inflow data from era data for each country, is distributed over all power plant by there p_nom values
            # inflow in MW oder auch MW/h kommt darauf an wie inflow erzeugt wurde
            inflow_t = (inflow.sel(countries=inflow_countries)
                        .rename({'countries': 'name'})
                        .assign_coords(name=inflow_idx)
                        .transpose('time', 'name')
                        .to_pandas()
                        .multiply(dist_key, axis=1))
            
            # verfügbare ror einsepisung wird über länder inflow erzeugt
    if 'ror' in carriers and not ror.empty:
        n.madd("Generator", ror.index,
               carrier='ror',
               bus=ror['bus'],
               p_nom=ror['p_nom'],
               efficiency=costs.at['ror', 'efficiency'],
               capital_cost=costs.at['ror', 'capital_cost'],
               weight=ror['p_nom'],
               p_max_pu=(inflow_t[ror.index]
                         .divide(ror['p_nom'], axis=1)
                         .where(lambda df: df<=1., other=1.)))

    if 'PHS' in carriers and not phs.empty:
        # fill missing max hours to config value and assume no natural inflow
        # due to lack of data
        phs = phs.replace({'max_hours': {0: c['PHS_max_hours']}})
        n.madd('StorageUnit', phs.index,
               carrier='PHS',
               bus=phs['bus'],
               p_nom=phs['p_nom'],
               capital_cost=costs.at['PHS', 'capital_cost'],
               max_hours=phs['max_hours'],
               efficiency_store=np.sqrt(costs.at['PHS','efficiency']),
               efficiency_dispatch=np.sqrt(costs.at['PHS','efficiency']),
               cyclic_state_of_charge=True)

    if 'hydro' in carriers and not hydro.empty:
        hydro_max_hours = c.get('hydro_max_hours')
        hydro_stats = pd.read_csv(snakemake.input.hydro_capacities,
                                   comment="#", na_values='-', index_col=0)
        e_target = hydro_stats["E_store[TWh]"].clip(lower=0.2) * 1e6
        e_installed = hydro.eval('p_nom * max_hours').groupby(hydro.country).sum()
        e_missing = e_target - e_installed
        missing_mh_i = hydro.query('max_hours == 0').index

        if hydro_max_hours == 'energy_capacity_totals_by_country':
            # watch out some p_nom values like IE's are totally underrepresented
            max_hours_country = e_missing / \
                                hydro.loc[missing_mh_i].groupby('country').p_nom.sum()

        elif hydro_max_hours == 'estimate_by_large_installations':
            max_hours_country = hydro_stats['E_store[TWh]'] * 1e3 / \
                                hydro_stats['p_nom_discharge[GW]']

        missing_countries = (pd.Index(hydro['country'].unique())
                             .difference(max_hours_country.dropna().index))
        if not missing_countries.empty:
            logger.warning("Assuming max_hours=6 for hydro reservoirs in the countries: {}"
                           .format(", ".join(missing_countries)))
        hydro_max_hours = hydro.max_hours.where(hydro.max_hours > 0,
                                hydro.country.map(max_hours_country)).fillna(6)


        n.madd('StorageUnit', hydro.index, carrier='hydro',
               bus=hydro['bus'],
               p_nom=hydro['p_nom'],
               max_hours=hydro_max_hours,
               capital_cost=(costs.at['hydro', 'capital_cost']
                             if c.get('hydro_capital_cost') else 0.),
               marginal_cost=costs.at['hydro', 'marginal_cost'],
               p_max_pu=1.,  # dispatch
               p_min_pu=0.,  # no storing alloud store
               efficiency_dispatch=costs.at['hydro', 'efficiency'],
               efficiency_store=0.,
               cyclic_state_of_charge=True,
               inflow=inflow_t.loc[:, hydro.index])


def attach_extendable_generators(n, costs, ppl):
    elec_opts = snakemake.config['electricity']
    carriers = pd.Index(elec_opts['extendable_carriers']['Generator'])
    _add_missing_carriers_from_costs(n, costs, carriers)

    for tech in carriers:
        if tech.startswith('OCGT'):
            ocgt = ppl.query("carrier in ['OCGT', 'CCGT']").groupby('bus', as_index=False).first()
            n.madd('Generator', ocgt.index,
                   suffix=' OCGT',
                   bus=ocgt['bus'],
                   carrier=tech,
                   p_nom_extendable=True,
                   p_nom=0.,
                   capital_cost=costs.at['OCGT', 'capital_cost'],
                   marginal_cost=costs.at['OCGT', 'marginal_cost'],
                   efficiency=costs.at['OCGT', 'efficiency'])

        elif tech.startswith('CCGT'):
            ccgt = ppl.query("carrier in ['OCGT', 'CCGT']").groupby('bus', as_index=False).first()
            n.madd('Generator', ccgt.index,
                   suffix=' CCGT',
                   bus=ccgt['bus'],
                   carrier=tech,
                   p_nom_extendable=True,
                   p_nom=0.,
                   capital_cost=costs.at['CCGT', 'capital_cost'],
                   marginal_cost=costs.at['CCGT', 'marginal_cost'],
                   efficiency=costs.at['CCGT', 'efficiency'])

        elif tech.startswith('nuclear'):
            nuclear = ppl.query("carrier == 'nuclear'").groupby('bus', as_index=False).first()
            n.madd('Generator', nuclear.index,
                suffix=' nuclear',
                bus=nuclear['bus'],
                carrier=tech,
                p_nom_extendable=True,
                p_nom=0.,
                capital_cost=costs.at['nuclear', 'capital_cost'],
                marginal_cost=costs.at['nuclear', 'marginal_cost'],
                efficiency=costs.at['nuclear', 'efficiency'])

        else:
            raise NotImplementedError(f"Adding extendable generators for carrier "
                                      "'{tech}' is not implemented, yet. "
                                      "Only OCGT, CCGT and nuclear are allowed at the moment.")


def estimate_renewable_capacities(n, tech_map=None):
    if tech_map is None:
        tech_map = (snakemake.config['electricity']
                    .get('estimate_renewable_capacities_from_capacity_stats', {}))

    if len(tech_map) == 0: return

    capacities = (ppm.data.Capacity_stats().powerplant.convert_country_to_alpha2()
                  [lambda df: df.Energy_Source_Level_2]
                  .set_index(['Fueltype', 'Country']).sort_index())

    countries = n.buses.country.unique()

    for ppm_fueltype, techs in tech_map.items():
        tech_capacities = capacities.loc[ppm_fueltype, 'Capacity']\
                                    .reindex(countries, fill_value=0.)
        tech_i = n.generators.query('carrier in @techs').index
        n.generators.loc[tech_i, 'p_nom'] = (
            (n.generators_t.p_max_pu[tech_i].mean() *
             n.generators.loc[tech_i, 'p_nom_max']) # maximal yearly generation
             .groupby(n.generators.bus.map(n.buses.country)) # for each country
             .transform(lambda s: normed(s) * tech_capacities.at[s.name])
             .where(lambda s: s>0.1, 0.))  # only capacities above 100kW

def add_nice_carrier_names(n, config=None):
    if config is None: config = snakemake.config
    carrier_i = n.carriers.index
    nice_names = (pd.Series(config['plotting']['nice_names'])
                  .reindex(carrier_i).fillna(carrier_i.to_series().str.title()))
    n.carriers['nice_name'] = nice_names
    colors = pd.Series(config['plotting']['tech_colors']).reindex(carrier_i)
    if colors.isna().any():
        missing_i = list(colors.index[colors.isna()])
        logger.warning(f'tech_colors for carriers {missing_i} not defined '
                       'in config.')
    n.carriers['color'] = colors
    

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('add_electricity_today')
    configure_logging(snakemake)
    
    #loading pypsa Network and simulation year
    n = pypsa.Network(snakemake.input.base_network)
    Nyears = n.snapshot_weightings.sum()/8760.

    #loading powerplant data and cost data
    costs = load_costs(Nyears)
    ppl = load_powerplants()
    re_ppl = load_renewable_powerplants()
    
    #attach components to pypsa network
    
    #load
    attach_load(n)
    
    #conventianal ppl and available profiles also hydro    
    ppl_index = attach_conventional_generator_profiles(n, ppl)
    
    #filter ppl without profiles

    ppl = ppl.query('index not in @ppl_index')

    #attach ppl without profiles
    attach_conventional_generators(n, costs, ppl)
  
    #wind and solar with exact locataions
    attach_wind_and_solar_with_locations(n, costs, re_ppl)
    
    
    
    re_cap_country = pd.read_csv(snakemake.input.re_capacity, encoding='Latin-1',skiprows=3,thousands=',',index_col='Country',usecols=['Of which Solar PV', 'Of which Wind onshore','Of which Wind offshore','Country'])

    re_cap_country.rename(columns={'Of which Solar PV': 'solar',
                                'Of which Wind onshore': 'onwind',
                                'Of which Wind offshore': 'offwind'
                                }, inplace = True)
    
    
    attach_wind_and_solar(n, costs, re_cap_country, re_ppl)
    
    
    attach_hydro(n, costs, ppl)
    
    
    update_transmission_costs(n, costs)
    attach_extendable_generators(n, costs, ppl)

    estimate_renewable_capacities(n)
    add_nice_carrier_names(n)

    n.export_to_netcdf(snakemake.output[0])
