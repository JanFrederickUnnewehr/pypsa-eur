"""
Retrieves actual generation per generation unit from
`entsoe Transparency Platform <https://transparency.entsoe.eu/generation/r2/actualGenerationPerGenerationUnit/show>`_,
and assigns these to powerplants from powerplantmaching and creates power plant profiles.

Relevant Settings
-----------------

.. code:: yaml
`

Inputs
------

- `data/ActualGenerationOutputPerUnit`
  'data/JRC_OPEN_UNITS.csv'

Outputs
-------

- ``resources/profile_pp.nc``: A data set of conventional power plant generation (i.e. neither wind nor solar) with fields for fresan_id .


Description
-----------


"""

import logging
logger = logging.getLogger(__name__)
from _helpers import configure_logging


import glob
import numpy as np
import xarray as xr
import pandas as pd
import powerplantmatching as pm
import progressbar

def load_powerplants(ppl_fn=None):
    if ppl_fn is None:
        ppl_fn = snakemake.input.powerplants

    return (pd.read_csv(ppl_fn, index_col=0, dtype={'bus': 'str'}).pipe(pm.utils.projectID_to_dict))

def powerplant_generation (pp_id, generation_data, ppl):       
    
    """
    Provides power plant time series data from ENTSO-E with fresna ID.
    
    Parameters
    ----------
    pp_id : int
        Power plant ID from powerplantmatching (fresna_id)
    
    generation_data: pd.DataFrame
        DataFrame with generation per power plant unit

    ppl : pd.DataFrame
	    DataFram with the power plant database (powerplantmatching)
        
    Returns
    -------
    pp_gen : pd.DataFrame
        Production time-series for power plant pp_id
        
    """
    
    pp_gen = generation_data[['ActualGenerationOutput', 'ActualConsumption', 'ResolutionCode', 'InstalledGenCapacity']].loc[generation_data['fresna_id'].isin([pp_id])].copy()


    # hier sollte man vielleicht noch etwas hinzufügen, dass man eine Meldung bekommt, dass es keinen mathc zwischen fresna und entso gab
    if pp_gen.empty:
        raise IndexError()
     
    if ppl.Capacity.loc[pp_id] >= pp_gen.InstalledGenCapacity.max():        
        p_nom = ppl.Capacity.loc[pp_id]
        
    else:
        p_nom = pp_gen.InstalledGenCapacity.max()
        

    pp_gen.ActualGenerationOutput.where(pp_gen.ActualGenerationOutput <= p_nom, p_nom, inplace = True)

    if pp_gen['ResolutionCode'].iloc[0] == 'PT60M':
        t_index = pd.date_range(start='2018-01-01', end='2019-01-01', freq='H', closed='left')
    
        pp_gen = pp_gen.reindex(t_index).copy()
    
        pp_gen.fillna(value=0, inplace = True)
        
        pp_gen = pp_gen.resample(rule='60Min').apply({'ActualGenerationOutput': 'mean',
                                                      'ActualConsumption': 'mean'})


    elif pp_gen['ResolutionCode'].iloc[0] == 'PT30M':
        t_index = pd.date_range(start='2018-01-01', end='2019-01-01', freq='30Min', closed='left')
    
        pp_gen = pp_gen.reindex(t_index).copy()     
    
        pp_gen.fillna(0, inplace = True)
        
        pp_gen = pp_gen.resample(rule='60Min').apply({'ActualGenerationOutput': 'mean',
                                                      'ActualConsumption': 'mean'})
        
    elif pp_gen.ResolutionCode.iloc[0] == 'PT15M':
        t_index = pd.date_range(start='2018-01-01', end='2019-01-01', freq='15Min', closed='left')
    
        pp_gen = pp_gen.reindex(t_index).copy()
    
        pp_gen.fillna(0, inplace = True)
        
        pp_gen = pp_gen.resample(rule='60Min').apply({'ActualGenerationOutput': 'mean',
                                                      'ActualConsumption': 'mean'})
    
    # die beiden Zeilen stimmen wohl nicht. 
    # pp_gen['Generation-Consumption'] = pp_gen['ActualGenerationOutput'] - pp_gen['ActualConsumption']
    
    # pp_gen['Generation-Consumption_pu'] = pp_gen['Generation-Consumption'].div(p_nom)
    
    pp_gen['ActualGenerationOutput_pu'] = pp_gen['ActualGenerationOutput'].div(p_nom)
    
    pp_gen['ActualConsumption_pu'] = pp_gen['ActualConsumption'].div(p_nom)
   
    return pp_gen, p_nom

def import_entsoe_pp_timeseries(path):
    # combining path and .csv data
    
    filenames = sorted(glob.glob(path + "/*.csv"))
    
    # import csv files
    # using pd.concat function as import function to append data to dataframe
    # encoding: "utf-16" see entso-e documentation
    # colum selection is possible by using "usecols=['DateTime','ResolutionCode','AreaCode','AreaTypeCode','GenerationUnitEIC',...]" 
    
    entsoe_pp_timeseries = pd.concat((pd.read_csv(f, sep='\t', encoding='utf-16', index_col = 3) for f in filenames))
    
    entsoe_pp_timeseries.drop(columns=["Year","Month","Day"], inplace=True)
    
    entsoe_pp_timeseries.ProductionTypeName.replace(
                                {'Fossil Hard coal': 'Hard Coal',
                                 'Fossil Coal-derived gas': 'Other',
                                 '.*Hydro.*': 'Hydro',
                                 '.*Oil.*': 'Oil',
                                 '.*Peat': 'Bioenergy',
                                 'Fossil Brown coal/Lignite':'Lignite',
                                 'Biomass': 'Bioenergy',
                                 'Fossil Gas': 'Natural Gas'}, regex = True, inplace = True)
    
    entsoe_pp_timeseries.MapCode.replace({'.*DE.*' : 'DE'}, regex = True, inplace = True)
    
    entsoe_pp_timeseries.index = pd.to_datetime(entsoe_pp_timeseries.index)

    entsoe_pp_timeseries.reset_index(inplace=True)
    
    #set generation and consumtion as absolut value (assuming that the negative entries are incorrect)
    entsoe_pp_timeseries['ActualGenerationOutput'] = entsoe_pp_timeseries.ActualGenerationOutput.abs()
    
    entsoe_pp_timeseries['ActualConsumption'] = entsoe_pp_timeseries.ActualConsumption.abs()
    
    logger.info("ENTSO E Actual GenerationOutput Per Unit is loaded")
    
    return entsoe_pp_timeseries

def import_jrc_links(path, fillnan=False):
    jrc_links = pd.read_csv(path, decimal = '.')
    
    if fillnan == True:
        #achtung, dieser Teil mal mit Ramiz besprechen
        #wenn kein wert bei eic_g vorhanden ist, einfach von eic_p kopieren
        jrc_links.eic_g.fillna(jrc_links.eic_p, inplace = True)
        #wenn kein Wert bei eic_p vorhanden ist, von eic_p kopieren
        jrc_links.eic_p.fillna(jrc_links.eic_g, inplace = True)
        
    
    #ich benötige immer einen Eintrag in eic_g und eic_p (duplikate entfernen)
    #auch hier mal mit Ramiz sprechen
    jrc_links.drop_duplicates(subset='eic_g', keep = 'first' , inplace=True)
    
    jrc_links.drop(jrc_links.columns.difference(['eic_g','eic_p']), axis = 1, inplace=True)
    
    return jrc_links
    

if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_powerplant_profiles')
    configure_logging(snakemake)    
    
    #importing and prepare ENTSO E data
    entsoe_pp_timeseries = import_entsoe_pp_timeseries(snakemake.input.path_entsoe_pp_timeseries)
    
    #importing and prepare linker between eic_g (generation unit) and eic_p (production unit) 
    jrc_links = import_jrc_links(snakemake.input.jrc_units)
    
    #merging links and entso production timeseries
    data_merged = pd.merge(entsoe_pp_timeseries, jrc_links, how='left',left_on='GenerationUnitEIC',right_on='eic_g', indicator=True)

    data_merged.set_index('DateTime', inplace=True)
    
    #drop all entries without an eic_p value (EIC number for production units)
    data_merged.dropna(subset=['eic_p'], inplace=True)
        
    # aggregate to Production Unit (eic_p)
    # reset index is needed for .groupby function 
    data_merged.reset_index(inplace=True)

    data_merged_agg = data_merged.groupby(['eic_p','DateTime'],as_index=False).agg({'ActualGenerationOutput': 'sum',
                                                                                'ActualConsumption': 'sum',
                                                                                'InstalledGenCapacity': 'sum',
                                                                                'ProductionTypeName': 'first',
                                                                                'MapCode': 'first',
                                                                                'PowerSystemResourceName': 'first',
                                                                                'ResolutionCode': 'first'})
    #set index after groupby function
    data_merged_agg.set_index('DateTime', inplace=True)
    
    #loud ppl data 
    ppl = load_powerplants()


    # all eic_p numbers in data_merged_agg
    data_merged_agg_ID = list(data_merged_agg.eic_p.unique())

    #find all mateches with powerplantmaching (fresna_id)

    Entsoe_Fresna_dict = {}
    logger.info("Matching ENTSO E data to powerplantmatching fresna ID")

    with progressbar.ProgressBar(max_value=len(data_merged_agg_ID)) as bar:
        for i in range(len(data_merged_agg_ID)):
            bar.update(i)
            try:
                Entsoe_Fresna_dict[data_merged_agg_ID[i]] = int(pm.utils.select_by_projectID(df=ppl, projectID=data_merged_agg_ID[i]).index[0])
            except IndexError:
                pass

    # map all fresna isd to the data_merged_agg and set NaN if there is no match
    data_merged_agg['fresna_id'] = data_merged_agg['eic_p'].map(Entsoe_Fresna_dict)

    #drop all NaN for fresna_id 
    data_merged_agg.dropna(subset=['fresna_id'], inplace=True)
    
    # aggregate to fresna_id
    # reset index is needed for .groupby function
    data_merged_agg.reset_index(inplace=True)

    data_merged_agg_fresna = data_merged_agg.groupby(['fresna_id','DateTime'],as_index=False).agg({'ActualGenerationOutput': 'sum',
                                                                                'ActualConsumption': 'sum',
                                                                                'InstalledGenCapacity': 'sum',
                                                                                'ProductionTypeName': 'first',
                                                                                'MapCode': 'first',
                                                                                'PowerSystemResourceName': 'first',
                                                                                'ResolutionCode': 'first'})
    
    #set index after groupby function
    data_merged_agg_fresna.set_index('DateTime', inplace=True)
    
    generation_data = data_merged_agg_fresna
    
    
    
    #generating pp profiles out of the data
    
    pp_id = generation_data['fresna_id'].unique().astype(int).tolist()
    
    pp_profiles = pd.DataFrame()
    pp_capacity = dict()
    logger.info("Creating pp profiles")
    difsum = 0
    
    with progressbar.ProgressBar(max_value=len(pp_id)) as bar:     
        for i in range(len(pp_id)):
            bar.update(i)
            try:
                pp_profiles_temp, p_nom_temp = powerplant_generation(pp_id = pp_id[i], generation_data = generation_data, ppl=ppl)
                pp_profiles[pp_id[i]] = pp_profiles_temp['ActualGenerationOutput_pu']
                pp_capacity.update( {str(pp_id[i]) : p_nom_temp} )
                dif = ppl.Capacity.loc[pp_id[i]] - p_nom_temp
                difsum += dif
            except:
                pass
    
    pp_profile_data = xr.DataArray(pp_profiles, dims=["time","pp_id"], attrs=pp_capacity, name='profile')
    
    pp_profile_data.to_netcdf(snakemake.output.profile_pp)
    
    

