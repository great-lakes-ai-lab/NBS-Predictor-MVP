create view ciglr.all_forecasts as
select cfsrun
     , date_trunc
     , lake
     , 'basin' as series
     , evap
     , apcp
     , temp
     , lhfx
from ciglr.lake_forecasts_basin
union all
select cfsrun
     , date_trunc
     , lake
     , 'water' as series
     , evap
     , apcp
     , temp
     , lhfx
from ciglr.lake_forecasts_water
union all
select cfsrun
     , date_trunc
     , lake
     , 'land' as series
     , evap
     , apcp
     , temp
     , lhfx
from ciglr.lake_forecasts_land;

