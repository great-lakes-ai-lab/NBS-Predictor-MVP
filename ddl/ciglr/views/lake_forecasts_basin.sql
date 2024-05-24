create view ciglr.lake_forecasts_basin as
select t1.cfsrun
     , date_trunc('day', t1.cfs_runtime)::date
     , 'superior'        as lake
     , t1.basin_superior as evap
     , t2.basin_superior as apcp
     , t3.basin_superior as temp
     , t4.basin_superior as lhfx
from raw_data.evap_forecasts      t1
left join raw_data.apcp_forecasts t2 on t1.cfsrun = t2.cfsrun and t1.forecast_date = t2.forecast_date
left join raw_data.tmp_forecasts  t3 on t1.cfsrun = t3.cfsrun and t1.forecast_date = t3.forecast_date
left join raw_data.lhfx_forecasts t4 on t1.cfsrun = t4.cfsrun and t1.forecast_date = t4.forecast_date
union all
select t1.cfsrun
     , date_trunc('day', t1.cfs_runtime)::date
     , 'erie'        as lake
     , t1.basin_erie as evap
     , t2.basin_erie as apcp
     , t3.basin_erie as temp
     , t4.basin_erie as lhfx
from raw_data.evap_forecasts      t1
left join raw_data.apcp_forecasts t2 on t1.cfsrun = t2.cfsrun and t1.forecast_date = t2.forecast_date
left join raw_data.tmp_forecasts  t3 on t1.cfsrun = t3.cfsrun and t1.forecast_date = t3.forecast_date
left join raw_data.lhfx_forecasts t4 on t1.cfsrun = t4.cfsrun and t1.forecast_date = t4.forecast_date
where t1.cfs_runtime < '2024-04-01'
union all
select t1.cfsrun
     , date_trunc('day', t1.cfs_runtime)::date
     , 'huron'        as lake
     , t1.basin_huron as evap
     , t2.basin_huron as apcp
     , t3.basin_huron as temp
     , t4.basin_huron as lhfx
from raw_data.evap_forecasts      t1
left join raw_data.apcp_forecasts t2 on t1.cfsrun = t2.cfsrun and t1.forecast_date = t2.forecast_date
left join raw_data.tmp_forecasts  t3 on t1.cfsrun = t3.cfsrun and t1.forecast_date = t3.forecast_date
left join raw_data.lhfx_forecasts t4 on t1.cfsrun = t4.cfsrun and t1.forecast_date = t4.forecast_date
union all
select t1.cfsrun
     , date_trunc('day', t1.cfs_runtime)::date
     , 'michigan'        as lake
     , t1.basin_michigan as evap
     , t2.basin_michigan as apcp
     , t3.basin_michigan as temp
     , t4.basin_michigan as lhfx
from raw_data.evap_forecasts      t1
left join raw_data.apcp_forecasts t2 on t1.cfsrun = t2.cfsrun and t1.forecast_date = t2.forecast_date
left join raw_data.tmp_forecasts  t3 on t1.cfsrun = t3.cfsrun and t1.forecast_date = t3.forecast_date
left join raw_data.lhfx_forecasts t4 on t1.cfsrun = t4.cfsrun and t1.forecast_date = t4.forecast_date
union all
select t1.cfsrun
     , date_trunc('day', t1.cfs_runtime)::date
     , 'ontario'        as lake
     , t1.basin_ontario as evap
     , t2.basin_ontario as apcp
     , t3.basin_ontario as temp
     , t4.basin_ontario as lhfx
from raw_data.evap_forecasts      t1
left join raw_data.apcp_forecasts t2 on t1.cfsrun = t2.cfsrun and t1.forecast_date = t2.forecast_date
left join raw_data.tmp_forecasts  t3 on t1.cfsrun = t3.cfsrun and t1.forecast_date = t3.forecast_date
left join raw_data.lhfx_forecasts t4 on t1.cfsrun = t4.cfsrun and t1.forecast_date = t4.forecast_date;

