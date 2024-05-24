create or replace function public.get_forecast_date(cfs_date timestamp, month_index integer) returns date as
$$
    -- EXAMPLE Code Description
/***************************************************************************************************
Procedure:          dbo.usp_DoSomeStuff
Create Date:        2024-05-24
Author:             Matt McAnear
Description:        Given the current CFS date and the month index, find whether the forecast was for next year or this year. For example, if CFS is run in June, the prediction for January should be in 2025, NOT 2024.
Call by:            None
Affected table(s):  ciglr.lhfx_water
Used By:            Functional Area this is use in, for example, Payroll, Accounting, Finance
Parameter(s):       @cfs_date - The date that the CFS forecasting ran
                    @month_index - The month for which we want to find the date of forecast.
Usage:              select public.get_forecast_date(cfs_runtime, unnest(array [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))
****************************************************************************************************
SUMMARY OF CHANGES
Date(yyyy-mm-dd)    Author              Comments
------------------- ------------------- ------------------------------------------------------------
2024-05-24          Matt McAnear        Initial creation
***************************************************************************************************/
select case
           when date_part('day', date_trunc('month', cfs_date) -
                                 make_date(date_part('year', cfs_date)::integer, month_index, 1)) < 0
               then make_date(date_part('year', cfs_date)::integer, month_index, 1)
           else make_date(date_part('year', cfs_date)::integer + 1, month_index, 1)
           end
$$
    language sql;