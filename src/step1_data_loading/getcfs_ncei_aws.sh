#/bin/bash

basedir='/mnt/projects/hpc/mroczka/cyclones/cyclonetracking-master/data/CFS/monthly/'
ncodir='/usr/bin/'
awsdir='/home/mroczka/aws/v2/2.8.0/bin/'

##############################

#[W m-2]   ---->   [converting to]   ---->   [mm day-1]

 #W          kg           J         m^3       1000 mm     86400 sec     mm
#---  X  ----------  X  -----  X  -------  X  -------  X  ---------  =  ---
#m^2     2.5x10^6 J     W sec     1000 kg       1 m          day        day

######################

# [kg m-2]  ---->  [converting to]  ---->  [mm]

#   kg       m^3        1000 mm
#   --   X  -----    X  -------  =  mm
#   m^2     1000 kg       1 m

########

#https://www.ncei.noaa.gov/data/climate-forecast-system/access/

#https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/

#https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/2023/202312/20231202/2023120200/pgbf.01.2023120200.202312.avrg.grib.grb2

#pgbf.01.2023120200.202401.avrg.grib.grb2
#...
#...
#...
#pgbf.01.2023120200.202409.avrg.grib.grb2

#pgbf.01.2023120206.202312.avrg.grib.grb2
#pgbf.01.2023120206.202401.avrg.grib.grb2
#...
#...
#...
#pgbf.01.2023120206.202409.avrg.grib.grb2

#flx also

#let DIFF=(`date +%s -d 20240410`-`date +%s -d 20240320`)/86400
#echo $DIFF

#exit

rm -f *.grb2
rm -f *.nc
rm -f *.nc2
rm -f *.tmp
rm -f *allmonths*.nc

now="$(date +'%Y%m%d')"

UTC=`date -u +%H` #double digit UTC hour 09
#inpast=10
#inpast=$1
#source=$2 #source options ncei or aws

targetdate=$1
source=$2 #source options ncei or aws

let inpast=(`date +%s -d ${now}`-`date +%s -d ${targetdate}`)/86400

echo $inpast

tsoffset=10

#if [ $source == 'aws' ]; then

# /home/mroczka/aws/v2/current/bin/aws s3 cp --no-sign-request s3://noaa-cfs-pds/cfs.20240405/00/monthly_grib_01/pgbf.01.2024040500.202406.avrg.grib.grb2 .

# exit

#fi


if [ $UTC -le 23 ]; then

 YYYYMMDD=`date -u +%Y%m%d --date="${inpast} days ago"`
 YYYYMMDD10=`date -u +%Y%m%d --date="${tsoffset} days ago"`
 YYYYMMDDP1=`date -u +%Y%m%d --date="${inpast} year"`
 YYYYMMDD2=`date -u +%Y%m%d --date="${inpast} days ago"`
 YYYY=`date -u +%Y --date="${inpast} days ago"`
 YYYYMM=`date -u +%Y%m --date="${inpast} days ago"`
 YYYYMMP0=`date -u +%Y%m --date="${inpast} days ago"`
 yesterday=`date -u +"%Y%m%d" --date "${inpast} days ago"`
 echo $yesterday
 YYYYMMP1=`date -d "${yesterday}+1 month" +"%Y%m"`
 echo $YYYYMMP1
# echo $(date -d "${yesterday}+1 month - $(date +%d) days " | awk '{ printf "%s - %s days", $2, $3 }')
 YYYYMMP2=`date -d "${yesterday}+2 month" +"%Y%m"`
 echo $YYYYMMP2
 YYYYMMP3=`date -d "${yesterday}+3 month" +"%Y%m"`
 echo $YYYYMMP3
 YYYYMMP4=`date -d "${yesterday}+4 month" +"%Y%m"`
 echo $YYYYMMP4
 YYYYMMP5=`date -d "${yesterday}+5 month" +"%Y%m"`
 echo $YYYYMMP5
 YYYYMMP6=`date -d "${yesterday}+6 month" +"%Y%m"`
 echo $YYYYMMP6
 YYYYMMP7=`date -d "${yesterday}+7 month" +"%Y%m"`
 echo $YYYYMMP7
 YYYYMMP8=`date -d "${yesterday}+8 month" +"%Y%m"`
 echo $YYYYMMP8
 YYYYMMP9=`date -d "${yesterday}+9 month" +"%Y%m"`
 echo $YYYYMMP9

 DD=`date -u +%d --date="${inpast} days ago"`
 MM=`date -u +%m --date="${inpast} days ago"`
fi

#/mnt/projects/hpc/mroczka/cyclones/cyclonetracking-master/data/CFS/monthly/${YYYYMMDD};

if test -d /mnt/projects/hpc/mroczka/cyclones/cyclonetracking-master/data/CFS/monthly/${YYYYMMDD}; then
  echo "Directory exists."
else
  echo "Directory does not exist...will create it..."
  mkdir $YYYYMMDD
fi

MM1=`date -u +%m --date="0 days ago"`
MM2=`date -u +%m --date="${inpast} days ago"`
echo $MM $MM1 $MM2

if [ $MM1 == $MM2 ];
then
 echo 'same month'
 samemonth='yes'
 if [ $MM == "01" ]; then
   daysinmonth=31
 elif [ $MM == "02" ]; then
   daysinmonth=28
 elif [ $MM == "03" ]; then
   daysinmonth=31
 elif [ $MM == "04" ]; then
   daysinmonth=30
 elif [ $MM == "05" ]; then
   daysinmonth=31
 elif [ $MM == "06" ]; then
   daysinmonth=30
 elif [ $MM == "07" ]; then
   daysinmonth=31
 elif [ $MM == "08" ]; then
   daysinmonth=31
 elif [ $MM == "09" ]; then
   daysinmonth=30 
 elif [ $MM == "10" ]; then
   daysinmonth=31
 elif [ $MM == "11" ]; then
   daysinmonth=30
 elif [ $MM == "12" ]; then
   daysinmonth=31
 fi
 if [ ${DD} == '08' ]; then
  DD=8
 fi
 if [ ${DD} == '09' ]; then
  DD=9
 fi
 daysleft=$[daysinmonth - ${DD} + 1]
 echo 'days left = ' $daysleft
 secondsleft=$[daysleft*86400]
 DDminus=$[${DD}-1]
 secondssince=$[${DDminus}*86400]
 echo 'seconds left: '$secondsleft
 echo 'seconds since: '$secondssince

 yesterday=`date +"%Y%m%d" --date "${inpast} days ago"`
 echo $yesterday


# dayofrun=`date +"%m %Y" --date "${inpast} days ago"`
# echo $dayofrun


 yestplusdays=`date -d "${yesterday}+${daysleft} days" +"%m %Y"`
 echo $yestplusdays
 echo 'new stuff'
# numdaysplus1=`cal ${yesterday}`
# echo 'numdaysplus1 = ' $numdaysplus1
 numdaysplus=`cal ${yesterday} | awk 'NF {DAYS = $NF}; END {print DAYS}'`
 echo $numdaysplus
 numdaysplus=`cal ${yestplusdays} | awk 'NF {DAYS = $NF}; END {print DAYS}'`
 echo 'num days plus : '$numdaysplus

 numdaysminus=`cal ${yesterday} | awk 'NF {DAYS = $NF}; END {print DAYS}'`
 echo $numdaysminus
 numdaysminus=`cal ${yestplusdays} | awk 'NF {DAYS = $NF}; END {print DAYS}'`
 echo $numdaysminus
 echo 'begin'

 numdays1=`cal ${yestplusdays} | awk 'NF {DAYS = $NF}; END {print DAYS}'`
# echo $numdays1
 numdays1sec=$[${numdays1}*86400 + $secondsleft]

# what=`date -d "${yesterday}+${daysleft} days + 1 month" +"%m %Y")`
# echo 'what = ' $what

# numdays21=`cal $(date -d "${yesterday}+${daysleft} days + 1 month" +"%m %Y")`
# echo 'num21 = ' $numdays21

 nextmonth=`date -d "${yesterday}+${daysleft} days" +"%Y%m01"`
 numdays2=`cal $(date -d "${nextmonth} + 1 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
 echo $numdays2

 numdays2sec=$[${numdays2}*86400 + $numdays1sec]
 numdays3=`cal $(date -d "${nextmonth} + 2 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
# numdays3=`cal $(date -d "${yesterday}+${daysleft} days + 2 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
 numdays3sec=$[${numdays3}*86400 + $numdays2sec]
 numdays4=`cal $(date -d "${nextmonth} + 3 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
# numdays4=`cal $(date -d "${yesterday}+${daysleft} days + 3 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
 numdays4sec=$[${numdays4}*86400 + $numdays3sec]
 numdays5=`cal $(date -d "${nextmonth} + 4 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
# numdays5=`cal $(date -d "${yesterday}+${daysleft} days + 4 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
 numdays5sec=$[${numdays5}*86400 + $numdays4sec]
 numdays6=`cal $(date -d "${nextmonth} + 5 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
# numdays6=`cal $(date -d "${yesterday}+${daysleft} days + 5 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
 numdays6sec=$[${numdays6}*86400 + $numdays5sec]
 numdays7=`cal $(date -d "${nextmonth} + 6 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
# numdays7=`cal $(date -d "${yesterday}+${daysleft} days + 6 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
 numdays7sec=$[${numdays7}*86400 + $numdays6sec]
 numdays8=`cal $(date -d "${nextmonth} + 7 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
# numdays8=`cal $(date -d "${yesterday}+${daysleft} days + 7 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
 numdays8sec=$[${numdays8}*86400 + $numdays7sec]
 numdays9=`cal $(date -d "${nextmonth} + 8 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
# numdays9=`cal $(date -d "${yesterday}+${daysleft} days + 8 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
 numdays9sec=$[${numdays9}*86400 + $numdays8sec]
 echo 'numdays1sec = ' $numdays1sec

else
 samemonth='no'
 if [ $MM == "01" ]; then
   daysinmonth=31
 elif [ $MM == "02" ]; then
   daysinmonth=28
 elif [ $MM == "03" ]; then
   daysinmonth=31
 elif [ $MM == "04" ]; then
   daysinmonth=30
 elif [ $MM == "05" ]; then
   daysinmonth=31
 elif [ $MM == "06" ]; then
   daysinmonth=30
 elif [ $MM == "07" ]; then
   daysinmonth=31
 elif [ $MM == "08" ]; then
   daysinmonth=31
 elif [ $MM == "09" ]; then
   daysinmonth=30 
 elif [ $MM == "10" ]; then
   daysinmonth=31
 elif [ $MM == "11" ]; then
   daysinmonth=30
 elif [ $MM == "12" ]; then
   daysinmonth=31
 fi
 if [ ${DD} == '08' ]; then
  DD=8
 fi
 if [ ${DD} == '09' ]; then
  DD=9
 fi
 daysleft=$[daysinmonth - ${DD}+1]
 echo 'days left: '${daysleft}
 secondsleft=$[daysleft*86400]
 echo $secondsleft
 DDminus=$[${DD}-1]
 secondssince=$[${DDminus}*86400]
 echo 'seconds left: '$secondsleft
 echo 'seconds since: '$secondssince
 
 yesterday=`date +"%Y%m%d" --date "${inpast} days ago"`
 echo $yesterday
 yestplusdays=`date -d "${yesterday}+${daysleft} days" +"%m %Y"`
 echo $yestplusdays
 echo 'new stuff'
 numdaysplus=`cal ${yesterday} | awk 'NF {DAYS = $NF}; END {print DAYS}'`
 echo $numdaysplus
 numdaysplus=`cal ${yestplusdays} | awk 'NF {DAYS = $NF}; END {print DAYS}'`
 echo 'num days plus : '$numdaysplus

 numdaysminus=`cal ${yesterday} | awk 'NF {DAYS = $NF}; END {print DAYS}'`
 echo $numdaysminus
 numdaysminus=`cal ${yestplusdays} | awk 'NF {DAYS = $NF}; END {print DAYS}'`
 echo $numdaysminus
 echo 'begin'

 numdays1=`cal ${yestplusdays} | awk 'NF {DAYS = $NF}; END {print DAYS}'`
 echo 'numdays1 = ' $numdays1
 numdays1sec=$[${numdays1}*86400 + $secondsleft]

 nextmonth=`date -d "${yesterday}+${daysleft} days" +"%Y%m01"`
 echo 'nextmonth = ' $nextmonth
 numdays2=`cal $(date -d "${nextmonth} + 1 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
# echo $numdays2

 numdays2sec=$[${numdays2}*86400 + $numdays1sec]
 numdays3=`cal $(date -d "${nextmonth} + 2 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
# numdays3=`cal $(date -d "${yesterday}+${daysleft} days + 2 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
 numdays3sec=$[${numdays3}*86400 + $numdays2sec]
 numdays4=`cal $(date -d "${nextmonth} + 3 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
# numdays4=`cal $(date -d "${yesterday}+${daysleft} days + 3 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
 numdays4sec=$[${numdays4}*86400 + $numdays3sec]
 numdays5=`cal $(date -d "${nextmonth} + 4 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
# numdays5=`cal $(date -d "${yesterday}+${daysleft} days + 4 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
 numdays5sec=$[${numdays5}*86400 + $numdays4sec]
 numdays6=`cal $(date -d "${nextmonth} + 5 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
# numdays6=`cal $(date -d "${yesterday}+${daysleft} days + 5 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
 numdays6sec=$[${numdays6}*86400 + $numdays5sec]
 numdays7=`cal $(date -d "${nextmonth} + 6 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
# numdays7=`cal $(date -d "${yesterday}+${daysleft} days + 6 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
 numdays7sec=$[${numdays7}*86400 + $numdays6sec]
 numdays8=`cal $(date -d "${nextmonth} + 7 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
# numdays8=`cal $(date -d "${yesterday}+${daysleft} days + 7 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
 numdays8sec=$[${numdays8}*86400 + $numdays7sec]
 numdays9=`cal $(date -d "${nextmonth} + 8 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
# numdays9=`cal $(date -d "${yesterday}+${daysleft} days + 8 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
 numdays9sec=$[${numdays9}*86400 + $numdays8sec]

# numdays3=`cal $(date -d "${yesterday}+${daysleft} days + 2 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
# numdays3sec=$[${numdays3}*86400 + $numdays2sec]
# numdays4=`cal $(date -d "${yesterday}+${daysleft} days + 3 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
# numdays4sec=$[${numdays4}*86400 + $numdays3sec]
# numdays5=`cal $(date -d "${yesterday}+${daysleft} days + 4 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
# numdays5sec=$[${numdays5}*86400 + $numdays4sec]
# numdays6=`cal $(date -d "${yesterday}+${daysleft} days + 5 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
# numdays6sec=$[${numdays6}*86400 + $numdays5sec]
# numdays7=`cal $(date -d "${yesterday}+${daysleft} days + 6 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
# numdays7sec=$[${numdays7}*86400 + $numdays6sec]
# numdays8=`cal $(date -d "${yesterday}+${daysleft} days + 7 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
# numdays8sec=$[${numdays8}*86400 + $numdays7sec]
# numdays9=`cal $(date -d "${yesterday}+${daysleft} days + 8 month" +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'`
# numdays9sec=$[${numdays9}*86400 + $numdays8sec]

 echo 'secondssince = ' $secondssince
 echo 'secondsleft = ' $secondsleft
 echo 'numdays1sec= ' $numdays1sec
 echo 'numdays2= ' $numdays2
 echo 'numdays2sec = ' $numdays2sec
 echo 'numdays3= ' $numdays3
 echo 'numdays3sec = ' $numdays3sec
 echo 'numdays4= ' $numdays4
 echo 'numdays4sec = ' $numdays4sec
 echo 'numdays5= ' $numdays5
 echo 'numdays5sec = ' $numdays5sec
 echo 'numdays6= ' $numdays6
 echo 'numdays6sec = ' $numdays6sec
 echo 'numdays7= ' $numdays7
 echo 'numdays7sec = ' $numdays7sec
 echo 'numdays8= ' $numdays8
 echo 'numdays8sec = ' $numdays8sec
 echo 'numdays9= ' $numdays9
 echo 'numdays9sec = ' $numdays9sec

fi

#exit

fhr=0
chka=0

products=( 'pgb' 'flx' )
utctimes=( 00 06 12 18 )
#utctimes=( 18 20 )

for u in "${utctimes[@]}"
do
 for pr in "${products[@]}"
 do

###month 0

#/usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2

#./aws s3 cp --no-sign-request s3://noaa-cfs-pds/cfs.20240325/00/monthly_grib_01/pgbf.01.2024032500.202412.avrg.grib.grb2 .
  if [ $MM1 == $MM2 ]; then 
    echo 'same month again'
    if [ $source == 'ncei' ]; then 
     wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP0}.avrg.grib.grb2
    elif [ $source == 'aws' ]; then
     ${awsdir}aws s3 cp --no-sign-request s3://noaa-cfs-pds/cfs.${YYYYMMDD}/${u}/monthly_grib_01/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP0}.avrg.grib.grb2 ${basedir}
    else
     echo 'source not recognized...'
    fi
    if [ $pr == 'pgb' ]; then 
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -g2clib 0 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP0}.avrg.grib.grb2 | egrep ':(APCP:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP0}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP0}.cnbs.grb2
    else
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP0}.avrg.grib.grb2 | egrep ':(TMP:2 m above ground|SHTFL:surface|LHTFL:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP0}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP0}.cnbs2.grb2

    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP0}.cnbs2.grb2 -new_grid latlon 0:360:1 90:181:-1 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP0}.cnbs.grb2

    fi

    rm -f ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP0}.avrg.grib.grb2
    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP0}.cnbs.grb2 -netcdf ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP0}.cnbs.nc
    echo 'at ncap2'
    echo $secondssince
    ${ncodir}ncap2 -O -s "time=(time-${secondssince})" ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP0}.cnbs.nc ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP0}.cnbs.nc2
    echo 'past ncap2'

###month 1

#    wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP1}.avrg.grib.grb2


    if [ $source == 'ncei' ]; then 
     wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP1}.avrg.grib.grb2
    elif [ $source == 'aws' ]; then
     ${awsdir}aws s3 cp --no-sign-request s3://noaa-cfs-pds/cfs.${YYYYMMDD}/${u}/monthly_grib_01/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP1}.avrg.grib.grb2 ${basedir}
    else
     echo 'source not recognized...'
    fi


    if [ $pr == 'pgb' ]; then 
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -g2clib 0 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP1}.avrg.grib.grb2 | egrep ':(APCP:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP1}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP1}.cnbs.grb2
    else
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP1}.avrg.grib.grb2 | egrep ':(TMP:2 m above ground|SHTFL:surface|LHTFL:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP1}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP1}.cnbs2.grb2

    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP1}.cnbs2.grb2 -new_grid latlon 0:360:1 90:181:-1 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP1}.cnbs.grb2

    fi

    rm -f ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP1}.avrg.grib.grb2
    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP1}.cnbs.grb2 -netcdf ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP1}.cnbs.nc

    ${ncodir}ncap2 -O -s "time=(time+${secondsleft})" ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP1}.cnbs.nc ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP1}.cnbs.nc2

###month 2

#    wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP2}.avrg.grib.grb2


    if [ $source == 'ncei' ]; then 
     wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP2}.avrg.grib.grb2
    elif [ $source == 'aws' ]; then
     ${awsdir}aws s3 cp --no-sign-request s3://noaa-cfs-pds/cfs.${YYYYMMDD}/${u}/monthly_grib_01/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP2}.avrg.grib.grb2 ${basedir}
    else
     echo 'source not recognized...'
    fi


    if [ $pr == 'pgb' ]; then 
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -g2clib 0 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP2}.avrg.grib.grb2 | egrep ':(APCP:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP2}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP2}.cnbs.grb2
    else
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP2}.avrg.grib.grb2 | egrep ':(TMP:2 m above ground|SHTFL:surface|LHTFL:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP2}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP2}.cnbs2.grb2

    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP2}.cnbs2.grb2 -new_grid latlon 0:360:1 90:181:-1 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP2}.cnbs.grb2

    fi

    rm -f ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP2}.avrg.grib.grb2
    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP2}.cnbs.grb2 -netcdf ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP2}.cnbs.nc
    echo 'ncap2 step'
    ${ncodir}ncap2 -O -s "time=(time+${numdays1sec})" ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP2}.cnbs.nc ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP2}.cnbs.nc2  
    echo 'after ncpa2 step'

###month 3

#    wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP3}.avrg.grib.grb2


    if [ $source == 'ncei' ]; then 
     wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP3}.avrg.grib.grb2
    elif [ $source == 'aws' ]; then
     ${awsdir}aws s3 cp --no-sign-request s3://noaa-cfs-pds/cfs.${YYYYMMDD}/${u}/monthly_grib_01/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP3}.avrg.grib.grb2 ${basedir}
    else
     echo 'source not recognized...'
    fi


    if [ $pr == 'pgb' ]; then 
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -g2clib 0 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP3}.avrg.grib.grb2 | egrep ':(APCP:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP3}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP3}.cnbs.grb2
    else
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP3}.avrg.grib.grb2 | egrep ':(TMP:2 m above ground|SHTFL:surface|LHTFL:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP3}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP3}.cnbs2.grb2

    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP3}.cnbs2.grb2 -new_grid latlon 0:360:1 90:181:-1 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP3}.cnbs.grb2

    fi

    rm -f ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP3}.avrg.grib.grb2
    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP3}.cnbs.grb2 -netcdf ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP3}.cnbs.nc

    ${ncodir}ncap2 -O -s "time=(time+${numdays2sec})" ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP3}.cnbs.nc ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP3}.cnbs.nc2

###month 4

 #   wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP4}.avrg.grib.grb2


    if [ $source == 'ncei' ]; then 
     wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP4}.avrg.grib.grb2
    elif [ $source == 'aws' ]; then
     ${awsdir}aws s3 cp --no-sign-request s3://noaa-cfs-pds/cfs.${YYYYMMDD}/${u}/monthly_grib_01/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP4}.avrg.grib.grb2 ${basedir}
    else
     echo 'source not recognized...'
    fi

    if [ $pr == 'pgb' ]; then 
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -g2clib 0 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP4}.avrg.grib.grb2 | egrep ':(APCP:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP4}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP4}.cnbs.grb2
    else
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP4}.avrg.grib.grb2 | egrep ':(TMP:2 m above ground|SHTFL:surface|LHTFL:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP4}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP4}.cnbs2.grb2

    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP4}.cnbs2.grb2 -new_grid latlon 0:360:1 90:181:-1 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP4}.cnbs.grb2

    fi

    rm -f ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP4}.avrg.grib.grb2
    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP4}.cnbs.grb2 -netcdf ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP4}.cnbs.nc

    ${ncodir}ncap2 -O -s "time=(time+${numdays3sec})" ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP4}.cnbs.nc ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP4}.cnbs.nc2

###month 5

#    wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP5}.avrg.grib.grb2  


    if [ $source == 'ncei' ]; then 
     wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP5}.avrg.grib.grb2
    elif [ $source == 'aws' ]; then
     ${awsdir}aws s3 cp --no-sign-request s3://noaa-cfs-pds/cfs.${YYYYMMDD}/${u}/monthly_grib_01/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP5}.avrg.grib.grb2 ${basedir}
    else
     echo 'source not recognized...'
    fi

    if [ $pr == 'pgb' ]; then 
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -g2clib 0 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP5}.avrg.grib.grb2 | egrep ':(APCP:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP5}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP5}.cnbs.grb2
    else
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP5}.avrg.grib.grb2 | egrep ':(TMP:2 m above ground|SHTFL:surface|LHTFL:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP5}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP5}.cnbs2.grb2

    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP5}.cnbs2.grb2 -new_grid latlon 0:360:1 90:181:-1 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP5}.cnbs.grb2

    fi

    rm -f ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP5}.avrg.grib.grb2
    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP5}.cnbs.grb2 -netcdf ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP5}.cnbs.nc

    ${ncodir}ncap2 -O -s "time=(time+${numdays4sec})" ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP5}.cnbs.nc ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP5}.cnbs.nc2

###month 6

#    wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP6}.avrg.grib.grb2

    if [ $source == 'ncei' ]; then 
     wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP6}.avrg.grib.grb2
    elif [ $source == 'aws' ]; then
     ${awsdir}aws s3 cp --no-sign-request s3://noaa-cfs-pds/cfs.${YYYYMMDD}/${u}/monthly_grib_01/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP6}.avrg.grib.grb2 ${basedir}
    else
     echo 'source not recognized...'
    fi


    if [ $pr == 'pgb' ]; then 
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -g2clib 0 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP6}.avrg.grib.grb2 | egrep ':(APCP:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP6}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP6}.cnbs.grb2
    else
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP6}.avrg.grib.grb2 | egrep ':(TMP:2 m above ground|SHTFL:surface|LHTFL:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP6}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP6}.cnbs2.grb2

    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP6}.cnbs2.grb2 -new_grid latlon 0:360:1 90:181:-1 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP6}.cnbs.grb2

    fi

    rm -f ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP6}.avrg.grib.grb2
    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP6}.cnbs.grb2 -netcdf ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP6}.cnbs.nc

    ${ncodir}ncap2 -O -s "time=(time+${numdays5sec})" ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP6}.cnbs.nc ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP6}.cnbs.nc2

###month 7

#    wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP7}.avrg.grib.grb2

    if [ $source == 'ncei' ]; then 
     wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP7}.avrg.grib.grb2
    elif [ $source == 'aws' ]; then
     ${awsdir}aws s3 cp --no-sign-request s3://noaa-cfs-pds/cfs.${YYYYMMDD}/${u}/monthly_grib_01/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP7}.avrg.grib.grb2 ${basedir}
    else
     echo 'source not recognized...'
    fi
 

    if [ $pr == 'pgb' ]; then 
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -g2clib 0 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP7}.avrg.grib.grb2 | egrep ':(APCP:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP7}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP7}.cnbs.grb2
    else
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP7}.avrg.grib.grb2 | egrep ':(TMP:2 m above ground|SHTFL:surface|LHTFL:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP7}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP7}.cnbs2.grb2

    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP7}.cnbs2.grb2 -new_grid latlon 0:360:1 90:181:-1 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP7}.cnbs.grb2

    fi

    rm -f ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP7}.avrg.grib.grb2
    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP7}.cnbs.grb2 -netcdf ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP7}.cnbs.nc

    ${ncodir}ncap2 -O -s "time=(time+${numdays6sec})" ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP7}.cnbs.nc ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP7}.cnbs.nc2

###month 8

#    wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP8}.avrg.grib.grb2

    if [ $source == 'ncei' ]; then 
     wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP8}.avrg.grib.grb2
    elif [ $source == 'aws' ]; then
     ${awsdir}aws s3 cp --no-sign-request s3://noaa-cfs-pds/cfs.${YYYYMMDD}/${u}/monthly_grib_01/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP8}.avrg.grib.grb2 ${basedir}
    else
     echo 'source not recognized...'
    fi

    if [ $pr == 'pgb' ]; then 
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -g2clib 0 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP8}.avrg.grib.grb2 | egrep ':(APCP:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP8}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP8}.cnbs.grb2
    else
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP8}.avrg.grib.grb2 | egrep ':(TMP:2 m above ground|SHTFL:surface|LHTFL:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP8}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP8}.cnbs2.grb2

    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP8}.cnbs2.grb2 -new_grid latlon 0:360:1 90:181:-1 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP8}.cnbs.grb2

    fi

    rm -f ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP8}.avrg.grib.grb2
    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP8}.cnbs.grb2 -netcdf ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP8}.cnbs.nc

    ${ncodir}ncap2 -O -s "time=(time+${numdays7sec})" ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP8}.cnbs.nc ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP8}.cnbs.nc2

###month 9

#    wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP9}.avrg.grib.grb2

    if [ $source == 'ncei' ]; then 
     wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP9}.avrg.grib.grb2
    elif [ $source == 'aws' ]; then
     ${awsdir}aws s3 cp --no-sign-request s3://noaa-cfs-pds/cfs.${YYYYMMDD}/${u}/monthly_grib_01/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP9}.avrg.grib.grb2 ${basedir}
    else
     echo 'source not recognized...'
    fi


    if [ $pr == 'pgb' ]; then 
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -g2clib 0 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP9}.avrg.grib.grb2 | egrep ':(APCP:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP9}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP9}.cnbs.grb2
    else
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP9}.avrg.grib.grb2 | egrep ':(TMP:2 m above ground|SHTFL:surface|LHTFL:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP9}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP9}.cnbs2.grb2

    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP9}.cnbs2.grb2 -new_grid latlon 0:360:1 90:181:-1 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP9}.cnbs.grb2

    fi

    rm -f ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP9}.avrg.grib.grb2
    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP9}.cnbs.grb2 -netcdf ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP9}.cnbs.nc

    ${ncodir}ncap2 -O -s "time=(time+${numdays8sec})" ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP9}.cnbs.nc ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP9}.cnbs.nc2

  else

#    wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP0}.avrg.grib.grb2

    if [ $source == 'ncei' ]; then 
     wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP0}.avrg.grib.grb2
    elif [ $source == 'aws' ]; then
     ${awsdir}aws s3 cp --no-sign-request s3://noaa-cfs-pds/cfs.${YYYYMMDD}/${u}/monthly_grib_01/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP0}.avrg.grib.grb2 ${basedir}
    else
     echo 'source not recognized...'
    fi

    if [ $pr == 'pgb' ]; then 
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -g2clib 0 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP0}.avrg.grib.grb2 | egrep ':(APCP:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP0}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP0}.cnbs.grb2
    else
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP0}.avrg.grib.grb2 | egrep ':(TMP:2 m above ground|SHTFL:surface|LHTFL:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP0}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP0}.cnbs2.grb2

    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP0}.cnbs2.grb2 -new_grid latlon 0:360:1 90:181:-1 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP0}.cnbs.grb2

    fi

    rm -f ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP0}.avrg.grib.grb2
    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP0}.cnbs.grb2 -netcdf ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP0}.cnbs.nc

    ${ncodir}ncap2 -O -s "time=(time-${secondssince})" ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP0}.cnbs.nc ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP0}.cnbs.nc2

###month 1

#    wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP1}.avrg.grib.grb2

    if [ $source == 'ncei' ]; then 
     wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP1}.avrg.grib.grb2
    elif [ $source == 'aws' ]; then
     ${awsdir}aws s3 cp --no-sign-request s3://noaa-cfs-pds/cfs.${YYYYMMDD}/${u}/monthly_grib_01/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP1}.avrg.grib.grb2 ${basedir}
    else
     echo 'source not recognized...'
    fi

    if [ $pr == 'pgb' ]; then 
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -g2clib 0 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP1}.avrg.grib.grb2 | egrep ':(APCP:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP1}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP1}.cnbs.grb2
    else
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP1}.avrg.grib.grb2 | egrep ':(TMP:2 m above ground|SHTFL:surface|LHTFL:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP1}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP1}.cnbs2.grb2

    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP1}.cnbs2.grb2 -new_grid latlon 0:360:1 90:181:-1 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP1}.cnbs.grb2

    fi

    rm -f ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP1}.avrg.grib.grb2
    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP1}.cnbs.grb2 -netcdf ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP1}.cnbs.nc

    ${ncodir}ncap2 -O -s "time=(time+${secondsleft})" ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP1}.cnbs.nc ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP1}.cnbs.nc2

###month 2

#    wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP2}.avrg.grib.grb2

    if [ $source == 'ncei' ]; then 
     wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP2}.avrg.grib.grb2
    elif [ $source == 'aws' ]; then
     ${awsdir}aws s3 cp --no-sign-request s3://noaa-cfs-pds/cfs.${YYYYMMDD}/${u}/monthly_grib_01/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP2}.avrg.grib.grb2 ${basedir}
    else
     echo 'source not recognized...'
    fi

    if [ $pr == 'pgb' ]; then 
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -g2clib 0 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP2}.avrg.grib.grb2 | egrep ':(APCP:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP2}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP2}.cnbs.grb2
    else
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP2}.avrg.grib.grb2 | egrep ':(TMP:2 m above ground|SHTFL:surface|LHTFL:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP2}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP2}.cnbs2.grb2

    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP2}.cnbs2.grb2 -new_grid latlon 0:360:1 90:181:-1 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP2}.cnbs.grb2

    fi

    rm -f ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP2}.avrg.grib.grb2
    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP2}.cnbs.grb2 -netcdf ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP2}.cnbs.nc
    echo 'ncap2 step'
    ${ncodir}ncap2 -O -s "time=(time+${numdays1sec})" ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP2}.cnbs.nc ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP2}.cnbs.nc2  
    echo 'after ncpa2 step'

###month 3

#    wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP3}.avrg.grib.grb2

    if [ $source == 'ncei' ]; then 
     wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP3}.avrg.grib.grb2
    elif [ $source == 'aws' ]; then
     ${awsdir}aws s3 cp --no-sign-request s3://noaa-cfs-pds/cfs.${YYYYMMDD}/${u}/monthly_grib_01/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP3}.avrg.grib.grb2 ${basedir}
    else
     echo 'source not recognized...'
    fi

    if [ $pr == 'pgb' ]; then 
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -g2clib 0 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP3}.avrg.grib.grb2 | egrep ':(APCP:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP3}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP3}.cnbs.grb2
    else
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP3}.avrg.grib.grb2 | egrep ':(TMP:2 m above ground|SHTFL:surface|LHTFL:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP3}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP3}.cnbs2.grb2

    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP3}.cnbs2.grb2 -new_grid latlon 0:360:1 90:181:-1 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP3}.cnbs.grb2

    fi

    rm -f ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP3}.avrg.grib.grb2
    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP3}.cnbs.grb2 -netcdf ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP3}.cnbs.nc

    ${ncodir}ncap2 -O -s "time=(time+${numdays2sec})" ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP3}.cnbs.nc ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP3}.cnbs.nc2

###month 4

#    wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP4}.avrg.grib.grb2

    if [ $source == 'ncei' ]; then 
     wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP4}.avrg.grib.grb2
    elif [ $source == 'aws' ]; then
     ${awsdir}aws s3 cp --no-sign-request s3://noaa-cfs-pds/cfs.${YYYYMMDD}/${u}/monthly_grib_01/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP4}.avrg.grib.grb2 ${basedir}
    else
     echo 'source not recognized...'
    fi

    if [ $pr == 'pgb' ]; then 
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -g2clib 0 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP4}.avrg.grib.grb2 | egrep ':(APCP:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP4}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP4}.cnbs.grb2
    else
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP4}.avrg.grib.grb2 | egrep ':(TMP:2 m above ground|SHTFL:surface|LHTFL:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP4}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP4}.cnbs2.grb2

    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP4}.cnbs2.grb2 -new_grid latlon 0:360:1 90:181:-1 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP4}.cnbs.grb2

    fi

    rm -f ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP4}.avrg.grib.grb2
    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP4}.cnbs.grb2 -netcdf ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP4}.cnbs.nc

    ${ncodir}ncap2 -O -s "time=(time+${numdays3sec})" ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP4}.cnbs.nc ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP4}.cnbs.nc2

###month 5

#    wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP5}.avrg.grib.grb2

    if [ $source == 'ncei' ]; then 
     wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP5}.avrg.grib.grb2
    elif [ $source == 'aws' ]; then
     ${awsdir}aws s3 cp --no-sign-request s3://noaa-cfs-pds/cfs.${YYYYMMDD}/${u}/monthly_grib_01/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP5}.avrg.grib.grb2 ${basedir}
    else
     echo 'source not recognized...'
    fi

    if [ $pr == 'pgb' ]; then 
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -g2clib 0 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP5}.avrg.grib.grb2 | egrep ':(APCP:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP5}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP5}.cnbs.grb2
    else
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP5}.avrg.grib.grb2 | egrep ':(TMP:2 m above ground|SHTFL:surface|LHTFL:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP5}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP5}.cnbs2.grb2

    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP5}.cnbs2.grb2 -new_grid latlon 0:360:1 90:181:-1 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP5}.cnbs.grb2

    fi

    rm -f ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP5}.avrg.grib.grb2
    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP5}.cnbs.grb2 -netcdf ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP5}.cnbs.nc

    ${ncodir}ncap2 -O -s "time=(time+${numdays4sec})" ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP5}.cnbs.nc ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP5}.cnbs.nc2

###month 6

#    wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP6}.avrg.grib.grb2

    if [ $source == 'ncei' ]; then 
     wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP6}.avrg.grib.grb2
    elif [ $source == 'aws' ]; then
     ${awsdir}aws s3 cp --no-sign-request s3://noaa-cfs-pds/cfs.${YYYYMMDD}/${u}/monthly_grib_01/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP6}.avrg.grib.grb2 ${basedir}
    else
     echo 'source not recognized...'
    fi

    if [ $pr == 'pgb' ]; then 
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -g2clib 0 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP6}.avrg.grib.grb2 | egrep ':(APCP:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP6}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP6}.cnbs.grb2
    else
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP6}.avrg.grib.grb2 | egrep ':(TMP:2 m above ground|SHTFL:surface|LHTFL:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP6}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP6}.cnbs2.grb2

    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP6}.cnbs2.grb2 -new_grid latlon 0:360:1 90:181:-1 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP6}.cnbs.grb2

    fi

    rm -f ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP6}.avrg.grib.grb2
    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP6}.cnbs.grb2 -netcdf ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP6}.cnbs.nc

    ${ncodir}ncap2 -O -s "time=(time+${numdays5sec})" ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP6}.cnbs.nc ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP6}.cnbs.nc2

###month 7

#    wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP7}.avrg.grib.grb2

    if [ $source == 'ncei' ]; then 
     wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP7}.avrg.grib.grb2
    elif [ $source == 'aws' ]; then
     ${awsdir}aws s3 cp --no-sign-request s3://noaa-cfs-pds/cfs.${YYYYMMDD}/${u}/monthly_grib_01/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP7}.avrg.grib.grb2 ${basedir}
    else
     echo 'source not recognized...'
    fi

    if [ $pr == 'pgb' ]; then 
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -g2clib 0 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP7}.avrg.grib.grb2 | egrep ':(APCP:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP7}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP7}.cnbs.grb2
    else
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP7}.avrg.grib.grb2 | egrep ':(TMP:2 m above ground|SHTFL:surface|LHTFL:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP7}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP7}.cnbs2.grb2

    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP7}.cnbs2.grb2 -new_grid latlon 0:360:1 90:181:-1 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP7}.cnbs.grb2

    fi

    rm -f ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP7}.avrg.grib.grb2
    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP7}.cnbs.grb2 -netcdf ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP7}.cnbs.nc

    ${ncodir}ncap2 -O -s "time=(time+${numdays6sec})" ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP7}.cnbs.nc ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP7}.cnbs.nc2

###month 8

#    wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP8}.avrg.grib.grb2

    if [ $source == 'ncei' ]; then 
     wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP8}.avrg.grib.grb2
    elif [ $source == 'aws' ]; then
     ${awsdir}aws s3 cp --no-sign-request s3://noaa-cfs-pds/cfs.${YYYYMMDD}/${u}/monthly_grib_01/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP8}.avrg.grib.grb2 ${basedir}
    else
     echo 'source not recognized...'
    fi

    if [ $pr == 'pgb' ]; then 
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -g2clib 0 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP8}.avrg.grib.grb2 | egrep ':(APCP:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP8}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP8}.cnbs.grb2
    else
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP8}.avrg.grib.grb2 | egrep ':(TMP:2 m above ground|SHTFL:surface|LHTFL:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP8}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP8}.cnbs2.grb2

    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP8}.cnbs2.grb2 -new_grid latlon 0:360:1 90:181:-1 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP8}.cnbs.grb2

    fi

    rm -f ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP8}.avrg.grib.grb2
    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP8}.cnbs.grb2 -netcdf ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP8}.cnbs.nc

    ${ncodir}ncap2 -O -s "time=(time+${numdays7sec})" ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP8}.cnbs.nc ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP8}.cnbs.nc2

###month 9

#    wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP9}.avrg.grib.grb2

    if [ $source == 'ncei' ]; then 
     wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/${YYYY}/${YYYYMM}/${YYYYMMDD}/${YYYYMMDD}${u}/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP9}.avrg.grib.grb2
    elif [ $source == 'aws' ]; then
     ${awsdir}aws s3 cp --no-sign-request s3://noaa-cfs-pds/cfs.${YYYYMMDD}/${u}/monthly_grib_01/${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP9}.avrg.grib.grb2 ${basedir}
    else
     echo 'source not recognized...'
    fi

    if [ $pr == 'pgb' ]; then 
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -g2clib 0 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP9}.avrg.grib.grb2 | egrep ':(APCP:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP9}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP9}.cnbs.grb2
    else
     /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP9}.avrg.grib.grb2 | egrep ':(TMP:2 m above ground|SHTFL:surface|LHTFL:surface)' | /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 -i ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP9}.avrg.grib.grb2 -grib ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP9}.cnbs2.grb2

    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP9}.cnbs2.grb2 -new_grid latlon 0:360:1 90:181:-1 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP9}.cnbs.grb2

    fi

    rm -f ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP9}.avrg.grib.grb2
    /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP9}.cnbs.grb2 -netcdf ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP9}.cnbs.nc

    ${ncodir}ncap2 -O -s "time=(time+${numdays8sec})" ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP9}.cnbs.nc ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP9}.cnbs.nc2

  fi
###
 echo "catting"

 /usr/bin/ncrcat -O ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP0}.cnbs.nc2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP1}.cnbs.nc2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP2}.cnbs.nc2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP3}.cnbs.nc2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP4}.cnbs.nc2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP5}.cnbs.nc2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP6}.cnbs.nc2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP7}.cnbs.nc2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP8}.cnbs.nc2 ${pr}f.01.${YYYYMMDD}${u}.${YYYYMMP9}.cnbs.nc2 ${pr}f.01.${YYYYMMDD}${u}.allmonths.cnbs.nc
 
 done

# Now download time-series mslp for the idea of cyclone unit per basin for incorporating Jamie's work....discussed at data driven model meeting # on 4/16/24

 if test -d /mnt/projects/hpc/mroczka/cyclones/cyclonetracking-master/data/CFS/monthly/tracks/premonthly/${YYYYMMDD10}${u}; then
  echo "Directory exists."
 else
  echo "Directory does not exist...will create it..."
  mkdir ${basedir}tracks/premonthly/${YYYYMMDD}${u}
 fi

#https://noaa-cfs-pds.s3.amazonaws.com/index.html#cfs.20240416/00/time_grib_01/

 if [ $source == 'aws' ]; then 
  ${awsdir}aws s3 cp --no-sign-request s3://noaa-cfs-pds/cfs.${YYYYMMDD10}/${u}/time_grib_01/prmsl.01.${YYYYMMDD10}${u}.daily.grb2 ${basedir}tracks/premonthly/${YYYYMMDD10}${u}/prmsl.01.${YYYYMMDD10}${u}.daily.grb2
 elif [ $source == 'ncei' ]; then
  wget https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/time-series/${YYYY}/${YYYYMMP0}/${YYYYMMDD10}/${YYYYMMDD10}${u}/prmsl.01.${YYYYMMDD10}${u}.daily.grb2 -O ${basedir}tracks/premonthly/${YYYYMMDD10}${u}/prmsl.01.${YYYYMMDD10}${u}.daily.grb2
 else
  echo 'source not recognized...'
 fi

 /usr/local/Modules/apps/wgrib2/3.1.2/bin/wgrib2 ${basedir}tracks/premonthly/${YYYYMMDD10}${u}/prmsl.01.${YYYYMMDD10}${u}.daily.grb2 -netcdf /mnt/projects/hpc/mroczka/cyclones/cyclonetracking-master/data/CFS/monthly/tracks/premonthly/${YYYYMMDD10}${u}/prmsl.01.${YYYYMMDD10}${u}.daily.nc

done

mv *allmonths*.nc ${targetdate}/

exit

