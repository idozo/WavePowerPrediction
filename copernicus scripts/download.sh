
#!/usr/bin/bash
if [ $# -eq 0 ]
  	then
    echo "No arguments supplied"
    exit 1
fi
year=$1
longitude=18.75
latitude=34.64583

max_month=12
echo "DOWNLODING FOR YEAR: " $year
if [ "$year" == "2019" ]
	then
	max_month=5
fi

g++ dumpToCsv.cpp -I/usr/local/include -I/include -L/usr/local/lib -lnetcdf_c++4 -lnetcdf -lm -o dumpToCsv

for ((month=1;month<=max_month;month++)); do
	
	str_month=$month
	if [[ ${#str_month} -lt 2 ]] ; then
    	str_month="0${str_month}"
	fi
	days_in_month=30
	if  ((month==1 || month==3 || month==5 || month==7  || month==8 || month==10 || month== 12)); then
		days_in_month=31
	fi
	if ((month==2)); then 
		days_in_month=28
	fi
	
	for ((day=1;day<=$days_in_month;day++)); do
		str_day=$day
		if [[ ${#str_day} -lt 2 ]] ; then
    		str_day="0${str_day}"
		fi
		
		python -m motuclient --motu http://nrt.cmems-du.eu/motu-web/Motu --service-id MEDSEA_ANALYSIS_FORECAST_WAV_006_017-TDS --product-id med-hcmr-wav-an-fc-h --longitude-min $longitude --longitude-max $longitude --latitude-min $latitude --latitude-max $latitude --date-min $year"-"$str_month"-"$str_day" 00:00:00" --date-max $year"-"$str_month"-"$str_day" 23:00:00" --variable VHM0 --variable VTM10 --variable VTM02 --variable VMDR --variable VSDX --variable VSDY --variable VHM0_WW --variable VTM01_WW --variable VMDR_WW --variable VHM0_SW1 --variable VTM01_SW1 --variable VMDR_SW1 --variable VHM0_SW2 --variable VTM01_SW2 --variable VMDR_SW2 --variable VPED --variable VTPK  --out-dir ./data --out-name $str_day$str_month$year".nc" --user omazor --pwd OrrCMEMS2018
		./dumpToCsv $str_day $str_month $year
		rm -f ./data/$str_day$str_month$year".nc"
	done

done 
