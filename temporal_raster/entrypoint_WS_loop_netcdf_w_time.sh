  # INPUT_PATH='./temporal_raster/data/climate_data_Weather_source_era5_precip_201401010000_201501010000.nc'
  INPUT_PATH='./temporal_raster/data/climate_data_Weather_source_era5_precip_201501010000_201601010000.nc'
  BQ_UPLOAD=0
  SF_UPLOAD=1
	# cdo remapbil,global_0.25 $INPUT_PATH "${INPUT_PATH/.nc/}_regridded.nc"
  #   INPUT_PATH="${INPUT_PATH/.nc/}_regridded.nc"

    timestamp=1
  for idate in $(cdo showtimestamp $INPUT_PATH) 
    do
      echo 'Processing idate: ' $idate
      ymd="${idate:0:10}"  # Extract year, month, and day
      d="${idate:8:2}"     # Extract day
      hour="${idate:11:2}" # Extract hour
      gdal_translate -ot Float64 NETCDF:$INPUT_PATH:precipitation_in -b $timestamp "${INPUT_PATH/.nc/}_${d}_${hour}.tif"
      ((timestamp++))

      TIF_PATH="${INPUT_PATH/.nc/}_${d}_${hour}.tif"
      filename=$(basename "$TIF_PATH")
      echo $filename
      gdalwarp -s_srs EPSG:4326 -t_srs EPSG:3857 $TIF_PATH "${TIF_PATH/.tif/_webmercator.tif}"
      rm $TIF_PATH
  
      WEBMERCATOR_PATH="${TIF_PATH/.tif/_webmercator.tif}"
      OUTPUT_PATH="${WEBMERCATOR_PATH/_webmercator.tif/_quadbin.tif}"
      gdalwarp "$WEBMERCATOR_PATH" \
      -of COG  \
      -co TILING_SCHEME=GoogleMapsCompatible \
      -co COMPRESS=DEFLATE -co OVERVIEWS=NONE -co ADD_ALPHA=NO -co RESAMPLING=NEAREST "$OUTPUT_PATH"
      rm $WEBMERCATOR_PATH

      # TABLE="${filename/.tif/_quadbin}"


      # Get the number of bands in the GeoTIFF file
      N_BANDS=$(gdalinfo "$OUTPUT_PATH" | grep "Band " | wc -l)

if [ $BQ_UPLOAD -eq 1 ]; then
  # GCP_PROJECT="cartodb-data-engineering-team" GCP_DATASET="vdelacruz_carto" GCP_TABLE="climate_data_weather_source_era5_precip_201401010000_201601010000" . ./temporal_raster/entrypoint_WS_loop_netcdf_w_time.sh

      COMMAND="echo \"yes\" | carto bigquery upload"
      for ((band=1; band<=$N_BANDS; band++)); do
          COMMAND+=" --band $band"
      done
      COMMAND+=" --file_path \"$OUTPUT_PATH\""
      COMMAND+=" --project \"$GCP_PROJECT\""
      COMMAND+=" --dataset \"$GCP_DATASET\""
      COMMAND+=" --table \"$GCP_TABLE\""
      COMMAND+=" --append"

      eval "$COMMAND"
fi

if [ $SF_UPLOAD -eq 1 ]; then

  # SF_DATABASE="CARTO_DATA_ENGINEERING_TEAM" SF_SCHEMA="vdelacruz_carto" SF_TABLE="climate_data_weather_source_era5_precip_201401010000_201601010000" SF_ACCOUNT="sxa81489.us-east-1" SF_USERNAME="SUPERUSER_DATA_ENG_TEAM" SF_PASSWORD="XXXXX" . ./temporal_raster/entrypoint_WS_loop_netcdf_w_time.sh

      COMMAND="echo \"yes\" | carto snowflake upload"
      # for ((band=1; band<=$N_BANDS; band++)); do
      #     COMMAND+=" --band $band"
      # done
      COMMAND+=" --band 1 --band_name precipitation"
      COMMAND+=" --file_path \"$OUTPUT_PATH\""
      COMMAND+=" --database \"$SF_DATABASE\""
      COMMAND+=" --schema \"$SF_SCHEMA\""
      COMMAND+=" --table \"$SF_TABLE\""
      COMMAND+=" --account \"$SF_ACCOUNT\""
      COMMAND+=" --username \"$SF_USERNAME\""
      COMMAND+=" --password \"$SF_PASSWORD\""
      COMMAND+=" --append"
      COMMAND+=" --timestamp $idate"

      eval "$COMMAND"
fi
      rm $OUTPUT_PATH
      # if timestamp > 10 break
# if [ $timestamp -gt 0 ]; then
#   break # Remove this line to process all the files
# fi
done
