import ee
import pandas as pd
import numpy as np
import subprocess
import time
import datetime
import yaml

from config.config import PIPELINE_PARAMS

# Function to convert an ee.FeatureCollection to a pandas dataframe
def GEE_FC_to_pd(fc):
    result = []

    values = fc.toList(500000).getInfo()

    BANDS = fc.first().propertyNames().getInfo()

    if 'system:index' in BANDS: BANDS.remove('system:index')

    for item in values:
        values_item = item['properties']
        row = [values_item[key] for key in BANDS]
        result.append(row)

    df = pd.DataFrame([item for item in result], columns = BANDS)
    df.replace('None', np.nan, inplace = True)

    return df

# Add point coordinates to FC as properties
def addLatLon(f, latString, longString):
    lat = f.geometry().coordinates().get(1)
    lon = f.geometry().coordinates().get(0)
    return f.set(latString, lat).set(longString, lon)

def breakandwait(var_of_interest):
    # !! Break and wait
    count = 1
    while count >= 1:
        taskList = [str(i) for i in ee.batch.Task.list()]
        subsetList = [s for s in taskList if var_of_interest in s]
        subsubList = [s for s in subsetList if any(xs in s for xs in ['RUNNING', 'READY'])]
        count = len(subsubList)
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Number of running jobs:', count)
        time.sleep(PIPELINE_PARAMS['general_params']['wait_time'])
    print('Moving on...')

def upload_table(csv_file):
    # Format the bash call to upload the file to the Google Cloud Storage (GCS) bucket
    gsutilBashUploadList = (
        PIPELINE_PARAMS['cloud_params']['bash_function_gsutil'] +
        PIPELINE_PARAMS['cloud_params']['gcs_cp'] +
        [csv_file] +
        ['gs://' + PIPELINE_PARAMS['cloud_params']['cloud_bucket'] + '/' + csv_file]
    )
    
    # Run the command to upload to GCS
    subprocess.run(gsutilBashUploadList)
    print(f"{csv_file} uploaded to Google Cloud Storage Bucket!")

    # Wait for a short period to ensure the command has been received by the server
    time.sleep(1)

    # Wait for the GSUTIL uploading process to finish before moving on
    while not all(
        x in subprocess.run(
            PIPELINE_PARAMS['cloud_params']['bash_function_gsutil'] + 
            PIPELINE_PARAMS['cloud_params']['gcs_ls'] + 
            ['gs://' + PIPELINE_PARAMS['cloud_params']['cloud_bucket'] + '/' + csv_file],
            stdout=subprocess.PIPE
        ).stdout.decode('utf-8') for x in [csv_file]
    ):
        time.sleep(1)
    print('Everything is uploaded; moving on...')

    earthEngineUploadTableCommands = (
        PIPELINE_PARAMS['cloud_params']['bash_function_earthengine'] +
        PIPELINE_PARAMS['cloud_params']['pre_ee_upload_table'] +
        [PIPELINE_PARAMS['cloud_params']['asset_id_string_prefix'] + 
         PIPELINE_PARAMS['general_params']['gee_project_folder'] + '/' + csv_file] +
        ['gs://' + PIPELINE_PARAMS['cloud_params']['cloud_bucket'] + '/' + csv_file] +
        PIPELINE_PARAMS['cloud_params']['post_ee_upload_table']
    )

    # Run the command to upload to Earth Engine
    subprocess.run(earthEngineUploadTableCommands)
    print('Upload to Earth Engine queued!')

    # Wait for a short period to ensure the command has been received by the server
    breakandwait(PIPELINE_PARAMS['model_params']['response_var'])

# task list
def task_list():
    taskList = [str(i) for i in ee.batch.Task.list()]
    print(taskList)