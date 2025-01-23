import pandas as pd
import ee
import yaml
import subprocess

from functions.cv_functions import computeCVAccuracyAndRMSE
from functions.helper_functions import GEE_FC_to_pd

from config.config import PIPELINE_PARAMS

# Author: Johan van den Hoogen

# Initialize ee API
if PIPELINE_PARAMS['cloud_params']['use_service_account'] == True:
    credentials = ee.ServiceAccountCredentials(PIPELINE_PARAMS['cloud_params']['service_account'], PIPELINE_PARAMS['cloud_params']['service_account_key'])
    ee.Initialize(credentials, project = PIPELINE_PARAMS['cloud_params']['cloud_project'])
else:
    ee.Initialize()

##################################################################################################################################################################
# Hyperparameter tuning
##################################################################################################################################################################
fcOI = ee.FeatureCollection(PIPELINE_PARAMS['general_params']['gee_project_folder']+PIPELINE_PARAMS['general_params']['training_data_filename'])
def gee_hyperparameter_tuning():
    '''
    Perform hyperparameter tuning for random forest regression model
    '''

    # Define hyperparameters for grid search
    varsPerSplit_list = list(range(2, PIPELINE_PARAMS['model_params']['var_per_split'], 2))
    leafPop_list = list(range(2, PIPELINE_PARAMS['model_params']['min_leaf_pop'], 2))

    classifierList = []
    # Create list of classifiers for regression
    for vps in varsPerSplit_list:
        for lp in leafPop_list:

            model_name = PIPELINE_PARAMS['general_params']['response_var'] + '_rf_VPS' + str(vps) + '_LP' + str(lp) + '_REGRESSION'

            rf = ee.Feature(ee.Geometry.Point([0,0])).set('cName',model_name,'c',ee.Classifier.smileRandomForest(
            numberOfTrees = PIPELINE_PARAMS['model_params']['n_trees'],
            variablesPerSplit = vps,
            minLeafPopulation = lp,
            bagFraction = 0.632,
            seed = PIPELINE_PARAMS['model_params']['seed']
            ).setOutputMode('REGRESSION'))

            classifierList.append(rf)

    # # If grid search was not performed yet:
    # Make a feature collection from the k-fold assignment list
    # kFoldAssignmentFC = ee.FeatureCollection(ee.List(list(range(1,PIPELINE_PARAMS['cv_params']['k']+1))).map(lambda n: ee.Feature(ee.Geometry.Point([0,0])).set('Fold',n)))

    finished_models = list()

    # Check if any models have been completed
    try:
        grid_search_results = ee.FeatureCollection(PIPELINE_PARAMS['general_params']['gee_project_folder']+'/'+PIPELINE_PARAMS['general_params']['response_var']+'_regression_grid_search_results')
        print(grid_search_results.size().getInfo())

    except Exception as e:
        try:
            # Create list of finished models
            finished_models = subprocess.run(PIPELINE_PARAMS['cloud_params']['gcs_ls']+['projects/'+PIPELINE_PARAMS['cloud_params']['project_id']+'/assets'+'/'+PIPELINE_PARAMS['general_params']['gee_project_folder']+'/hyperparameter_tuning/'], stdout=subprocess.PIPE).stdout.splitlines()
            finished_models = [model.decode('ascii').split('/')[-1] for model in finished_models]
            
        except Exception as e:
            classDfRegression = pd.DataFrame(columns = ['Mean_R2', 'StDev_R2','Mean_RMSE', 'StDev_RMSE','Mean_MAE', 'StDev_MAE', 'cName'])

        # Perform model testing for remaining hyperparameter settings
        for rf in classifierList:
            if rf.get('cName').getInfo() in finished_models:
                print('Model', classifierList.index(rf), 'out of total of', len(classifierList), 'already finished')
            else:
                print('Testing model', classifierList.index(rf), 'out of total of', len(classifierList))
                fcOI = ee.FeatureCollection(PIPELINE_PARAMS['general_params']['gee_project_folder']+PIPELINE_PARAMS['general_params']['training_data_filename'])
                accuracy_feature = ee.Feature(computeCVAccuracyAndRMSE(rf))
                accuracy_featureExport = ee.batch.Export.table.toAsset(
                    collection = ee.FeatureCollection([accuracy_feature]),
                    description = PIPELINE_PARAMS['today']+'_'+PIPELINE_PARAMS['model_params']['resonse_var']+'_hyperparameter_tuning_'+rf.get('cName').getInfo(),
                    assetId = PIPELINE_PARAMS['general_params']['gee_project_folder']+'/'+PIPELINE_PARAMS['today']+'_'+PIPELINE_PARAMS['model_params']['resonse_var']+'_hyperparameter_tuning_'+rf.get('cName').getInfo())
                accuracy_featureExport.start()

    # Fetch FC from GEE
    grid_search_results = ee.FeatureCollection([ee.FeatureCollection(PIPELINE_PARAMS['general_params']['gee_project_folder']+'/'+PIPELINE_PARAMS['today']+'_'+PIPELINE_PARAMS['model_params']['resonse_var']+'_hyperparameter_tuning_'+rf.get('cName').getInfo()) for rf in classifierList]).flatten()
    grid_search_results_df = GEE_FC_to_pd(grid_search_results)

    grid_search_results_export = ee.batch.Export.table.toAsset(
        collection = grid_search_results,
        description = PIPELINE_PARAMS['today']+'_'+PIPELINE_PARAMS['model_params']['resonse_var']+'_grid_search_results',
        assetId = PIPELINE_PARAMS['general_params']['gee_project_folder']+'/'+PIPELINE_PARAMS['today']+'_'+PIPELINE_PARAMS['model_params']['resonse_var']+'_grid_search_results'
        )
    grid_search_results_export.start()

    # Sort values
    grid_search_results_df.sort_values(PIPELINE_PARAMS['cv_params']['sort_acc_prop'], ascending = False)

    # Write model results to csv
    grid_search_results_df.to_csv('output/'+PIPELINE_PARAMS['today']+'_'+PIPELINE_PARAMS['model_params']['resonse_var']+'_grid_search_results.csv', index=False)

    # Get top model name
    bestModelName = grid_search_results.limit(1, PIPELINE_PARAMS['cv_params']['sort_acc_prop'], False).first().get('cName')

    # Get top 10 models
    top_10Models = grid_search_results.limit(10, PIPELINE_PARAMS['cv_params']['sort_acc_prop'], False).aggregate_array('cName')

    return bestModelName, top_10Models, classifierList