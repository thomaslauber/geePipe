# Import the modules of interest
import pandas as pd
import numpy as np
import subprocess
import time
import datetime
import ee
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from itertools import combinations
from itertools import repeat
from functions.determineBlockSizeForCV import *

service_account_file = '/Users/johanvandenhoogen/ETH/Projects/google_cloud/crowther-gee-serviceaccount/gem-eth-analysis-96ea9ecb2158.json'

service_account = 'crowther-gee@gem-eth-analysis.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, service_account_file)
ee.Initialize(credentials)
# ee.Initialize()

today = datetime.date.today().strftime("%Y%m%d")

guild = 'ectomycorrhizal'

####################################################################################################################################################################
# Configuration
####################################################################################################################################################################
# Input the name of the username that serves as the home folder for asset storage
usernameFolderString = 'johanvandenhoogen'

# Input the Cloud Storage Bucket that will hold the bootstrap collections when uploading them to Earth Engine
# !! This bucket should be pre-created before running this script
bucketOfInterest = 'johanvandenhoogen'

# Input the name of the classification property
classProperty = guild + '_richness'

# Input the name of the project folder inside which all of the assets will be stored
# This folder will be generated automatically below, if it isn't yet present
projectFolder = '000_SPUN/2024_darkdiv_v3/' + classProperty

# Input the normal wait time (in seconds) for "wait and break" cells
normalWaitTime = 5

# Input a longer wait time (in seconds) for "wait and break" cells
longWaitTime = 10

# Specify the column names where the latitude and longitude information is stored
latString = 'Pixel_Lat'
longString = 'Pixel_Long'

# Log transform classProperty? Boolean, either True or False
log_transform_classProperty = True

# Ensemble of top 10 models?
ensemble = True

# Spatial leave-one-out cross-validation settings
# skip test points outside training space after removing points in buffer zone? This might reduce extrapolation but overestimate accuracy
loo_cv_wPointRemoval = False

# Define buffer size in meters; use Moran's I or other test to determine SAC range
# Alternatively: specify buffer size as list, to test across multiple buffer sizes
buffer_size = 100000

####################################################################################################################################################################
# Covariate data settings
####################################################################################################################################################################

# List of the covariates to use
covariateList = [
'CGIAR_PET',
'CHELSA_BIO_Annual_Mean_Temperature',
'CHELSA_BIO_Annual_Precipitation',
'CHELSA_BIO_Max_Temperature_of_Warmest_Month',
'CHELSA_BIO_Precipitation_Seasonality',
'ConsensusLandCover_Human_Development_Percentage',
# 'ConsensusLandCoverClass_Barren',
# 'ConsensusLandCoverClass_Deciduous_Broadleaf_Trees',
# 'ConsensusLandCoverClass_Evergreen_Broadleaf_Trees',
# 'ConsensusLandCoverClass_Evergreen_Deciduous_Needleleaf_Trees',
# 'ConsensusLandCoverClass_Herbaceous_Vegetation',
# 'ConsensusLandCoverClass_Mixed_Other_Trees',
# 'ConsensusLandCoverClass_Shrubs',
'EarthEnvTexture_CoOfVar_EVI',
'EarthEnvTexture_Correlation_EVI',
'EarthEnvTexture_Homogeneity_EVI',
'EarthEnvTopoMed_AspectCosine',
'EarthEnvTopoMed_AspectSine',
'EarthEnvTopoMed_Elevation',
'EarthEnvTopoMed_Slope',
'EarthEnvTopoMed_TopoPositionIndex',
'EsaCci_BurntAreasProbability',
'GHS_Population_Density',
'GlobBiomass_AboveGroundBiomass',
# 'GlobPermafrost_PermafrostExtent',
'MODIS_NPP',
# 'PelletierEtAl_SoilAndSedimentaryDepositThicknesses',
'SG_Depth_to_bedrock',
'SG_Sand_Content_005cm',
'SG_SOC_Content_005cm',
'SG_Soil_pH_H2O_005cm',
]

compositeOfInterest = ee.Image.cat([
		ee.Image("projects/crowtherlab/Composite/CrowtherLab_bioComposite_30ArcSec"),
		ee.Image("projects/crowtherlab/Composite/CrowtherLab_climateComposite_30ArcSec"),
		ee.Image("projects/crowtherlab/Composite/CrowtherLab_geoComposite_30ArcSec"),
		ee.Image("projects/crowtherlab/Composite/CrowtherLab_processComposite_30ArcSec"),
		])

project_vars = [
'sequencing_platform454Roche',
'sequencing_platformIllumina',
'sequencing_platformIonTorrent',
'sequencing_platformPacBio',
'sample_typerhizosphere_soil',
'sample_typesoil',
'sample_typetopsoil',
'primers5_8S_Fun_ITS4_Fun',
'primersfITS7_ITS4',
'primersfITS9_ITS4',
'primersgITS7_ITS4',
'primersgITS7_ITS4_then_ITS9_ITS4',
'primersgITS7_ITS4_ITS4arch',
'primersgITS7_ITS4m',
'primersgITS7_ITS4ngs',
'primersgITS7ngs_ITS4ngsUni',
'primersITS_S2F___ITS3_mixed_1_1_ITS4',
'primersITS1_ITS4',
'primersITS1F_ITS4',
'primersITS1F_ITS4_then_fITS7_ITS4',
'primersITS1F_ITS4_then_ITS3_ITS4',
'primersITS1ngs_ITS4ngs_or_ITS1Fngs_ITS4ngs',
'primersITS3_KYO2_ITS4',
'primersITS3_ITS4',
'primersITS3ngs1_to_5___ITS3ngs10_ITS4ngs',
'primersITS3ngs1_to_ITS3ngs11_ITS4ngs',
'primersITS86F_ITS4',
'primersITS9MUNngs_ITS4ngsUni',
]

covariateList = covariateList + project_vars

####################################################################################################################################################################
# Cross validation settings
####################################################################################################################################################################
# Set k for k-fold CV
k = 10

# Make a list of the k-fold CV assignments to use
kList = list(range(1,k+1))

# Set number of trees in RF models
nTrees = 250

# Specify whether to use spatial or random CV
spatialCV = True 

# Input the name of the property that holds the CV fold assignment
cvFoldHeader = 'CV_Fold'

cvFoldString_Spatial = cvFoldHeader + '_Spatial'
cvFoldString_Random = cvFoldHeader + '_Random'

# Metric to use for sorting k-fold CV hyperparameter tuning (default: R2)
sort_acc_prop = 'Mean_R2' # (either one of 'Mean_R2', 'Mean_MAE', 'Mean_RMSE')

if spatialCV == True:
    sort_acc_prop = sort_acc_prop + '_Spatial'
else:
    sort_acc_prop = sort_acc_prop + '_Random'

# Input the title of the CSV that will hold all of the data that has been given a CV fold assignment
titleOfCSVWithCVAssignments = today+'_'+classProperty+"_training_data"

# Asset ID of uploaded dataset after processing
assetIDForCVAssignedColl = 'projects/johanvandenhoogen/assets/'+projectFolder+'/'+titleOfCSVWithCVAssignments

# Write the name of a local staging area folder for outputted CSV's
holdingFolder = '/Users/johanvandenhoogen/SPUN/darkdiv/data/'
outputFolder = '/Users/johanvandenhoogen/SPUN/darkdiv/output'

# Create directory to hold training data
Path(holdingFolder).mkdir(parents=True, exist_ok=True)

####################################################################################################################################################################
# Export settings
####################################################################################################################################################################

# Set pyramidingPolicy for exporting purposes
pyramidingPolicy = 'mean'

# Load a geometry to use for the export
exportingGeometry = ee.Geometry.Polygon([[[-180, 88], [180, 88], [180, -88], [-180, -88]]], None, False)

####################################################################################################################################################################
# Bootstrap settings
####################################################################################################################################################################

# Number of bootstrap iterations
bootstrapIterations = 100

# Generate the seeds for bootstrapping
seedsToUseForBootstrapping = list(range(1, bootstrapIterations+1))

# Input the header text that will name the bootstrapped dataset
bootstrapSamples = classProperty+'_bootstrapSamples'

# Write the name of the variable used for stratification
stratificationVariableString = "Resolve_Biome"

# Input the dictionary of values for each of the stratification category levels
# !! This area breakdown determines the proportion of each biome to include in every bootstrap
strataDict = {
    1: 14.900835665820974,
    2: 2.941697660221864,
    3: 0.526059731441294,
    4: 9.56387696566245,
    5: 2.865354077500338,
    6: 11.519674266872787,
    7: 16.26999434439293,
    8: 8.047078485979089,
    9: 0.861212221078014,
    10: 3.623974712557433,
    11: 6.063922959332467,
    12: 2.5132866428302836,
    13: 20.037841544639985,
    14: 0.26519072167008,
}

####################################################################################################################################################################
# Bash and Google Cloud Bucket settings
####################################################################################################################################################################
# Specify the necessary arguments to upload the files to a Cloud Storage bucket
# I.e., create bash variables in order to create/check/delete Earth Engine Assets

# Specify main bash functions being used
bashFunction_EarthEngine = '/Users/johanvandenhoogen/miniconda3/envs/ee/bin/earthengine'
bashFunctionGSUtil = '/Users/johanvandenhoogen/google-cloud-sdk/bin/gsutil'

# Specify the arguments to these functions
arglist_preEEUploadTable = ['upload','table']
arglist_postEEUploadTable = ['--x_column', longString, '--y_column', latString]
arglist_preGSUtilUploadFile = ['cp']
formattedBucketOI = 'gs://'+bucketOfInterest
assetIDStringPrefix = '--asset_id='
arglist_CreateCollection = ['create','collection']
arglist_CreateFolder = ['create','folder']
arglist_Detect = ['asset','info']
arglist_Delete = ['rm','-r']
arglist_ls = ['ls']
stringsOfInterest = ['Asset does not exist or is not accessible']

# Compose the arguments into lists that can be run via the subprocess module
bashCommandList_Detect = [bashFunction_EarthEngine]+arglist_Detect
bashCommandList_Delete = [bashFunction_EarthEngine]+arglist_Delete
bashCommandList_ls = [bashFunction_EarthEngine]+arglist_ls
bashCommandList_CreateCollection = [bashFunction_EarthEngine]+arglist_CreateCollection
bashCommandList_CreateFolder = [bashFunction_EarthEngine]+arglist_CreateFolder

####################################################################################################################################################################
# Helper functions
####################################################################################################################################################################
# Function to convert GEE FC to pd.DataFrame. Not ideal as it's calling .getInfo(), but does the job
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
def addLatLon(f):
    lat = f.geometry().coordinates().get(1)
    lon = f.geometry().coordinates().get(0)
    return f.set(latString, lat).set(longString, lon)

# R^2 function
def coefficientOfDetermination(fcOI,propertyOfInterest,propertyOfInterest_Predicted):
    # Compute the mean of the property of interest
    propertyOfInterestMean = ee.Number(ee.Dictionary(ee.FeatureCollection(fcOI).select([propertyOfInterest]).reduceColumns(ee.Reducer.mean(),[propertyOfInterest])).get('mean'))

    # Compute the total sum of squares
    def totalSoSFunction(f):
        return f.set('Difference_Squared',ee.Number(ee.Feature(f).get(propertyOfInterest)).subtract(propertyOfInterestMean).pow(ee.Number(2)))
    totalSumOfSquares = ee.Number(ee.Dictionary(ee.FeatureCollection(fcOI).map(totalSoSFunction).select(['Difference_Squared']).reduceColumns(ee.Reducer.sum(),['Difference_Squared'])).get('sum'))

    # Compute the residual sum of squares
    def residualSoSFunction(f):
        return f.set('Residual_Squared',ee.Number(ee.Feature(f).get(propertyOfInterest)).subtract(ee.Number(ee.Feature(f).get(propertyOfInterest_Predicted))).pow(ee.Number(2)))
    residualSumOfSquares = ee.Number(ee.Dictionary(ee.FeatureCollection(fcOI).map(residualSoSFunction).select(['Residual_Squared']).reduceColumns(ee.Reducer.sum(),['Residual_Squared'])).get('sum'))

    # Finalize the calculation
    r2 = ee.Number(1).subtract(residualSumOfSquares.divide(totalSumOfSquares))

    return ee.Number(r2)

# RMSE function
def RMSE(fcOI,propertyOfInterest,propertyOfInterest_Predicted):
    # Compute the squared difference between observed and predicted
    def propDiff(f):
        diff = ee.Number(f.get(propertyOfInterest)).subtract(ee.Number(f.get(propertyOfInterest_Predicted)))

        return f.set('diff', diff.pow(2))

    # calculate RMSE from squared difference
    rmse = ee.Number(fcOI.map(propDiff).reduceColumns(ee.Reducer.mean(), ['diff']).get('mean')).sqrt()

    return rmse

# MAE function
def MAE(fcOI,propertyOfInterest,propertyOfInterest_Predicted):
    # Compute the absolute difference between observed and predicted
    def propDiff(f):
        diff = ee.Number(f.get(propertyOfInterest)).subtract(ee.Number(f.get(propertyOfInterest_Predicted)))

        return f.set('diff', diff.abs())

    # calculate MAE from squared difference
    MAE = ee.Number(fcOI.map(propDiff).reduceColumns(ee.Reducer.mean(), ['diff']).get('mean'))

    return MAE

# Function to take a feature with a classifier of interest
def computeCVAccuracyAndRMSE(featureWithClassifier):
    # Pull the classifier from the feature
    cOI = ee.Classifier(featureWithClassifier.get('c'))

    # Get the model type
    modelType = cOI.mode().getInfo()

    # Create a function to map through the fold assignments and compute the overall accuracy
    # for all validation folds
    def computeAccuracyForFold(foldFeature):
        # Organize the training and validation data

        foldNumber = ee.Number(ee.Feature(foldFeature).get('Fold'))
        trainingData_Random = fcOI.filterMetadata(cvFoldString_Random,'not_equals',foldNumber)
        validationData_Random = fcOI.filterMetadata(cvFoldString_Random,'equals',foldNumber)

        trainingData_Spatial = fcOI.filterMetadata(cvFoldString_Spatial,'not_equals',foldNumber)
        validationData_Spatial = fcOI.filterMetadata(cvFoldString_Spatial,'equals',foldNumber)

        # Train the classifier and classify the validation dataset
        trainedClassifier_Random = cOI.train(trainingData_Random,classProperty,covariateList)
        outputtedPropName_Random = classProperty+'_Predicted_Random'
        classifiedValidationData_Random = validationData_Random.classify(trainedClassifier_Random,outputtedPropName_Random)

        trainedClassifier_Spatial = cOI.train(trainingData_Spatial,classProperty,covariateList)
        outputtedPropName_Spatial = classProperty+'_Predicted_Spatial'
        classifiedValidationData_Spatial = validationData_Spatial.classify(trainedClassifier_Spatial,outputtedPropName_Spatial)

        if modelType == 'CLASSIFICATION':
            # Compute the overall accuracy of the classification
            errorMatrix_Random = classifiedValidationData_Random.errorMatrix(classProperty,outputtedPropName_Random,categoricalLevels)
            overallAccuracy_Random = ee.Number(errorMatrix_Random.accuracy())

            errorMatrix_Spatial = classifiedValidationData_Spatial.errorMatrix(classProperty,outputtedPropName_Spatial,categoricalLevels)
            overallAccuracy_Spatial = ee.Number(errorMatrix_Spatial.accuracy())
            return foldFeature.set('overallAccuracy_Random',overallAccuracy_Random).set('overallAccuracy_Spatial',overallAccuracy_Spatial)
        
        if modelType == 'REGRESSION':
            # Compute accuracy metrics
            r2ToSet_Random = coefficientOfDetermination(classifiedValidationData_Random,classProperty,outputtedPropName_Random)
            rmseToSet_Random = RMSE(classifiedValidationData_Random,classProperty,outputtedPropName_Random)
            maeToSet_Random = MAE(classifiedValidationData_Random,classProperty,outputtedPropName_Random)

            r2ToSet_Spatial = coefficientOfDetermination(classifiedValidationData_Spatial,classProperty,outputtedPropName_Spatial)
            rmseToSet_Spatial = RMSE(classifiedValidationData_Spatial,classProperty,outputtedPropName_Spatial)
            maeToSet_Spatial = MAE(classifiedValidationData_Spatial,classProperty,outputtedPropName_Spatial)
            return foldFeature.set('R2_Random',r2ToSet_Random).set('RMSE_Random', rmseToSet_Random).set('MAE_Random', maeToSet_Random)\
                                .set('R2_Spatial',r2ToSet_Spatial).set('RMSE_Spatial', rmseToSet_Spatial).set('MAE_Spatial', maeToSet_Spatial)

    # Compute the mean and std dev of the accuracy values of the classifier across all folds
    accuracyFC = kFoldAssignmentFC.map(computeAccuracyForFold)

    if modelType == 'REGRESSION':
        meanAccuracy_Random = accuracyFC.aggregate_mean('R2_Random')
        tsdAccuracy_Random = accuracyFC.aggregate_total_sd('R2_Random')
        meanAccuracy_Spatial = accuracyFC.aggregate_mean('R2_Spatial')
        tsdAccuracy_Spatial = accuracyFC.aggregate_total_sd('R2_Spatial')

        # Calculate mean and std dev of RMSE
        RMSEvals_Random = accuracyFC.aggregate_array('RMSE_Random')
        RMSEvalsSquared_Random = RMSEvals_Random.map(lambda f: ee.Number(f).multiply(f))
        sumOfRMSEvalsSquared_Random = RMSEvalsSquared_Random.reduce(ee.Reducer.sum())
        meanRMSE_Random = ee.Number.sqrt(ee.Number(sumOfRMSEvalsSquared_Random).divide(k))
        RMSEvals_Spatial = accuracyFC.aggregate_array('RMSE_Spatial')
        RMSEvalsSquared_Spatial = RMSEvals_Spatial.map(lambda f: ee.Number(f).multiply(f))
        sumOfRMSEvalsSquared_Spatial = RMSEvalsSquared_Spatial.reduce(ee.Reducer.sum())
        meanRMSE_Spatial = ee.Number.sqrt(ee.Number(sumOfRMSEvalsSquared_Spatial).divide(k))

        RMSEdiff_Random = accuracyFC.aggregate_array('RMSE_Random').map(lambda f: ee.Number(ee.Number(f).subtract(meanRMSE_Random)).pow(2))
        sumOfRMSEdiff_Random = RMSEdiff_Random.reduce(ee.Reducer.sum())
        sdRMSE_Random = ee.Number.sqrt(ee.Number(sumOfRMSEdiff_Random).divide(k))
        RMSEdiff_Spatial = accuracyFC.aggregate_array('RMSE_Spatial').map(lambda f: ee.Number(ee.Number(f).subtract(meanRMSE_Spatial)).pow(2))
        sumOfRMSEdiff_Spatial = RMSEdiff_Spatial.reduce(ee.Reducer.sum())
        sdRMSE_Spatial = ee.Number.sqrt(ee.Number(sumOfRMSEdiff_Spatial).divide(k))

        # Calculate mean and std dev of MAE
        meanMAE_Random = accuracyFC.aggregate_mean('MAE_Random')
        tsdMAE_Random= accuracyFC.aggregate_total_sd('MAE_Random')
        meanMAE_Spatial = accuracyFC.aggregate_mean('MAE_Spatial')
        tsdMAE_Spatial= accuracyFC.aggregate_total_sd('MAE_Spatial')

        # Compute the feature to return
        featureToReturn = featureWithClassifier.select(['cName']).set('Mean_R2_Random',meanAccuracy_Random,'StDev_R2_Random',tsdAccuracy_Random, 'Mean_RMSE_Random',meanRMSE_Random,'StDev_RMSE_Random',sdRMSE_Random, 'Mean_MAE_Random',meanMAE_Random,'StDev_MAE_Random',tsdMAE_Random)\
                                                                .set('Mean_R2_Spatial',meanAccuracy_Spatial,'StDev_R2_Spatial',tsdAccuracy_Spatial, 'Mean_RMSE_Spatial',meanRMSE_Spatial,'StDev_RMSE_Spatial',sdRMSE_Spatial, 'Mean_MAE_Spatial',meanMAE_Spatial,'StDev_MAE_Spatial',tsdMAE_Spatial)

    if modelType == 'CLASSIFICATION':
        accuracyFC_Random = kFoldAssignmentFC.map(computeAccuracyForFold)
        meanAccuracy_Random = accuracyFC_Random.aggregate_mean('overallAccuracy_Random')
        tsdAccuracy_Random = accuracyFC_Random.aggregate_total_sd('overallAccuracy_Random')
        accuracyFC_Spatial = kFoldAssignmentFC.map(computeAccuracyForFold)
        meanAccuracy_Spatial = accuracyFC_Spatial.aggregate_mean('overallAccuracy_Spatial')
        tsdAccuracy_Spatial = accuracyFC_Spatial.aggregate_total_sd('overallAccuracy_Spatial')

        # Compute the feature to return
        featureToReturn = featureWithClassifier.select(['cName']).set('Mean_overallAccuracy_Random',meanAccuracy_Random,'StDev_overallAccuracy_Random',tsdAccuracy_Random)\
                                                                .set('Mean_overallAccuracy_Spatial',meanAccuracy_Spatial,'StDev_overallAccuracy_Spatial',tsdAccuracy_Spatial)  

    return featureToReturn

####################################################################################################################################################################
# Initialization
####################################################################################################################################################################
# Turn the folder string into an assetID
assetIDToCreate_Folder = 'projects/johanvandenhoogen/assets/'+projectFolder
if any(x in subprocess.run(bashCommandList_Detect+[assetIDToCreate_Folder],stdout=subprocess.PIPE).stdout.decode('utf-8') for x in stringsOfInterest) == False:
    pass
else:
    # perform the folder creation
    print(assetIDToCreate_Folder,'being created...')

    # Create the folder within Earth Engine
    subprocess.run(bashCommandList_CreateFolder+[assetIDToCreate_Folder])
    while any(x in subprocess.run(bashCommandList_Detect+[assetIDToCreate_Folder],stdout=subprocess.PIPE).stdout.decode('utf-8') for x in stringsOfInterest):
        print('Waiting for asset to be created...')
        time.sleep(normalWaitTime)
    print('Asset created!')

    # Sleep to allow the server time to receive incoming requests
    time.sleep(normalWaitTime/2)

assetIDToCreate_Folder = 'projects/johanvandenhoogen/assets'+'/'+projectFolder+'/hyperparameter_tuning'
if any(x in subprocess.run(bashCommandList_Detect+[assetIDToCreate_Folder],stdout=subprocess.PIPE).stdout.decode('utf-8') for x in stringsOfInterest) == False:
    pass
else:
    # perform the folder creation
    print(assetIDToCreate_Folder,'being created...')

    # Create the folder within Earth Engine
    subprocess.run(bashCommandList_CreateFolder+[assetIDToCreate_Folder])
    while any(x in subprocess.run(bashCommandList_Detect+[assetIDToCreate_Folder],stdout=subprocess.PIPE).stdout.decode('utf-8') for x in stringsOfInterest):
        print('Waiting for asset to be created...')
        time.sleep(normalWaitTime)
    print('Asset created!')

    # Sleep to allow the server time to receive incoming requests
    time.sleep(normalWaitTime/2)

####################################################################################################################################################################
# Data processing
####################################################################################################################################################################
try:
    # try whether fcOI is present
    fcOI = ee.FeatureCollection(assetIDForCVAssignedColl)
    print(fcOI.size().getInfo(), 'features in', assetIDForCVAssignedColl)

    preppedCollection_wSpatialFolds = pd.read_csv(holdingFolder+'/'+titleOfCSVWithCVAssignments+'.csv')

except Exception as e:
    # Import raw data
    rawPointCollection = pd.read_csv('data/20241106_darkdiv_EM_richness_rarefied_sampled_oneHot.csv', float_precision='round_trip')
    print('Size of original Collection', rawPointCollection.shape[0])

    # Rename classification property column
    rawPointCollection.rename(columns={'rarefied': classProperty}, inplace=True)

    # Shuffle the data frame while setting a new index to ensure geographic clumps of points are not clumped in any way
    fcToAggregate = rawPointCollection.sample(frac = 1, random_state = 42).reset_index(drop=True)

    # Remove duplicates
    preppedCollection = fcToAggregate.drop_duplicates(subset = covariateList+[classProperty], keep = 'first')[['sample_id']+covariateList+["Resolve_Biome"]+[classProperty]+['Pixel_Lat', 'Pixel_Long']]
    print('Number of aggregated pixels', preppedCollection.shape[0])

    # Drop NAs
    preppedCollection = preppedCollection.dropna(how='any')
    print('After dropping NAs', preppedCollection.shape[0])

    # Log transform classProperty; if specified
    if log_transform_classProperty == True:
        preppedCollection[classProperty] = np.log(preppedCollection[classProperty] + 1)

    # Convert biome column to int, to correct odd rounding errors
    preppedCollection[stratificationVariableString] = preppedCollection[stratificationVariableString].astype(int)

    # Generate random folds, stratified by biome
    preppedCollection[cvFoldString_Random] = (preppedCollection.groupby('Resolve_Biome').cumcount() % k) + 1

    # Write the CSV to disk and upload it to Earth Engine as a Feature Collection
    localPathToCVAssignedData = holdingFolder+'/'+titleOfCSVWithCVAssignments+'.csv'
    preppedCollection.to_csv(localPathToCVAssignedData,index=False)

    # Generate spatial folds. Folds are added to the csv file
    # R_call = ["/Library/Frameworks/R.framework/Resources/bin/Rscript functions/generateFoldsForCV.R " + \
    # 		"--k 10 " + \
    # 		"--type Hexagon " +  \
    # 		"--crs EPSG:8857 " + \
    # 		"--seed 42 " + \
    # 		"--lon Pixel_Long " + \
    # 		"--lat Pixel_Lat " + \
    # 		"--path " + localPathToCVAssignedData]

    # # Run the R script
    # subprocess.run(R_call, shell = True) 

    # Generate folds with knndm function
    R_call = ["/Library/Frameworks/R.framework/Resources/bin/Rscript functions/generateFoldsKNNDM.R " + \
            "--path_training " + localPathToCVAssignedData + " " + \
            "--ppoints " + "data/filtered_randomPoints_ECM.csv " + \
            "--k 10 " + \
            "--maxp 0.5 " + \
            "--clustering hierarchical " + \
            "--linkf ward.D2 " + \
            "--samplesize 1000 " + \
            "--sampling regular" ]

    # Run the R script
    subprocess.run(R_call, shell = True) 

    # Determine block size for spatial folds
    # blockCVsize = determineBlockSizeForCV(localPathToCVAssignedData, 'Pixel_Lat', 'Pixel_Long', seed = 42, classProperty = classProperty)

    # Read in the CSV with the spatial folds
    preppedCollection_wSpatialFolds = pd.read_csv(localPathToCVAssignedData)

    # Retain only the spatial fold column of blockcvsize
    # preppedCollection_wSpatialFolds[cvFoldString_Spatial] = preppedCollection_wSpatialFolds['foldID_' + blockCVsize]
    # preppedCollection_wSpatialFolds = preppedCollection_wSpatialFolds.drop(columns = [x for x in preppedCollection_wSpatialFolds.columns if 'foldID_' in x])
    preppedCollection_wSpatialFolds[cvFoldString_Spatial] = preppedCollection_wSpatialFolds['knndmw_CV_folds']
    preppedCollection_wSpatialFolds = preppedCollection_wSpatialFolds.drop(columns = ['knndmw_CV_folds', 'geometry'])

    # Write the CSV to disk 
    preppedCollection_wSpatialFolds.to_csv(localPathToCVAssignedData,index=False)

    try:
        # try whether fcOI is present
        fcOI = ee.FeatureCollection('projects/johanvandenhoogen/assets'+'/'+projectFolder+'/'+titleOfCSVWithCVAssignments)

        # Print size of dataset
        print(guild, fcOI.size().getInfo())

    except Exception as e:
        # Format the bash call to upload the file to the Google Cloud Storage bucket
        gsutilBashUploadList = [bashFunctionGSUtil]+arglist_preGSUtilUploadFile+[localPathToCVAssignedData]+[formattedBucketOI]
        subprocess.run(gsutilBashUploadList)
        print(titleOfCSVWithCVAssignments+' uploaded to a GCSB!')

        # Wait for a short period to ensure the command has been received by the server
        time.sleep(normalWaitTime/2)

        # Wait for the GSUTIL uploading process to finish before moving on
        while not all(x in subprocess.run([bashFunctionGSUtil,'ls',formattedBucketOI],stdout=subprocess.PIPE).stdout.decode('utf-8') for x in [titleOfCSVWithCVAssignments]):
            print('Not everything is uploaded...')
            time.sleep(normalWaitTime)
        print('Everything is uploaded; moving on...')

        # Upload the file into Earth Engine as a table asset
        assetIDForCVAssignedColl = 'projects/johanvandenhoogen/assets'+'/'+projectFolder+'/'+titleOfCSVWithCVAssignments
        earthEngineUploadTableCommands = [bashFunction_EarthEngine]+arglist_preEEUploadTable+[assetIDStringPrefix+assetIDForCVAssignedColl]+[formattedBucketOI+'/'+titleOfCSVWithCVAssignments+'.csv']+arglist_postEEUploadTable
        subprocess.run(earthEngineUploadTableCommands)
        print('Upload to EE queued!')

        # Wait for a short period to ensure the command has been received by the server
        time.sleep(normalWaitTime/2)

        # !! Break and wait
        count = 1
        while count >= 1:
            taskList = [str(i) for i in ee.batch.Task.list()]
            subsetList = [s for s in taskList if classProperty in s]
            subsubList = [s for s in subsetList if any(xs in s for xs in ['RUNNING', 'READY'])]
            count = len(subsubList)
            print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Number of running jobs:', count)
            time.sleep(normalWaitTime)
        print('Moving on...')

##################################################################################################################################################################
# Hyperparameter tuning
##################################################################################################################################################################
fcOI = ee.FeatureCollection('projects/johanvandenhoogen/assets'+'/'+projectFolder+'/'+titleOfCSVWithCVAssignments)

# Define hyperparameters for grid search
varsPerSplit_list = list(range(4,14,2))
leafPop_list = list(range(2,14,2))

classifierListRegression = []
# Create list of classifiers for regression
for vps in varsPerSplit_list:
    for lp in leafPop_list:

        model_name = classProperty + '_rf_VPS' + str(vps) + '_LP' + str(lp) + '_REGRESSION'

        rf = ee.Feature(ee.Geometry.Point([0,0])).set('cName',model_name,'c',ee.Classifier.smileRandomForest(
        numberOfTrees = nTrees,
        variablesPerSplit = vps,
        minLeafPopulation = lp,
        bagFraction = 0.632,
        seed = 42
        ).setOutputMode('REGRESSION'))

        classifierListRegression.append(rf)

classifierListClassification = []
# Create list of classifiers for classification
for vps in varsPerSplit_list:
    for lp in leafPop_list:

        model_name = classProperty + '_rf_VPS' + str(vps) + '_LP' + str(lp) + 'CLASSIFICATION'

        rf = ee.Feature(ee.Geometry.Point([0,0])).set('cName',model_name,'c',ee.Classifier.smileRandomForest(
        numberOfTrees = nTrees,
        variablesPerSplit = vps,
        minLeafPopulation = lp,
        bagFraction = 0.632,
        seed = 42
        ).setOutputMode('CLASSIFICATION'))

        classifierListClassification.append(rf)

# # If grid search was not performed yet:
# Make a feature collection from the k-fold assignment list
kFoldAssignmentFC = ee.FeatureCollection(ee.List(kList).map(lambda n: ee.Feature(ee.Geometry.Point([0,0])).set('Fold',n)))

finished_models_regression = list()

# Check if any models have been completed
try:
    grid_search_resultsRegression = ee.FeatureCollection('projects/johanvandenhoogen/assets'+'/'+projectFolder+'/'+classProperty+'_Regression_grid_search_results')
    print(grid_search_resultsRegression.size().getInfo())

except Exception as e:
    try:
        # Create list of finished models
        finished_models_regression = subprocess.run(bashCommandList_ls+['projects/johanvandenhoogen/assets'+'/'+projectFolder+'/hyperparameter_tuning/'], stdout=subprocess.PIPE).stdout.splitlines()
        finished_models_regression = [model.decode('ascii').split('/')[-1] for model in finished_models_regression]
        
    except Exception as e:
        classDfRegression = pd.DataFrame(columns = ['Mean_R2', 'StDev_R2','Mean_RMSE', 'StDev_RMSE','Mean_MAE', 'StDev_MAE', 'cName'])

    # Perform model testing for remaining hyperparameter settings
    for rf in classifierListRegression:
        if rf.get('cName').getInfo() in finished_models_regression:
            print('Model', classifierListRegression.index(rf), 'out of total of', len(classifierListRegression), 'already finished')
        else:
            print('Testing model', classifierListRegression.index(rf), 'out of total of', len(classifierListRegression))
            #  train classifier only on data not equalling zero
            # train classifier only on GlobalFungi data
            fcOI = ee.FeatureCollection('projects/johanvandenhoogen/assets'+'/'+projectFolder+'/'+titleOfCSVWithCVAssignments)\
                    .filter(ee.Filter.neq(classProperty, 0))
            accuracy_feature = ee.Feature(computeCVAccuracyAndRMSE(rf))
            accuracy_featureExport = ee.batch.Export.table.toAsset(
                collection = ee.FeatureCollection([accuracy_feature]),
                description = classProperty+rf.get('cName').getInfo(),
                assetId = 'projects/johanvandenhoogen/assets'+'/'+projectFolder+'/hyperparameter_tuning/'+rf.get('cName').getInfo())
            accuracy_featureExport.start()

# Check if any models have been completed
finished_models_classification = list()
try:
    grid_search_resultsClassification = ee.FeatureCollection('projects/johanvandenhoogen/assets'+'/'+projectFolder+'/'+classProperty+'_Classification_grid_search_results')
    print(grid_search_resultsClassification.size().getInfo())
    
except Exception as e:
    try:
        # Create list of finished models
        finished_models_classification = subprocess.run(bashCommandList_ls+['projects/johanvandenhoogen/assets'+'/'+projectFolder+'/hyperparameter_tuning/'], stdout=subprocess.PIPE).stdout.splitlines()
        finished_models_classification = [model.decode('ascii').split('/')[-1] for model in finished_models_classification]

    except Exception as e:
        classDfClassification = pd.DataFrame(columns = ['Mean_overallAccuracy', 'StDev_overallAccuracy', 'cName'])

    # Perform model testing for remaining hyperparameter settings
    for rf in classifierListClassification:
        if rf.get('cName').getInfo() in finished_models_classification:
            print('Model', classifierListClassification.index(rf), 'out of total of', len(classifierListClassification), 'already finished')
        else:
            print('Testing model', classifierListClassification.index(rf), 'out of total of', len(classifierListClassification))

            fcOI = ee.FeatureCollection('projects/johanvandenhoogen/assets'+'/'+projectFolder+'/'+titleOfCSVWithCVAssignments)
            fcOI = fcOI.map(lambda f: f.set(classProperty, ee.Number(f.get(classProperty)).divide(f.get(classProperty)))) # train classifier on 0 (classProperty == 0) or 1 (classProperty != 0)
            categoricalLevels = fcOI.aggregate_array(classProperty).distinct().getInfo()

            accuracy_feature = ee.Feature(computeCVAccuracyAndRMSE(rf))
            accuracy_featureExport = ee.batch.Export.table.toAsset(
                collection = ee.FeatureCollection([accuracy_feature]),
                description = rf.get('cName').getInfo(),
                assetId = 'projects/johanvandenhoogen/assets'+'/'+projectFolder+'/hyperparameter_tuning/'+rf.get('cName').getInfo())
            accuracy_featureExport.start()

    # !! Break and wait
    count = 1
    while count >= 1:
        taskList = [str(i) for i in ee.batch.Task.list()]
        subsetList = [s for s in taskList if classProperty in s]
        subsubList = [s for s in subsetList if any(xs in s for xs in ['RUNNING', 'READY'])]
        count = len(subsubList)
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Number of running jobs:', count)
        time.sleep(normalWaitTime)
    print('Moving on...')

# Fetch FC from GEE
grid_search_resultsRegression = ee.FeatureCollection([ee.FeatureCollection('projects/johanvandenhoogen/assets'+'/'+projectFolder+'/hyperparameter_tuning/'+rf.get('cName').getInfo()) for rf in classifierListRegression]).flatten()
classDfRegression = GEE_FC_to_pd(grid_search_resultsRegression)
grid_search_resultsClassification = ee.FeatureCollection([ee.FeatureCollection('projects/johanvandenhoogen/assets'+'/'+projectFolder+'/hyperparameter_tuning/'+rf.get('cName').getInfo()) for rf in classifierListClassification]).flatten()
classDfClassification = GEE_FC_to_pd(grid_search_resultsClassification)

grid_search_resultsRegression_export = ee.batch.Export.table.toAsset(
    collection = grid_search_resultsRegression,
    description = classProperty+'_Regression_grid_search_results',
    assetId = 'projects/johanvandenhoogen/assets'+'/'+projectFolder+'/'+classProperty+'_Regression_grid_search_results')
grid_search_resultsRegression_export.start()

grid_search_resultsClassification_export = ee.batch.Export.table.toAsset(
    collection = grid_search_resultsClassification,
    description = classProperty+'_Classification_grid_search_results',
    assetId = 'projects/johanvandenhoogen/assets'+'/'+projectFolder+'/'+classProperty+'_Classification_grid_search_results')
grid_search_resultsClassification_export.start()

# Sort values
classDfSortedRegression = classDfRegression.sort_values([sort_acc_prop], ascending = False)
classDfSortedClassification = classDfClassification.sort_values(['Mean_overallAccuracy_Random'], ascending = False)

# Write model results to csv
classDfSortedRegression.to_csv('output/'+today+'_'+classProperty+'_grid_search_results_Regression_kNNDMW_guildsFixed.csv', index=False)
classDfSortedClassification.to_csv('output/'+today+'_'+classProperty+'_grid_search_results_Classification_kNNDMW_guildsFixed.csv', index=False)

# Get top model name
bestModelNameRegression = grid_search_resultsRegression.limit(1, sort_acc_prop, False).first().get('cName')
bestModelNameClassification = grid_search_resultsClassification.limit(1, 'Mean_overallAccuracy_Random', False).first().get('cName')

# Get top 10 models
top_10ModelsRegression = grid_search_resultsRegression.limit(10, sort_acc_prop, False).aggregate_array('cName')
top_10ModelsClassification = grid_search_resultsClassification.limit(10, 'Mean_overallAccuracy_Random', False).aggregate_array('cName')

print('Moving on...')

# ##################################################################################################################################################################
# # Predicted - Observed
# ##################################################################################################################################################################
# fcOI = ee.FeatureCollection('projects/johanvandenhoogen/assets'+'/'+projectFolder+'/'+titleOfCSVWithCVAssignments)

# try:
#     predObs_wResiduals = ee.FeatureCollection('projects/johanvandenhoogen/assets'+'/'+projectFolder+'/'+classProperty+'_pred_obs')
#     predObs_wResiduals.size().getInfo()

# except Exception as e:
#     for n in list(range(0,10)):
#         modelNameRegression = top_10ModelsRegression.get(n)
#         modelNameClassification = top_10ModelsClassification.get(n)

#         # Load the best model from the classifier list
#         classifierRegression = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierListRegression).filterMetadata('cName', 'equals', modelNameRegression).first()).get('c'))
#         classifierClassification = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierListClassification).filterMetadata('cName', 'equals', modelNameClassification).first()).get('c'))

#         # Train the classifier with the collection
#         # REGRESSION
#         fcOI_forRegression = fcOI.filter(ee.Filter.neq(classProperty, 0))
#         trainedClassiferRegression = classifierRegression.train(fcOI_forRegression, classProperty, covariateList)

#         # Classification
#         fcOI_forClassification = fcOI.map(lambda f: f.set(classProperty+'_forClassification', ee.Number(f.get(classProperty)).divide(f.get(classProperty)))) # train classifier on 0 (classProperty == 0) or 1 (classProperty != 0)
#         trainedClassiferClassification = classifierClassification.train(fcOI_forClassification, classProperty+'_forClassification', covariateList)

#         # Classify the FC
#         def classifyFunction(f):
#             classfiedRegression = ee.FeatureCollection([f]).classify(trainedClassiferRegression,classProperty+'_Regressed').first()
#             classfiedClassification = ee.FeatureCollection([f]).classify(trainedClassiferClassification,classProperty+'_Classified').first()

#             featureToReturn = classfiedRegression.set(classProperty+'_Classified', classfiedClassification.get(classProperty+'_Classified'))

#             # Calculate final predicted value as product of classification and regression
#             featureToReturn = featureToReturn.set(classProperty+'_Predicted', ee.Number(featureToReturn.get(classProperty+'_Classified')).multiply(ee.Number(featureToReturn.get(classProperty+'_Regressed'))))
#             return featureToReturn

#         # Classify fcOI
#         predObs = fcOI.map(classifyFunction)

#         # Add coordinates to FC
#         predObs = predObs.map(addLatLon)

#         # back-log transform predicted and observed values
#         if log_transform_classProperty == True:
#             predObs = predObs.map(lambda f: f.set(classProperty, ee.Number(f.get(classProperty)).exp().subtract(1)))
#             predObs = predObs.map(lambda f: f.set(classProperty+'_Predicted', ee.Number(f.get(classProperty+'_Predicted')).exp().subtract(1)))
#             predObs = predObs.map(lambda f: f.set(classProperty+'_Regressed', ee.Number(f.get(classProperty+'_Regressed')).exp().subtract(1)))

#         # Add residuals to FC
#         predObs_wResiduals = predObs.map(lambda f: f.set('Residual', ee.Number(f.get(classProperty+'_Predicted')).subtract(f.get(classProperty))))

#         # Export to Assets
#         predObsexport = ee.batch.Export.table.toAsset(
#             collection = predObs_wResiduals,
#             description = classProperty+'_pred_obs_rep_'+str(n),
#             assetId = 'projects/johanvandenhoogen/assets'+'/'+projectFolder+'/'+classProperty+'_pred_obs_rep_'+str(n)
#         )
#         predObsexport.start()

#     # !! Break and wait
#     count = 1
#     while count >= 1:
#         taskList = [str(i) for i in ee.batch.Task.list()]
#         subsetList = [s for s in taskList if classProperty in s]
#         subsubList = [s for s in subsetList if any(xs in s for xs in ['RUNNING', 'READY'])]
#         count = len(subsubList)
#         print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Waiting for pred/obs to complete...', end = '\r')
#         time.sleep(normalWaitTime)
#     print('Moving on...')

#     predObsList = []
#     for n in list(range(0,10)):
#         predObs = ee.FeatureCollection('projects/johanvandenhoogen/assets'+'/'+projectFolder+'/'+classProperty+'_pred_obs')
#         predObsList.append(predObs)

#     predObs_wResiduals = ee.FeatureCollection(predObsList).flatten()
        
# # Convert to pd
# predObs_df = GEE_FC_to_pd(predObs_wResiduals)

# # Group by sample ID to return mean across ensemble prediction
# predObs_df = pd.DataFrame(predObs_df.groupby('sample_id').mean().to_records())

# predObs_df.to_csv('output/'+today+'_'+classProperty+'_pred_obs.csv')

# print('Predicted Observed done, moving on...')

#################################################################################################################################################################
# Classify image
#################################################################################################################################################################
# Reference covariate levels for mapping:
sequencing_platform454Roche = ee.Image.constant(0)
sequencing_platformIllumina = ee.Image.constant(1) # Reference level
sequencing_platformIonTorrent = ee.Image.constant(0)
sequencing_platformPacBio = ee.Image.constant(0)
sample_typerhizosphere_soil = ee.Image.constant(0)
sample_typesoil = ee.Image.constant(1)  # Reference level
sample_typetopsoil = ee.Image.constant(0)
primers5_8S_Fun_ITS4_Fun = ee.Image.constant(0)
primersfITS7_ITS4 = ee.Image.constant(0)
primersfITS9_ITS4 = ee.Image.constant(0)
primersgITS7_ITS4 = ee.Image.constant(0)
primersgITS7_ITS4_then_ITS9_ITS4 = ee.Image.constant(0)
primersgITS7_ITS4_ITS4arch = ee.Image.constant(0)
primersgITS7_ITS4m = ee.Image.constant(0)
primersgITS7_ITS4ngs = ee.Image.constant(0)
primersgITS7ngs_ITS4ngsUni = ee.Image.constant(0)
primersITS_S2F___ITS3_mixed_1_1_ITS4 = ee.Image.constant(0)
primersITS1_ITS4 = ee.Image.constant(0)
primersITS1F_ITS4 = ee.Image.constant(0)
primersITS1F_ITS4_then_fITS7_ITS4 = ee.Image.constant(0)
primersITS1F_ITS4_then_ITS3_ITS4 = ee.Image.constant(0)
primersITS1ngs_ITS4ngs_or_ITS1Fngs_ITS4ngs = ee.Image.constant(0)
primersITS3_KYO2_ITS4 = ee.Image.constant(0)
primersITS3_ITS4 = ee.Image.constant(1) # Reference level
primersITS3ngs1_to_5___ITS3ngs10_ITS4ngs = ee.Image.constant(0)
primersITS3ngs1_to_ITS3ngs11_ITS4ngs = ee.Image.constant(0)
primersITS86F_ITS4 = ee.Image.constant(0)
primersITS9MUNngs_ITS4ngsUni = ee.Image.constant(0)

constant_imgs = ee.ImageCollection.fromImages([
    sequencing_platform454Roche,
    sequencing_platformIllumina,
    sequencing_platformIonTorrent,
    sequencing_platformPacBio,
    sample_typerhizosphere_soil,
    sample_typesoil,
    sample_typetopsoil,
    primers5_8S_Fun_ITS4_Fun,
    primersfITS7_ITS4,
    primersfITS9_ITS4,
    primersgITS7_ITS4,
    primersgITS7_ITS4_then_ITS9_ITS4,
    primersgITS7_ITS4_ITS4arch,
    primersgITS7_ITS4m,
    primersgITS7_ITS4ngs,
    primersgITS7ngs_ITS4ngsUni,
    primersITS_S2F___ITS3_mixed_1_1_ITS4,
    primersITS1_ITS4,
    primersITS1F_ITS4,
    primersITS1F_ITS4_then_fITS7_ITS4,
    primersITS1F_ITS4_then_ITS3_ITS4,
    primersITS1ngs_ITS4ngs_or_ITS1Fngs_ITS4ngs,
    primersITS3_KYO2_ITS4,
    primersITS3_ITS4, 
    primersITS3ngs1_to_5___ITS3ngs10_ITS4ngs,
    primersITS3ngs1_to_ITS3ngs11_ITS4ngs,
    primersITS86F_ITS4,
    primersITS9MUNngs_ITS4ngsUni,
]).toBands().rename([
    'sequencing_platform454Roche',
    'sequencing_platformIllumina',
    'sequencing_platformIonTorrent',
    'sequencing_platformPacBio',
    'sample_typerhizosphere_soil',
    'sample_typesoil',
    'sample_typetopsoil',
    'primers5_8S_Fun_ITS4_Fun',
    'primersfITS7_ITS4',
    'primersfITS9_ITS4',
    'primersgITS7_ITS4',
    'primersgITS7_ITS4_then_ITS9_ITS4',
    'primersgITS7_ITS4_ITS4arch',
    'primersgITS7_ITS4m',
    'primersgITS7_ITS4ngs',
    'primersgITS7ngs_ITS4ngsUni',
    'primersITS_S2F___ITS3_mixed_1_1_ITS4',
    'primersITS1_ITS4',
    'primersITS1F_ITS4',
    'primersITS1F_ITS4_then_fITS7_ITS4',
    'primersITS1F_ITS4_then_ITS3_ITS4',
    'primersITS1ngs_ITS4ngs_or_ITS1Fngs_ITS4ngs',
    'primersITS3_KYO2_ITS4',
    'primersITS3_ITS4',
    'primersITS3ngs1_to_5___ITS3ngs10_ITS4ngs',
    'primersITS3ngs1_to_ITS3ngs11_ITS4ngs',
    'primersITS86F_ITS4',
    'primersITS9MUNngs_ITS4ngsUni',
])
fcOI = ee.FeatureCollection('projects/johanvandenhoogen/assets'+'/'+projectFolder+'/'+titleOfCSVWithCVAssignments)

def finalImageClassification(compositeImg):
    if ensemble == False:
        # Load the best model from the classifier list
        classifierRegression = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierListRegression).filterMetadata('cName', 'equals', bestModelNameRegression).first()).get('c'))
        classifierClassification = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierListClassification).filterMetadata('cName', 'equals', bestModelNameClassification).first()).get('c'))

        # Train the classifier with the collection
        # REGRESSION
        fcOI_forRegression = fcOI.filter(ee.Filter.neq(classProperty, 0))
        trainedClassiferRegression = classifierRegression.train(fcOI_forRegression, classProperty, covariateList)

        # Classification
        fcOI_forClassification = fcOI.map(lambda f: f.set(classProperty+'_forClassification', ee.Number(f.get(classProperty)).divide(f.get(classProperty)))) # train classifier on 0 (classProperty == 0) or 1 (classProperty != 0)
        trainedClassiferClassification = classifierClassification.train(fcOI_forClassification, classProperty+'_forClassification', covariateList)

        # Classify the FC
        classifiedImage_Regression = compositeImg.classify(trainedClassiferRegression,classProperty+'_Regressed')
        classifiedImage_Classification = compositeImg.classify(trainedClassiferClassification,classProperty+'_Classified')

        # Calculate final predicted value as product of classification and regression
        classifiedImage = classifiedImage_Regression.multiply(classifiedImage_Classification).rename(classProperty+'_Predicted')
        # classifiedImage = classifiedImage_Regression.rename(classProperty+'_Predicted')

        finalImage = ee.Image.cat(classifiedImage, classifiedImage_Regression, classifiedImage_Classification)
        # finalImage = classifiedImage

        return finalImage

    if ensemble == True:
        def classifyImage(classifiers):
            modelNameRegression = ee.List(classifiers).get(0)
            modelNameClassification = ee.List(classifiers).get(1)
            # modelNameRegression = classifiers

            # Load the best model from the classifier list
            classifierRegression = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierListRegression).filterMetadata('cName', 'equals', modelNameRegression).first()).get('c'))
            classifierClassification = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierListClassification).filterMetadata('cName', 'equals', modelNameClassification).first()).get('c'))

            # Train the classifier with the collection
            # REGRESSION
            fcOI_forRegression = fcOI.filter(ee.Filter.neq(classProperty, 0))
            trainedClassiferRegression = classifierRegression.train(fcOI_forRegression, classProperty, covariateList)

            # Classification
            fcOI_forClassification = fcOI.map(lambda f: f.set(classProperty+'_forClassification', ee.Number(f.get(classProperty)).divide(f.get(classProperty)))) # train classifier on 0 (classProperty == 0) or 1 (classProperty != 0)
            trainedClassiferClassification = classifierClassification.train(fcOI_forClassification, classProperty+'_forClassification', covariateList)

            # Classify the FC
            classifiedImage_Regression = compositeImg.classify(trainedClassiferRegression,classProperty+'_Regressed')
            classifiedImage_Classification = compositeImg.classify(trainedClassiferClassification,classProperty+'_Classified')

            # Calculate final predicted value as product of classification and regression
            classifiedImage = classifiedImage_Regression.multiply(classifiedImage_Classification).rename(classProperty+'_Predicted')

            finalImage = ee.Image.cat(classifiedImage, classifiedImage_Regression, classifiedImage_Classification)
            # finalImage = classifiedImage_Regression

            return finalImage

        # Classify the images, return mean
        # classifiedImage = ee.ImageCollection(top_10ModelsRegression.zip(top_10ModelsClassification).map(classifyImage)).mean()
        classifiedImage_Regression = ee.ImageCollection(top_10ModelsRegression.zip(top_10ModelsClassification).map(classifyImage)).select(classProperty+'_Regressed').mean()
        # classifiedImage_Regression = ee.ImageCollection(top_10ModelsRegression.map(classifyImage)).select(classProperty+'_Regressed').mean()
        classifiedImage_Classification = ee.ImageCollection(top_10ModelsRegression.zip(top_10ModelsClassification).map(classifyImage)).select(classProperty+'_Classified').mode()
        classifiedImage = classifiedImage_Regression.addBands(classifiedImage_Classification)
    return classifiedImage

# Create appropriate composite image with bands to use
compositeToClassify = compositeOfInterest.addBands(constant_imgs).select(covariateList).reproject(compositeOfInterest.projection())
classifiedImage = finalImageClassification(compositeToClassify)

regressedImage = classifiedImage.select(classProperty+'_Regressed')
classifiedImage = classifiedImage.select(classProperty+'_Classified')
predictedImage = regressedImage.multiply(classifiedImage).rename(classProperty+'_Predicted')

ensemblePredictedImage = ee.Image.cat(regressedImage, classifiedImage, predictedImage)
# ensemblePredictedImage = regressedImage.rename(classProperty+'_Predicted')

# ##################################################################################################################################################################
# # Variable importance metrics
# ##################################################################################################################################################################
# if ensemble == False:
#     classifier = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierListRegression).filterMetadata('cName', 'equals', bestModelNameRegression).first()).get('c'))

#     # Train the classifier with the collection
#     trainedClassifer = classifier.train(fcOI, classProperty, covariateList)

#     # Get the feature importance from the trained classifier and write to a .csv file and as a bar plot as .png file
#     featureImportances = trainedClassifer.explain().get('importance').getInfo()

#     featureImportances = pd.DataFrame(featureImportances.items(),
#                                         columns=['Variable', 'Feature_Importance']).sort_values(by='Feature_Importance',
#                                                                                                 ascending=False)

#     # Scale values
#     featureImportances['Feature_Importance'] = featureImportances['Feature_Importance'] - featureImportances['Feature_Importance'].min()
#     featureImportances['Feature_Importance'] = featureImportances['Feature_Importance'] / featureImportances['Feature_Importance'].max()

# if ensemble == True:
#     # Instantiate empty dataframe
#     featureImportances = pd.DataFrame(columns=['Variable', 'Feature_Importance'])

#     for i in list(range(0,10)):
#         classifierName = top_10ModelsRegression.get(i)
#         classifier = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierListRegression).filterMetadata('cName', 'equals', classifierName).first()).get('c'))

#         # Train the classifier with the collection
#         trainedClassifer = classifier.train(fcOI, classProperty, covariateList)

#         # Get the feature importance from the trained classifier and write to a .csv file and as a bar plot as .png file
#         featureImportancesToAdd = trainedClassifer.explain().get('importance').getInfo()
#         featureImportancesToAdd = pd.DataFrame(featureImportancesToAdd.items(),
#                                             columns=['Variable', 'Feature_Importance']).sort_values(by='Feature_Importance',
#                                                                                                     ascending=False)
#         # Scale values
#         featureImportancesToAdd['Feature_Importance'] = featureImportancesToAdd['Feature_Importance'] - featureImportancesToAdd['Feature_Importance'].min()
#         featureImportancesToAdd['Feature_Importance'] = featureImportancesToAdd['Feature_Importance'] / featureImportancesToAdd['Feature_Importance'].max()

#         featureImportances = pd.concat([featureImportances, featureImportancesToAdd])

#     featureImportances = pd.DataFrame(featureImportances.groupby('Variable').mean().to_records())

# # Write to csv
# featureImportances.sort_values('Feature_Importance', ascending = False ,inplace = True)
# featureImportances.to_csv('output/'+today+'_'+classProperty+'_featureImportances.csv')

# # Create and save plot
# plt = featureImportances[:10].plot(x='Variable', y='Feature_Importance', kind='bar', legend=False,
#                                 title='Feature Importances')
# fig = plt.get_figure()
# fig.savefig('output/'+today+'_'+classProperty+'_featureImportances.png', bbox_inches='tight')

# print('Variable importance metrics complete! Moving on...')

##################################################################################################################################################################
# Bootstrapping
##################################################################################################################################################################
try:
    # Path to bootstrapped samples
    bootstrapFc = ee.FeatureCollection('projects/johanvandenhoogen/assets'+'/'+projectFolder+'/'+bootstrapSamples)
    print(bootstrapFc.size().getInfo())

except Exception as e:
    # Input the number of points to use for each bootstrap model: equal to number of observations in training dataset
    bootstrapModelSize = preppedCollection_wSpatialFolds.shape[0]

    # Run a for loop to create multiple bootstrap iterations and upload them to the Google Cloud Storage Bucket
    # Create an empty list to store all of the file name strings being uploaded (for later use)
    # fileNameList = []
    stratSample = preppedCollection_wSpatialFolds.head(0)

    for n in seedsToUseForBootstrapping:
        # Perform the subsetting
        sampleToConcat = preppedCollection_wSpatialFolds.groupby(stratificationVariableString, group_keys=False).apply(lambda x: x.sample(n=int(round((strataDict.get(x.name)/100)*bootstrapModelSize)), replace=True, random_state=n))
        sampleToConcat['bootstrapIteration'] = n
        stratSample = pd.concat([stratSample, sampleToConcat])

    # Format the title of the CSV and export it to a holding location
    fullLocalPath = holdingFolder+'/'+bootstrapSamples+'.csv'
    stratSample.to_csv(holdingFolder+'/'+bootstrapSamples+'.csv',index=False)

    # Format the bash call to upload the files to the Google Cloud Storage bucket
    gsutilBashUploadList = [bashFunctionGSUtil]+arglist_preGSUtilUploadFile+[fullLocalPath]+[formattedBucketOI]
    subprocess.run(gsutilBashUploadList)
    print(bootstrapSamples+' uploaded to a GCSB!')

    # Wait for the GSUTIL uploading process to finish before moving on
    while not all(x in subprocess.run([bashFunctionGSUtil,'ls',formattedBucketOI],stdout=subprocess.PIPE).stdout.decode('utf-8') for x in [bootstrapSamples]):
        print('Not everything is uploaded...')
        time.sleep(5)
    print('Everything is uploaded moving on...')

    # Upload the file into Earth Engine as a table asset
    assetIDForCVAssignedColl = 'projects/johanvandenhoogen/assets'+'/'+projectFolder+'/'+bootstrapSamples
    earthEngineUploadTableCommands = [bashFunction_EarthEngine]+arglist_preEEUploadTable+[assetIDStringPrefix+assetIDForCVAssignedColl]+[formattedBucketOI+'/'+bootstrapSamples+'.csv']+arglist_postEEUploadTable
    subprocess.run(earthEngineUploadTableCommands)
    print('Upload to EE queued!')

    # Wait for a short period to ensure the command has been received by the server
    time.sleep(normalWaitTime/2)

    # !! Break and wait
    count = 1
    while count >= 1:
        taskList = [str(i) for i in ee.batch.Task.list()]
        subsetList = [s for s in taskList if classProperty in s]
        subsubList = [s for s in subsetList if any(xs in s for xs in ['RUNNING', 'READY'])]
        count = len(subsubList)
        print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Number of running jobs:', count)
        time.sleep(normalWaitTime)
    print('Moving on...')

# Load the best model from the classifier list
classifierToBootstrapRegression = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierListRegression).filterMetadata('cName','equals',bestModelNameRegression).first()).get('c'))
classifierToBootstrapClassification = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierListClassification).filterMetadata('cName','equals',bestModelNameClassification).first()).get('c'))

# Create empty list to store all fcs
fcList = []
# Run a for loop to create multiple bootstrap iterations
for n in seedsToUseForBootstrapping:
    # Create the path to the collection
    collectionPath = 'projects/johanvandenhoogen/assets'+'/'+projectFolder+'/'+bootstrapSamples

    # Load the collection from the path
    fcToTrain = ee.FeatureCollection(collectionPath).filter(ee.Filter.eq('bootstrapIteration', n))

    # Append fc to list
    fcList.append(fcToTrain)

# Helper fucntion to train a RF classifier and classify the composite image
def bootstrapFunc(fc):
    # Train the classifier with the collection
    fcOI_forRegression = fc.filter(ee.Filter.neq(classProperty, 0)) #  train classifier only on data not equalling zero
    trainedClassiferRegression = classifierToBootstrapRegression.train(fcOI_forRegression, classProperty, covariateList)

    # Classification
    fcOI_forClassification = fcOI.map(lambda f: f.set(classProperty+'_forClassification', ee.Number(f.get(classProperty)).divide(f.get(classProperty)))) # train classifier on 0 (classProperty == 0) or 1 (classProperty != 0)
    trainedClassiferClassification = classifierToBootstrapClassification.train(fcOI_forClassification, classProperty+'_forClassification', covariateList)

    # Classify the image
    classifiedImageRegression = compositeToClassify.classify(trainedClassiferRegression,classProperty+'_Regressed')
    classifiedImageClassification = compositeToClassify.classify(trainedClassiferClassification,classProperty+'_Classified')

    classifiedImage = classifiedImageRegression.multiply(classifiedImageClassification).rename(classProperty+'_Predicted')
    # classifiedImage = classifiedImageRegression

    return classifiedImage

# Reduce bootstrap images to mean
meanImage = ee.ImageCollection.fromImages(list(map(bootstrapFunc, fcList))).reduce(
    reducer = ee.Reducer.mean()
)

# Reduce bootstrap images to lower and upper CIs
upperLowerCIImage = ee.ImageCollection.fromImages(list(map(bootstrapFunc, fcList))).reduce(
    reducer = ee.Reducer.percentile([2.5,97.5],['lower','upper'])
)

# Reduce bootstrap images to standard deviation
stdDevImage = ee.ImageCollection.fromImages(list(map(bootstrapFunc, fcList))).reduce(
    reducer = ee.Reducer.stdDev()
)

# Coefficient of Variation: stdDev divided by mean
coefOfVarImage = stdDevImage.divide(meanImage).rename('Bootstrapped_CoefOfVar')

##################################################################################################################################################################
# Univariate int-ext analysis
##################################################################################################################################################################
covariateList = [
'CGIAR_PET',
'CHELSA_BIO_Annual_Mean_Temperature',
'CHELSA_BIO_Annual_Precipitation',
'CHELSA_BIO_Max_Temperature_of_Warmest_Month',
'CHELSA_BIO_Precipitation_Seasonality',
'ConsensusLandCover_Human_Development_Percentage',
# 'ConsensusLandCoverClass_Barren',
# 'ConsensusLandCoverClass_Deciduous_Broadleaf_Trees',
# 'ConsensusLandCoverClass_Evergreen_Broadleaf_Trees',
# 'ConsensusLandCoverClass_Evergreen_Deciduous_Needleleaf_Trees',
# 'ConsensusLandCoverClass_Herbaceous_Vegetation',
# 'ConsensusLandCoverClass_Mixed_Other_Trees',
# 'ConsensusLandCoverClass_Shrubs',
'EarthEnvTexture_CoOfVar_EVI',
'EarthEnvTexture_Correlation_EVI',
'EarthEnvTexture_Homogeneity_EVI',
'EarthEnvTopoMed_AspectCosine',
'EarthEnvTopoMed_AspectSine',
'EarthEnvTopoMed_Elevation',
'EarthEnvTopoMed_Slope',
'EarthEnvTopoMed_TopoPositionIndex',
'EsaCci_BurntAreasProbability',
'GHS_Population_Density',
'GlobBiomass_AboveGroundBiomass',
# 'GlobPermafrost_PermafrostExtent',
'MODIS_NPP',
# 'PelletierEtAl_SoilAndSedimentaryDepositThicknesses',
'SG_Depth_to_bedrock',
'SG_Sand_Content_005cm',
'SG_SOC_Content_005cm',
'SG_Soil_pH_H2O_005cm',
]

# Create a feature collection with only the values from the image bands
fcForMinMax = fcOI.select(covariateList)

# Make a FC with the band names
fcWithBandNames = ee.FeatureCollection(ee.List(covariateList).map(lambda bandName: ee.Feature(None).set('BandName',bandName)))

def calcMinMax(f):
    bandBeingComputed = f.get('BandName')
    maxValueToSet = fcForMinMax.reduceColumns(ee.Reducer.minMax(),[bandBeingComputed])
    return f.set('MinValue',maxValueToSet.get('min')).set('MaxValue',maxValueToSet.get('max'))

# Map function
fcWithMinMaxValues = ee.FeatureCollection(fcWithBandNames).map(calcMinMax)

# Make two images from these values (a min and a max image)
maxValuesWNulls = fcWithMinMaxValues.toList(1000).map(lambda f: ee.Feature(f).get('MaxValue'))
maxDict = ee.Dictionary.fromLists(covariateList,maxValuesWNulls)
minValuesWNulls = fcWithMinMaxValues.toList(1000).map(lambda f: ee.Feature(f).get('MinValue'))
minDict = ee.Dictionary.fromLists(covariateList,minValuesWNulls)
minImage = minDict.toImage()
maxImage = maxDict.toImage()

totalBandsBinary = compositeToClassify.select(covariateList).gte(minImage.select(covariateList)).lt(maxImage.select(covariateList))
univariate_int_ext_image = totalBandsBinary.reduce('sum').divide(compositeToClassify.bandNames().length()).rename('univariate_pct_int_ext')
univariate_int_ext_image.bandNames().getInfo()
##################################################################################################################################################################
# Multivariate (PCA) int-ext analysis
##################################################################################################################################################################

# Input the proportion of variance that you would like to cover
propOfVariance = 90

# PCA interpolation/extrapolation helper function
def assessExtrapolation(fcOfInterest, propOfVariance):
    # Compute the mean and standard deviation of each band, then standardize the point data
    meanVector = fcOfInterest.mean()
    stdVector = fcOfInterest.std()
    standardizedData = (fcOfInterest-meanVector)/stdVector

    # Then standardize the composite from which the points were sampled
    meanList = meanVector.tolist()
    stdList = stdVector.tolist()
    bandNames = list(meanVector.index)
    meanImage = ee.Image(meanList).rename(bandNames)
    stdImage = ee.Image(stdList).rename(bandNames)
    standardizedImage = compositeToClassify.select(covariateList).subtract(meanImage).divide(stdImage)

    # Run a PCA on the point samples
    pcaOutput = PCA()
    pcaOutput.fit(standardizedData)

    # Save the cumulative variance represented by each PC
    cumulativeVariance = np.cumsum(np.round(pcaOutput.explained_variance_ratio_, decimals=4)*100)

    # Make a list of PC names for future organizational purposes
    pcNames = ['PC'+str(x) for x in range(1,fcOfInterest.shape[1]+1)]

    # Get the PC loadings as a data frame
    loadingsDF = pd.DataFrame(pcaOutput.components_,columns=[str(x)+'_Loads' for x in bandNames],index=pcNames)

    # Get the original data transformed into PC space
    transformedData = pd.DataFrame(pcaOutput.fit_transform(standardizedData,standardizedData),columns=pcNames)

    # Make principal components images, multiplying the standardized image by each of the eigenvectors
    # Collect each one of the images in a single image collection

    # First step: make an image collection wherein each image is a PC loadings image
    listOfLoadings = ee.List(loadingsDF.values.tolist())
    eePCNames = ee.List(pcNames)
    zippedList = eePCNames.zip(listOfLoadings)
    def makeLoadingsImage(zippedValue):
        return ee.Image.constant(ee.List(zippedValue).get(1)).rename(bandNames).set('PC',ee.List(zippedValue).get(0))
    loadingsImageCollection = ee.ImageCollection(zippedList.map(makeLoadingsImage))

    # Second step: multiply each of the loadings image by the standardized image and reduce it using a "sum"
    # to finalize the matrix multiplication
    def finalizePCImages(loadingsImage):
        PCName = ee.String(ee.Image(loadingsImage).get('PC'))
        return ee.Image(loadingsImage).multiply(standardizedImage).reduce('sum').rename([PCName]).set('PC',PCName)
    principalComponentsImages = loadingsImageCollection.map(finalizePCImages)

    # Choose how many principal components are of interest in this analysis based on amount of
    # variance explained
    numberOfComponents = sum(i < propOfVariance for i in cumulativeVariance)+1
    print('Number of Principal Components being used:',numberOfComponents)

    # Compute the combinations of the principal components being used to compute the 2-D convex hulls
    tupleCombinations = list(combinations(list(pcNames[0:numberOfComponents]),2))
    print('Number of Combinations being used:',len(tupleCombinations))

    # Generate convex hulls for an example of the principal components of interest
    cHullCoordsList = list()
    for c in tupleCombinations:
        firstPC = c[0]
        secondPC = c[1]
        outputCHull = ConvexHull(transformedData[[firstPC,secondPC]])
        listOfCoordinates = transformedData.loc[outputCHull.vertices][[firstPC,secondPC]].values.tolist()
        flattenedList = [val for sublist in listOfCoordinates for val in sublist]
        cHullCoordsList.append(flattenedList)

    # Reformat the image collection to an image with band names that can be selected programmatically
    pcImage = principalComponentsImages.toBands().rename(pcNames)

    # Generate an image collection with each PC selected with it's matching PC
    listOfPCs = ee.List(tupleCombinations)
    listOfCHullCoords = ee.List(cHullCoordsList)
    zippedListPCsAndCHulls = listOfPCs.zip(listOfCHullCoords)

    def makeToClassifyImages(zippedListPCsAndCHulls):
        imageToClassify = pcImage.select(ee.List(zippedListPCsAndCHulls).get(0)).set('CHullCoords',ee.List(zippedListPCsAndCHulls).get(1))
        classifiedImage = imageToClassify.rename('u','v').classify(ee.Classifier.spectralRegion([imageToClassify.get('CHullCoords')]))
        return classifiedImage

    classifedImages = ee.ImageCollection(zippedListPCsAndCHulls.map(makeToClassifyImages))
    finalImageToExport = classifedImages.sum().divide(ee.Image.constant(len(tupleCombinations)))

    return finalImageToExport

# PCA interpolation-extrapolation image
PCA_int_ext = assessExtrapolation(preppedCollection_wSpatialFolds[covariateList], propOfVariance).rename('PCA_pct_int_ext')

# underExploredMaps = ee.Image.cat(
#     univariate_int_ext_image.rename('univariate_pct_int_ext'),
#     PCA_int_ext.rename('PCA_pct_int_ext'))
# IntExtclassifiedImageExport = ee.batch.Export.image.toAsset(
#     image = underExploredMaps.toFloat(),
#     description = classProperty+'_IntExt',
#     assetId = 'projects/johanvandenhoogen/assets'+'/'+projectFolder+'/'+classProperty+'_IntExt',
#     crs = 'EPSG:4326',
#     crsTransform = '[0.08333333333333333,0,-180,0,-0.08333333333333333,90]',
#     region = exportingGeometry,
#     maxPixels = int(1e13),
#     pyramidingPolicy = {".default": pyramidingPolicy}
# )
# IntExtclassifiedImageExport.start()

##################################################################################################################################################################
# Final image export
##################################################################################################################################################################

# Construct final image to export
if log_transform_classProperty == True:
    finalImageToExport = ee.Image.cat(
    ensemblePredictedImage.select(0).exp().subtract(1).rename(classProperty+'_Ensemble_mean'),
    meanImage.exp().subtract(1).rename(classProperty+'_Bootstrapped_mean'),
    upperLowerCIImage.select(0).exp().subtract(1).rename(classProperty+'_Bootstrapped_lower'),
    upperLowerCIImage.select(1).exp().subtract(1).rename(classProperty+'_Bootstrapped_upper'),
    stdDevImage.exp().subtract(1).rename(classProperty+'_Bootstrapped_stdDev'),
    coefOfVarImage.exp().subtract(1).rename(classProperty+'_Bootstrapped_coefOfVar'),
    univariate_int_ext_image.rename('univariate_pct_int_ext'),
    PCA_int_ext.rename('PCA_pct_int_ext'))
else:
    finalImageToExport = ee.Image.cat(
    ensemblePredictedImage.select(0).rename(classProperty+'_Ensemble_mean'),
    meanImage.rename(classProperty+'_Bootstrapped_mean'),
    upperLowerCIImage.select(0).rename(classProperty+'_Bootstrapped_lower'),
    upperLowerCIImage.select(1).rename(classProperty+'_Bootstrapped_upper'),
    stdDevImage.rename(classProperty+'_Bootstrapped_stdDev'),
    coefOfVarImage.rename(classProperty+'_Bootstrapped_coefOfVar'),
    univariate_int_ext_image.rename('univariate_pct_int_ext'),
    PCA_int_ext.rename('PCA_pct_int_ext'))

FinalImageExport = ee.batch.Export.image.toAsset(
    image = finalImageToExport.toFloat(),
    description = classProperty+'_Bootstrapped_MultibandImage',
    assetId = 'projects/johanvandenhoogen/assets'+'/'+projectFolder+'/'+classProperty+'_Classified_MultibandImage',
    crs = 'EPSG:4326',
    crsTransform = '[0.008333333333333333,0,-180,0,-0.008333333333333333,90]',
    region = exportingGeometry,
    maxPixels = int(1e13),
    pyramidingPolicy = {".default": pyramidingPolicy}
)
FinalImageExport.start()

print('Map exports started! Moving on...')

##################################################################################################################################################################
# Spatial Leave-One-Out cross validation
##################################################################################################################################################################
'''assetIDToCreate_Folder = 'projects/crowtherlab/johan/SPUN/EM_sloo_cv'
if any(x in subprocess.run(bashCommandList_Detect+[assetIDToCreate_Folder],stdout=subprocess.PIPE).stdout.decode('utf-8') for x in stringsOfInterest) == False:
    pass
else:
    # perform the folder creation
    print(assetIDToCreate_Folder,'being created...')

    # Create the folder within Earth Engine
    subprocess.run(bashCommandList_CreateFolder+[assetIDToCreate_Folder])
    while any(x in subprocess.run(bashCommandList_Detect+[assetIDToCreate_Folder],stdout=subprocess.PIPE).stdout.decode('utf-8') for x in stringsOfInterest):
        print('Waiting for asset to be created...')
        time.sleep(normalWaitTime)
    print('Asset created!')

    # Sleep to allow the server time to receive incoming requests
    time.sleep(normalWaitTime/2)


# !! NOTE: this is a fairly computatinally intensive excercise, so there are some precautions to take to ensure servers aren't overloaded
# !! This operaion SHOULD NOT be performed on the entire dataset

# Define buffer sizes to test (in meters)
buffer_sizes = [1000, 2500, 5000, 10000, 50000, 100000, 250000, 500000, 750000, 1000000, 1500000, 2000000]

# # Set number of random points to test
# if preppedCollection.shape[0] > 1000:
#     n_points = 1000 # Don't increase this value!
# else:
#     n_points = preppedCollection.shape[0]
n_points = 1000

# Set number of repetitions
n_reps = 10
nList = list(range(0,n_reps))

# Perform BLOO-CV
for rep in nList:
    for buffer in buffer_sizes:
        # mapList = []
        # for item in nList:
        #     mapList = mapList + (list(zip([buffer], repeat(item))))

        # Make a feature collection from the buffer sizes list
        # fc_toMap = ee.FeatureCollection(ee.List(mapList).map(lambda n: ee.Feature(ee.Geometry.Point([0,0])).set('buffer_size',ee.List(n).get(0)).set('rep',ee.List(n).get(1))))
        fc_toMap = ee.FeatureCollection(ee.Feature(ee.Geometry.Point([0,0])).set('buffer_size',buffer).set('rep',rep))

        grid_search_resultsRegression = ee.FeatureCollection('projects/johanvandenhoogen/assets'+'/'+projectFolder+'/'+classProperty+'_Regression_grid_search_results')

        # Get top model name
        bestModelNameRegression = grid_search_resultsRegression.limit(1, 'Mean_R2', False).first().get('cName')

        # Get top 10 models
        top_10ModelsRegression = grid_search_resultsRegression.limit(10, 'Mean_R2', False).aggregate_array('cName')

        # Helper function 1: assess whether point is within sampled range
        def WithinRange(f):
            testFeature = f
            # Training FeatureCollection: all samples not within geometry of test feature
            trainFC = fcOI.filter(ee.Filter.geometry(f.geometry()).Not())

            # Make a FC with the band names
            fcWithBandNames = ee.FeatureCollection(ee.List(covariateList).map(lambda bandName: ee.Feature(None).set('BandName',bandName)))

            # Helper function 1b: assess whether training point is within sampled range; per band
            def getRange(f):
                bandBeingComputed = f.get('BandName')
                minValue = trainFC.aggregate_min(bandBeingComputed)
                maxValue = trainFC.aggregate_max(bandBeingComputed)
                testFeatureWithinRange = ee.Number(testFeature.get(bandBeingComputed)).gte(ee.Number(minValue)).bitwiseAnd(ee.Number(testFeature.get(bandBeingComputed)).lte(ee.Number(maxValue)))
                return f.set('within_range', testFeatureWithinRange)

            # Return value of 1 if all bands are within sampled range
            within_range = fcWithBandNames.map(getRange).aggregate_min('within_range')

            return f.set('within_range', within_range)

        # Helper function 1: Spatial Leave One Out cross-validation function:
        def BLOOcv(f):
            rep = f.get('rep')
            # Test feature
            testFeature = ee.FeatureCollection(f)

            # Training set: all samples not within geometry of test feature
            trainFC = fcOI.filter(ee.Filter.neq(classProperty, 0)).filter(ee.Filter.geometry(testFeature).Not())

            # Classifier to test: same hyperparameter settings as from grid search procedure
            classifierName = top_10ModelsRegression.get(rep)
            classifier = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierListRegression).filterMetadata('cName', 'equals', classifierName).first()).get('c'))

            # Train classifier
            trainedClassifer = classifier.train(trainFC, classProperty, covariateList)

            # Apply classifier
            classified = testFeature.classify(classifier = trainedClassifer, outputName = 'predicted')

            # Get predicted value
            predicted = classified.first().get('predicted')

            # Set predicted value to feature
            return f.set('predicted', predicted).copyProperties(f)

        # Helper function 2: R2 calculation function
        def calc_R2(f):
            rep = f.get('rep')
            # FeatureCollection holding the buffer radius
            buffer_size = f.get('buffer_size')

            # Sample 1000 validation points from the data
            fc_withRandom = fcOI.filter(ee.Filter.neq(classProperty, 0)).randomColumn(seed = rep)
            subsetData = fc_withRandom.sort('random').limit(n_points)

            # Add the iteration ID to the FC
            fc_toValidate = subsetData.map(lambda f: f.set('rep', rep))

            # Add the buffer around the validation data
            fc_wBuffer = fc_toValidate.map(lambda f: f.buffer(buffer_size))

            # Remove points not within sampled range
            fc_withinSampledRange = fc_wBuffer.map(WithinRange).filter(ee.Filter.eq('within_range', 1))

            # Apply blocked leave one out CV function
            # predicted = fc_withinSampledRange.map(BLOOcv)
            predicted = fc_wBuffer.map(BLOOcv)

            # Calculate R2 value
            R2_val = coefficientOfDetermination(predicted, classProperty, 'predicted')

            return(f.set('R2_val', R2_val))

        # Calculate R2 across range of buffer sizes
        sloo_cv = fc_toMap.map(calc_R2)

        # Export FC to assets
        bloo_cv_fc_export = ee.batch.Export.table.toAsset(
            collection = sloo_cv,
            description = classProperty+'_sloo_cv_results_woExtrapolation_'+str(buffer),
            assetId = 'projects/crowtherlab/johan/SPUN/EM_sloo_cv/'+classProperty+'_sloo_cv_results_wExtrapolation_'+str(buffer)+'_rep_'+str(rep),
        )

        # bloo_cv_fc_export.start()

    print('Blocked Leave-One-Out started! Moving on...')
'''