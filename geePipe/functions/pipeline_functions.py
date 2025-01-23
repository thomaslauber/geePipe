
import pandas as pd
import numpy as np
import subprocess
import time
import ee
import yaml
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from itertools import combinations

from config.config import get_config
from functions.helper_functions import upload_table
from functions.helper_functions import GEE_FC_to_pd
from functions.helper_functions import addLatLon
from functions.helper_functions import breakandwait

from config.config import PIPELINE_PARAMS

# Function to initialize project folder in Google Earth Engine
def gee_folder_initialization():     
    # Turn the folder string into an assetID
    if any(x in subprocess.run(
            PIPELINE_PARAMS['cloud_params']['bash_function_earthengine'] + 
            PIPELINE_PARAMS['cloud_params']['gcs_detect'] + 
            [PIPELINE_PARAMS['general_params']['gee_project_folder']],
            stdout=subprocess.PIPE).stdout.decode('utf-8') for x in PIPELINE_PARAMS['cloud_params']['asset_doesnt_exist']) == False:
        pass
    else:
        # perform the folder creation
        print(PIPELINE_PARAMS['general_params']['gee_project_folder'],'being created...')

        # Create the folder within Earth Engine
        subprocess.run(PIPELINE_PARAMS['cloud_params']['bash_function_earthengine'] + 
                       PIPELINE_PARAMS['cloud_params']['gcs_create_folder'] + 
                       [PIPELINE_PARAMS['general_params']['gee_project_folder']])
        while any(x in subprocess.run(
                PIPELINE_PARAMS['cloud_params']['bash_function_earthengine'] + 
                PIPELINE_PARAMS['cloud_params']['gcs_detect'] + 
                [PIPELINE_PARAMS['general_params']['gee_project_folder']],
                stdout=subprocess.PIPE).stdout.decode('utf-8') for x in PIPELINE_PARAMS['cloud_params']['asset_doesnt_exist']):
            print('Waiting for asset to be created...')
            time.sleep(PIPELINE_PARAMS['general_params']['wait_time'])
        print('Asset created!')

        # Sleep to allow the server time to receive incoming requests
        time.sleep(PIPELINE_PARAMS['general_params']['wait_time']/2)

def data_processing():
    try:
        # try whether fcOI is present
        fcOI = ee.FeatureCollection(PIPELINE_PARAMS['general_params']['gee_project_folder']+PIPELINE_PARAMS['general_params']['training_data_filename'])
        print(fcOI.size().getInfo(), 'features in training dataset')

        preppedCollection_wSpatialFolds = pd.read_csv(PIPELINE_PARAMS['general_params']['training_data_filename']+'.csv')

        return fcOI, preppedCollection_wSpatialFolds
        
    except Exception as e:
        rawPointCollection = pd.read_csv(PIPELINE_PARAMS['local_settings']['raw_data'], float_precision = 'round_trip')
        print('Size of original Collection', rawPointCollection.shape[0])

        # Rename classification property column
        rawPointCollection.rename(columns={'rarefied': PIPELINE_PARAMS['model_params']['response_var']}, inplace=True)

        # Shuffle the data frame while setting a new index to ensure geographic clumps of points are not clumped in any way
        fcToAggregate = rawPointCollection.sample(frac = 1, random_state = PIPELINE_PARAMS['model_params']['seed']).reset_index(drop = True)

        # Remove duplicates
        preppedCollection = fcToAggregate.drop_duplicates(subset = PIPELINE_PARAMS['model_params']['covariate_list']+[PIPELINE_PARAMS['model_params']['response_var']], keep = 'first')[['sample_id']+PIPELINE_PARAMS['model_params']['covariate_list']+[PIPELINE_PARAMS['cv_params']['stratification_var']]+[PIPELINE_PARAMS['model_params']['response_var']]+['Pixel_Lat', 'Pixel_Long']]
        print('Number of aggregated pixels', preppedCollection.shape[0])

        # Drop NAs
        preppedCollection = preppedCollection.dropna(how='any')
        print('After dropping NAs', preppedCollection.shape[0])

        # Log transform response_var; if specified
        if PIPELINE_PARAMS['model_params']['log_transform_response_var'] == True:
            preppedCollection[PIPELINE_PARAMS['model_params']['response_var']] = np.log(preppedCollection[PIPELINE_PARAMS['model_params']['response_var']] + 1)

        # Convert biome column to int to correct rounding errors
        preppedCollection[PIPELINE_PARAMS['cv_params']['stratification_var']] = preppedCollection[PIPELINE_PARAMS['cv_params']['stratification_var']].astype(int)

        # Generate random folds, stratified by biome
        preppedCollection[PIPELINE_PARAMS['cv_params']['stratification_var']] = (preppedCollection.groupby('Resolve_Biome').cumcount() % PIPELINE_PARAMS['cv_params']['k']) + 1

        # Write the CSV to disk and upload it to Earth Engine as a Feature Collection
        preppedCollection.to_csv('data/'+PIPELINE_PARAMS['general_params']['training_data_filename']+'.csv', index=False)

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
                "--path_training " + 'data/' + PIPELINE_PARAMS['general_params']['training_data_filename']+'.csv' + " " + \
                "--ppoints " + "data/filtered_randomPoints.csv " + \
                "--k 10 " + \
                "--maxp 0.5 " + \
                "--clustering hierarchical " + \
                "--linkf ward.D2 " + \
                "--samplesize 1000 " + \
                "--sampling regular" ]

        # Run the R script
        subprocess.run(R_call, shell = True) 

        # Determine block size for spatial folds
        # blockCVsize = determineBlockSizeForCV(localPathToCVAssignedData, 'Pixel_Lat', 'Pixel_Long', seed = 42, PIPELINE_PARAMS['model_params']['response_var'] = PIPELINE_PARAMS['model_params']['response_var'])

        # Read in the CSV with the spatial folds
        preppedCollection_wSpatialFolds = pd.read_csv('data/'+PIPELINE_PARAMS['general_params']['training_data_filename']+'.csv')

        # Retain only the spatial fold column of blockcvsize
        # preppedCollection_wSpatialFolds[cvFoldString_Spatial] = preppedCollection_wSpatialFolds['foldID_' + blockCVsize]
        # preppedCollection_wSpatialFolds = preppedCollection_wSpatialFolds.drop(columns = [x for x in preppedCollection_wSpatialFolds.columns if 'foldID_' in x])
        preppedCollection_wSpatialFolds[PIPELINE_PARAMS['cv_params']['cv_fold_header_spatial']] = preppedCollection_wSpatialFolds['knndmw_CV_folds']
        preppedCollection_wSpatialFolds = preppedCollection_wSpatialFolds.drop(columns = ['knndmw_CV_folds', 'geometry'])

        # Write the CSV to disk 
        preppedCollection_wSpatialFolds.to_csv('data/'+PIPELINE_PARAMS['general_params']['training_data_filename']+'.csv', index=False)

        try:
            # try whether fcOI is present
            fcOI = ee.FeatureCollection(PIPELINE_PARAMS['general_params']['gee_project_folder']+PIPELINE_PARAMS['general_params']['training_data_filename'])

            # Print size of dataset
            print('Size of training dataset: ', fcOI.size().getInfo())

        except Exception as e:
            # Upload the file into Earth Engine as a table asset
            upload_table('data/'+PIPELINE_PARAMS['general_params']['training_data_filename']+'.csv')
        
        return fcOI, preppedCollection_wSpatialFolds

def predicted_observed(top_10Models, classifierList):
    fcOI = ee.FeatureCollection(PIPELINE_PARAMS['general_params']['gee_project_folder']+PIPELINE_PARAMS['general_params']['training_data_filename'])

    try:
        predObs_wResiduals = ee.FeatureCollection(PIPELINE_PARAMS['general_params']['gee_project_folder']+PIPELINE_PARAMS['model_params']['response_var']+'_pred_obs')
        predObs_wResiduals.size().getInfo()

    except Exception as e:
        for n in list(range(0,10)):
            modelNameRegression = top_10Models.get(n)

            # Load the best model from the classifier list
            classifier = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierList).filterMetadata('cName', 'equals', modelNameRegression).first()).get('c'))

            # Train the classifier with the collection
            fcOI_forRegression = fcOI.filter(ee.Filter.neq(PIPELINE_PARAMS['model_params']['response_var'], 0))
            trainedClassiferRegression = classifier.train(fcOI_forRegression, PIPELINE_PARAMS['model_params']['response_var'], PIPELINE_PARAMS['general_params']['covariate_list'])

            # Classify the FC
            def classifyFunction(f):
                featureToReturn = ee.FeatureCollection([f]).classify(trainedClassiferRegression,PIPELINE_PARAMS['model_params']['response_var']+'_Regressed').first()

                return featureToReturn

            # Classify fcOI
            predObs = fcOI.map(classifyFunction)

            # Add coordinates to FC
            predObs = predObs.map(addLatLon)
            predObs = addLatLon(predObs, PIPELINE_PARAMS['general_params']['lat_string'], PIPELINE_PARAMS['general_params']['long_string'])

            # back-log transform predicted and observed values
            if PIPELINE_PARAMS['model_params']['log_transform_response_var'] == True:
                predObs = predObs.map(lambda f: f.set(PIPELINE_PARAMS['model_params']['response_var'], ee.Number(f.get(PIPELINE_PARAMS['model_params']['response_var'])).exp().subtract(1)))
                predObs = predObs.map(lambda f: f.set(PIPELINE_PARAMS['model_params']['response_var']+'_Predicted', ee.Number(f.get(PIPELINE_PARAMS['model_params']['response_var']+'_Predicted')).exp().subtract(1)))
                predObs = predObs.map(lambda f: f.set(PIPELINE_PARAMS['model_params']['response_var']+'_Regressed', ee.Number(f.get(PIPELINE_PARAMS['model_params']['response_var']+'_Regressed')).exp().subtract(1)))

            # Add residuals to FC
            predObs_wResiduals = predObs.map(lambda f: f.set('Residual', ee.Number(f.get(PIPELINE_PARAMS['model_params']['response_var']+'_Predicted')).subtract(f.get(PIPELINE_PARAMS['model_params']['response_var']))))

            # Export to Assets
            predObsexport = ee.batch.Export.table.toAsset(
                collection = predObs_wResiduals,
                description = PIPELINE_PARAMS['model_params']['response_var']+'_pred_obs_rep_'+str(n),
                assetId = PIPELINE_PARAMS['general_params']['gee_project_folder'] + '/' + PIPELINE_PARAMS['model_params']['response_var']+'_pred_obs_rep_'+str(n)
            )
            predObsexport.start()

        breakandwait(PIPELINE_PARAMS['model_params']['response_var'])

        predObsList = []
        for n in list(range(0,10)):
            predObs = ee.FeatureCollection(PIPELINE_PARAMS['general_params']['gee_project_folder']+'/'+PIPELINE_PARAMS['model_params']['response_var']+'_pred_obs')
            predObsList.append(predObs)

        predObs_wResiduals = ee.FeatureCollection(predObsList).flatten()
            
    # Convert to pd
    predObs_df = GEE_FC_to_pd(predObs_wResiduals)

    # Group by sample ID to return mean across ensemble prediction
    predObs_df = pd.DataFrame(predObs_df.groupby('sample_id').mean().to_records())

    predObs_df.to_csv('data/'+PIPELINE_PARAMS['today']+'_'+PIPELINE_PARAMS['model_params']['response_var']+'_pred_obs.csv', index=False)

    print('Predicted Observed done, moving on...')

    return predObs_df

def image_classification(composite_img, bestModelName, top_10Models, classifierList):
    fcOI = ee.FeatureCollection(PIPELINE_PARAMS['general_params']['gee_project_folder']+PIPELINE_PARAMS['general_params']['training_data_filename'])

    if PIPELINE_PARAMS['model_params']['ensemble'] == False:
        # Load the best model from the classifier list
        classifierRegression = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierList).filterMetadata('cName', 'equals', bestModelName).first()).get('c'))

        # Train the classifier with the collection
        # REGRESSION
        trainedClassiferRegression = classifierRegression.train(fcOI, PIPELINE_PARAMS['model_params']['response_var'], PIPELINE_PARAMS['model_params']['covariate_list'])

        # Classify the FC
        classifiedImage = composite_img.classify(trainedClassiferRegression,PIPELINE_PARAMS['model_params']['response_var']+'_predicted')

        return classifiedImage

    if PIPELINE_PARAMS['model_params']['ensemble'] == True:
        def classifyImage(classifier):
            # Load the best model from the classifier list
            classifier = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierList).filterMetadata('cName', 'equals', classifier).first()).get('c'))

            # Train the classifier with the collection
            # REGRESSION
            trainedClassifer = classifier.train(fcOI, PIPELINE_PARAMS['model_params']['response_var'], PIPELINE_PARAMS['model_params']['covariate_list'])

            # Classify the FC
            classifiedImage = composite_img.classify(trainedClassifer,PIPELINE_PARAMS['model_params']['response_var']+'_predicted')

            return classifiedImage

        # Classify the images, return mean
        classifiedImage = ee.ImageCollection(top_10Models.map(classifyImage)).select(PIPELINE_PARAMS['model_params']['response_var']+'_predicted').mean()
    
    return classifiedImage

def variable_importance(classifierList, bestModelName, top_10Models):
    fcOI = ee.FeatureCollection(PIPELINE_PARAMS['general_params']['gee_project_folder']+PIPELINE_PARAMS['general_params']['training_data_filename'])

    if PIPELINE_PARAMS['model_params']['ensemble'] == False:
        classifier = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierList).filterMetadata('cName', 'equals', bestModelName).first()).get('c'))

        # Train the classifier with the collection
        trainedClassifer = classifier.train(fcOI, PIPELINE_PARAMS['model_params']['response_var'], PIPELINE_PARAMS['model_params']['covariate_list'])

        # Get the feature importance from the trained classifier and write to a .csv file and as a bar plot as .png file
        featureImportances = trainedClassifer.explain().get('importance').getInfo()

        featureImportances = pd.DataFrame(featureImportances.items(),
                                            columns=['Variable', 'Feature_Importance']).sort_values(by='Feature_Importance',
                                                                                                    ascending=False)

        # Scale values
        featureImportances['Feature_Importance'] = featureImportances['Feature_Importance'] - featureImportances['Feature_Importance'].min()
        featureImportances['Feature_Importance'] = featureImportances['Feature_Importance'] / featureImportances['Feature_Importance'].max()

    if PIPELINE_PARAMS['model_params']['ensemble'] == True:
        # Instantiate empty dataframe
        featureImportances = pd.DataFrame(columns=['Variable', 'Feature_Importance'])

        for i in list(range(0,10)):
            classifierName = top_10Models.get(i)
            classifier = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierList).filterMetadata('cName', 'equals', classifierName).first()).get('c'))

            # Train the classifier with the collection
            trainedClassifer = classifier.train(fcOI, PIPELINE_PARAMS['model_params']['response_var'], PIPELINE_PARAMS['model_params']['covariate_list'])

            # Get the feature importance from the trained classifier and write to a .csv file and as a bar plot as .png file
            featureImportancesToAdd = trainedClassifer.explain().get('importance').getInfo()
            featureImportancesToAdd = pd.DataFrame(featureImportancesToAdd.items(),
                                                columns=['Variable', 'Feature_Importance']).sort_values(by='Feature_Importance',
                                                                                                        ascending=False)
            # Scale values
            featureImportancesToAdd['Feature_Importance'] = featureImportancesToAdd['Feature_Importance'] - featureImportancesToAdd['Feature_Importance'].min()
            featureImportancesToAdd['Feature_Importance'] = featureImportancesToAdd['Feature_Importance'] / featureImportancesToAdd['Feature_Importance'].max()

            featureImportances = pd.concat([featureImportances, featureImportancesToAdd])

        featureImportances = pd.DataFrame(featureImportances.groupby('Variable').mean().to_records())

    # Write to csv
    featureImportances.sort_values('Feature_Importance', ascending = False ,inplace = True)
    featureImportances.to_csv('output/'+PIPELINE_PARAMS['today']+'_'+PIPELINE_PARAMS['model_params']['response_var']+'_featureImportances.csv', index=False)

    # Create and save plot
    plt = featureImportances[:10].plot(x='Variable', y='Feature_Importance', kind='bar', legend=False,
                                    title='Feature Importances')
    fig = plt.get_figure()
    fig.savefig('output/'+PIPELINE_PARAMS['today']+'_'+PIPELINE_PARAMS['model_params']['response_var']+'_featureImportances.png', bbox_inches='tight')

    print('Variable importance metrics complete! Moving on...')

def bootstrapping(classifierList, bestModelName):
    try:
        # Path to bootstrapped samples
        bootstrapFc = ee.FeatureCollection(PIPELINE_PARAMS['general_params']['gee_project_folder']+PIPELINE_PARAMS['general_params']['bootstrap_filename'])
        print(bootstrapFc.size().getInfo())

    except Exception as e:
        training_data = pd.read_csv('data/'+PIPELINE_PARAMS['general_params']['training_data_filename']+'.csv')

        # Input the number of points to use for each bootstrap model: equal to number of observations in training dataset
        bootstrapModelSize = training_data.shape[0]

        # Run a for loop to create multiple bootstrap iterations and upload them to the Google Cloud Storage Bucket
        # Create an empty list to store all of the file name strings being uploaded (for later use)
        # fileNameList = []
        stratSample = training_data.head(0)

        for n in list(range(1, PIPELINE_PARAMS['general_params']['num_bootstrap_samples']+1)):
            # Perform the subsetting
            sampleToConcat = training_data.groupby(PIPELINE_PARAMS['cv_params']['stratification_var'], group_keys=False).apply(lambda x: x.sample(n=int(round((PIPELINE_PARAMS['general_params']['stratification_dict'].get(x.name)/100)*bootstrapModelSize)), replace=True, random_state=n))
            sampleToConcat['bootstrapIteration'] = n
            stratSample = pd.concat([stratSample, sampleToConcat])

        # Write the stratified sample to a CSV
        stratSample.to_csv('data/'+PIPELINE_PARAMS['general_params']['bootstrap_filename']+'.csv', index=False)

        upload_table(PIPELINE_PARAMS['general_params']['bootstrap_filename'])

    # Load the best model from the classifier list
    classifierToBootstrap = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierList).filterMetadata('cName','equals',bestModelName).first()).get('c'))

    # Create empty list to store all fcs
    fcList = []
    # Run a for loop to create multiple bootstrap iterations
    for n in list(range(1, PIPELINE_PARAMS['general_params']['num_bootstrap_samples']+1)):
        # Load the collection from the path
        fcToTrain = ee.FeatureCollection(PIPELINE_PARAMS['general_params']['gee_project_folder']+PIPELINE_PARAMS['general_params']['bootstrap_filename']).filter(ee.Filter.eq('bootstrapIteration', n))

        # Append fc to list
        fcList.append(fcToTrain)

    # Helper fucntion to train a RF classifier and classify the composite image
    def bootstrapFunc(fc):
        # Train the classifier with the collection
        trainedClassifer = classifierToBootstrap.train(fc, PIPELINE_PARAMS['model_params']['response_var'], PIPELINE_PARAMS['model_params']['covariate_list'])

        # Classify the image
        classifiedImage = eval(PIPELINE_PARAMS['model_params']['composite']).classify(trainedClassifer,PIPELINE_PARAMS['model_params']['response_var']+'_predicted')

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
    coefOfVarImage = stdDevImage.divide(meanImage)

    return ee.Image.cat(meanImage, upperLowerCIImage, stdDevImage, coefOfVarImage).rename(['mean_image', 'lower_ci', 'upper_ci', 'std_dev', 'coef_var'])

def interpolation_extrapolation():
    training_data = pd.read_csv('data/'+PIPELINE_PARAMS['general_params']['training_data_filename']+'.csv')

    ##################################################################################################################################################################
    # Univariate int-ext analysis
    ##################################################################################################################################################################
    fcOI = ee.FeatureCollection(PIPELINE_PARAMS['general_params']['gee_project_folder']+PIPELINE_PARAMS['general_params']['training_data_filename'])
    
    # Create a feature collection with only the values from the image bands
    fcForMinMax = fcOI.select(PIPELINE_PARAMS['model_params']['covariate_list'])

    # Make a FC with the band names
    fcWithBandNames = ee.FeatureCollection(ee.List(PIPELINE_PARAMS['model_params']['covariate_list']).map(lambda bandName: ee.Feature(None).set('BandName',bandName)))

    def calcMinMax(f):
        bandBeingComputed = f.get('BandName')
        maxValueToSet = fcForMinMax.reduceColumns(ee.Reducer.minMax(),[bandBeingComputed])
        return f.set('MinValue',maxValueToSet.get('min')).set('MaxValue',maxValueToSet.get('max'))

    # Map function
    fcWithMinMaxValues = ee.FeatureCollection(fcWithBandNames).map(calcMinMax)

    # Make two images from these values (a min and a max image)
    maxValuesWNulls = fcWithMinMaxValues.toList(1000).map(lambda f: ee.Feature(f).get('MaxValue'))
    maxDict = ee.Dictionary.fromLists(PIPELINE_PARAMS['model_params']['covariate_list'],maxValuesWNulls)
    minValuesWNulls = fcWithMinMaxValues.toList(1000).map(lambda f: ee.Feature(f).get('MinValue'))
    minDict = ee.Dictionary.fromLists(PIPELINE_PARAMS['model_params']['covariate_list'],minValuesWNulls)
    minImage = minDict.toImage()
    maxImage = maxDict.toImage()

    totalBandsBinary = eval(PIPELINE_PARAMS['model_params']['composite']).select(PIPELINE_PARAMS['model_params']['covariate_list']).gte(minImage.select(PIPELINE_PARAMS['model_params']['covariate_list'])).lt(maxImage.select(PIPELINE_PARAMS['model_params']['covariate_list']))
    univariate_int_ext_image = totalBandsBinary.reduce('sum').divide(eval(PIPELINE_PARAMS['model_params']['composite']).bandNames().length())

    ##################################################################################################################################################################
    # Multivariate (PCA) int-ext analysis
    ##################################################################################################################################################################

    # Input the proportion of variance that you would like to cover
    propOfVariance = 90

    # PCA interpolation/extrapolation helper function
    def assessExtrapolation(training_data, propOfVariance):
        # Compute the mean and standard deviation of each band, then standardize the point data
        meanVector = training_data.mean()
        stdVector = training_data.std()
        standardizedData = (training_data-meanVector)/stdVector

        # Then standardize the composite from which the points were sampled
        meanList = meanVector.tolist()
        stdList = stdVector.tolist()
        bandNames = list(meanVector.index)
        meanImage = ee.Image(meanList).rename(bandNames)
        stdImage = ee.Image(stdList).rename(bandNames)
        standardizedImage = eval(PIPELINE_PARAMS['model_params']['composite']).select(PIPELINE_PARAMS['model_params']['covariate_list']).subtract(meanImage).divide(stdImage)

        # Run a PCA on the point samples
        pcaOutput = PCA()
        pcaOutput.fit(standardizedData)

        # Save the cumulative variance represented by each PC
        cumulativeVariance = np.cumsum(np.round(pcaOutput.explained_variance_ratio_, decimals=4)*100)

        # Make a list of PC names for future organizational purposes
        pcNames = ['PC'+str(x) for x in range(1,training_data.shape[1]+1)]

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
    PCA_int_ext = assessExtrapolation(training_data[PIPELINE_PARAMS['model_params']['covariate_list']], propOfVariance)
    
    return ee.Image.cat(univariate_int_ext_image, PCA_int_ext).rename(['univariate_pct_int_ext', 'PCA_pct_int_ext'])


# ##################################################################################################################################################################
# # Spatial Leave-One-Out cross validation
# ##################################################################################################################################################################
# def sloo_cv():
#     ssetIDToCreate_Folder = 'projects/crowtherlab/johan/SPUN/EM_sloo_cv'
#     if any(x in subprocess.run(bashCommandList_Detect+[assetIDToCreate_Folder],stdout=subprocess.PIPE).stdout.decode('utf-8') for x in stringsOfInterest) == False:
#         pass
#     else:
#         # perform the folder creation
#         print(assetIDToCreate_Folder,'being created...')

#         # Create the folder within Earth Engine
#         subprocess.run(bashCommandList_CreateFolder+[assetIDToCreate_Folder])
#         while any(x in subprocess.run(bashCommandList_Detect+[assetIDToCreate_Folder],stdout=subprocess.PIPE).stdout.decode('utf-8') for x in stringsOfInterest):
#             print('Waiting for asset to be created...')
#             time.sleep(normalWaitTime)
#         print('Asset created!')

#         # Sleep to allow the server time to receive incoming requests
#         time.sleep(normalWaitTime/2)


#     # !! NOTE: this is a fairly computatinally intensive excercise, so there are some precautions to take to ensure servers aren't overloaded
#     # !! This operaion SHOULD NOT be performed on the entire dataset

#     # Define buffer sizes to test (in meters)
#     buffer_sizes = [1000, 2500, 5000, 10000, 50000, 100000, 250000, 500000, 750000, 1000000, 1500000, 2000000]

#     # # Set number of random points to test
#     # if preppedCollection.shape[0] > 1000:
#     #     n_points = 1000 # Don't increase this value!
#     # else:
#     #     n_points = preppedCollection.shape[0]
#     n_points = 1000

#     # Set number of repetitions
#     n_reps = 10
#     nList = list(range(0,n_reps))

#     # Perform BLOO-CV
#     for rep in nList:
#         for buffer in buffer_sizes:
#             # mapList = []
#             # for item in nList:
#             #     mapList = mapList + (list(zip([buffer], repeat(item))))

#             # Make a feature collection from the buffer sizes list
#             # fc_toMap = ee.FeatureCollection(ee.List(mapList).map(lambda n: ee.Feature(ee.Geometry.Point([0,0])).set('buffer_size',ee.List(n).get(0)).set('rep',ee.List(n).get(1))))
#             fc_toMap = ee.FeatureCollection(ee.Feature(ee.Geometry.Point([0,0])).set('buffer_size',buffer).set('rep',rep))

#             grid_search_resultsRegression = ee.FeatureCollection('projects/johanvandenhoogen/assets'+'/'+projectFolder+'/'+classProperty+'_Regression_grid_search_results')

#             # Get top model name
#             bestModelNameRegression = grid_search_resultsRegression.limit(1, 'Mean_R2', False).first().get('cName')

#             # Get top 10 models
#             top_10ModelsRegression = grid_search_resultsRegression.limit(10, 'Mean_R2', False).aggregate_array('cName')

#             # Helper function 1: assess whether point is within sampled range
#             def WithinRange(f):
#                 testFeature = f
#                 # Training FeatureCollection: all samples not within geometry of test feature
#                 trainFC = fcOI.filter(ee.Filter.geometry(f.geometry()).Not())

#                 # Make a FC with the band names
#                 fcWithBandNames = ee.FeatureCollection(ee.List(covariateList).map(lambda bandName: ee.Feature(None).set('BandName',bandName)))

#                 # Helper function 1b: assess whether training point is within sampled range; per band
#                 def getRange(f):
#                     bandBeingComputed = f.get('BandName')
#                     minValue = trainFC.aggregate_min(bandBeingComputed)
#                     maxValue = trainFC.aggregate_max(bandBeingComputed)
#                     testFeatureWithinRange = ee.Number(testFeature.get(bandBeingComputed)).gte(ee.Number(minValue)).bitwiseAnd(ee.Number(testFeature.get(bandBeingComputed)).lte(ee.Number(maxValue)))
#                     return f.set('within_range', testFeatureWithinRange)

#                 # Return value of 1 if all bands are within sampled range
#                 within_range = fcWithBandNames.map(getRange).aggregate_min('within_range')

#                 return f.set('within_range', within_range)

#             # Helper function 1: Spatial Leave One Out cross-validation function:
#             def BLOOcv(f):
#                 rep = f.get('rep')
#                 # Test feature
#                 testFeature = ee.FeatureCollection(f)

#                 # Training set: all samples not within geometry of test feature
#                 trainFC = fcOI.filter(ee.Filter.neq(classProperty, 0)).filter(ee.Filter.geometry(testFeature).Not())

#                 # Classifier to test: same hyperparameter settings as from grid search procedure
#                 classifierName = top_10ModelsRegression.get(rep)
#                 classifier = ee.Classifier(ee.Feature(ee.FeatureCollection(classifierListRegression).filterMetadata('cName', 'equals', classifierName).first()).get('c'))

#                 # Train classifier
#                 trainedClassifer = classifier.train(trainFC, classProperty, PIPELINE_PARAMS['model_params']['covariate_list'])

#                 # Apply classifier
#                 classified = testFeature.classify(classifier = trainedClassifer, outputName = 'predicted')

#                 # Get predicted value
#                 predicted = classified.first().get('predicted')

#                 # Set predicted value to feature
#                 return f.set('predicted', predicted).copyProperties(f)

#             # Helper function 2: R2 calculation function
#             def calc_R2(f):
#                 rep = f.get('rep')
#                 # FeatureCollection holding the buffer radius
#                 buffer_size = f.get('buffer_size')

#                 # Sample 1000 validation points from the data
#                 fc_withRandom = fcOI.filter(ee.Filter.neq(classProperty, 0)).randomColumn(seed = rep)
#                 subsetData = fc_withRandom.sort('random').limit(n_points)

#                 # Add the iteration ID to the FC
#                 fc_toValidate = subsetData.map(lambda f: f.set('rep', rep))

#                 # Add the buffer around the validation data
#                 fc_wBuffer = fc_toValidate.map(lambda f: f.buffer(buffer_size))

#                 # Remove points not within sampled range
#                 fc_withinSampledRange = fc_wBuffer.map(WithinRange).filter(ee.Filter.eq('within_range', 1))

#                 # Apply blocked leave one out CV function
#                 # predicted = fc_withinSampledRange.map(BLOOcv)
#                 predicted = fc_wBuffer.map(BLOOcv)

#                 # Calculate R2 value
#                 R2_val = coefficientOfDetermination(predicted, classProperty, 'predicted')

#                 return(f.set('R2_val', R2_val))

#             # Calculate R2 across range of buffer sizes
#             sloo_cv = fc_toMap.map(calc_R2)

#             # Export FC to assets
#             bloo_cv_fc_export = ee.batch.Export.table.toAsset(
#                 collection = sloo_cv,
#                 description = classProperty+'_sloo_cv_results_woExtrapolation_'+str(buffer),
#                 assetId = 'projects/crowtherlab/johan/SPUN/EM_sloo_cv/'+classProperty+'_sloo_cv_results_wExtrapolation_'+str(buffer)+'_rep_'+str(rep),
#             )

#             # bloo_cv_fc_export.start()

#         print('Blocked Leave-One-Out started! Moving on...')
    