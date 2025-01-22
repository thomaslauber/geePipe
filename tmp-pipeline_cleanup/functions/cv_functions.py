import ee
import yaml

from config.config import PIPELINE_PARAMS

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
    # Training dataset
    fcOI = ee.FeatureCollection(PIPELINE_PARAMS['general_params']['gee_project_folder']+PIPELINE_PARAMS['general_params']['training_data_filename'])

    # Pull the classifier from the feature
    classifier = ee.Classifier(featureWithClassifier.get('c'))

    # Get the model type
    modelType = classifier.mode().getInfo()

    # Create a function to map through the fold assignments and compute the overall accuracy
    # for all validation folds
    def computeAccuracyForFold(foldFeature):
        # Organize the training and validation data

        foldNumber = ee.Number(ee.Feature(foldFeature).get('Fold'))
        trainingData_Random = fcOI.filterMetadata(PIPELINE_PARAMS['cv_params']['cv_fold_header_random'],'not_equals',foldNumber)
        validationData_Random = fcOI.filterMetadata(PIPELINE_PARAMS['cv_params']['cv_fold_header_random'],'equals',foldNumber)

        trainingData_Spatial = fcOI.filterMetadata(PIPELINE_PARAMS['cv_params']['cv_fold_header_spatial'],'not_equals',foldNumber)
        validationData_Spatial = fcOI.filterMetadata(PIPELINE_PARAMS['cv_params']['cv_fold_header_spatial'],'equals',foldNumber)

        # Train the classifier and classify the validation dataset
        trainedClassifier_Random = classifier.train(trainingData_Random,PIPELINE_PARAMS['model_params']['response_var'],PIPELINE_PARAMS['general_params']['covariate_list'])
        outputtedPropName_Random = PIPELINE_PARAMS['model_params']['response_var']+'_Predicted_Random'
        classifiedValidationData_Random = validationData_Random.classify(trainedClassifier_Random,outputtedPropName_Random)

        trainedClassifier_Spatial = classifier.train(trainingData_Spatial,PIPELINE_PARAMS['model_params']['response_var'],PIPELINE_PARAMS['general_params']['covariate_list'])
        outputtedPropName_Spatial = PIPELINE_PARAMS['model_params']['response_var']+'_Predicted_Spatial'
        classifiedValidationData_Spatial = validationData_Spatial.classify(trainedClassifier_Spatial,outputtedPropName_Spatial)

        if modelType == 'CLASSIFICATION':
            # Compute the overall accuracy of the classification
            categoricalLevels = fcOI.aggregate_array(PIPELINE_PARAMS['model_params']['response_var']).distinct()#.getInfo()
            errorMatrix_Random = classifiedValidationData_Random.errorMatrix(PIPELINE_PARAMS['model_params']['response_var'],outputtedPropName_Random,categoricalLevels)
            overallAccuracy_Random = ee.Number(errorMatrix_Random.accuracy())

            errorMatrix_Spatial = classifiedValidationData_Spatial.errorMatrix(PIPELINE_PARAMS['model_params']['response_var'],outputtedPropName_Spatial,categoricalLevels)
            overallAccuracy_Spatial = ee.Number(errorMatrix_Spatial.accuracy())
            return foldFeature.set('overallAccuracy_Random',overallAccuracy_Random).set('overallAccuracy_Spatial',overallAccuracy_Spatial)
        
        if modelType == 'REGRESSION':
            # Compute accuracy metrics
            r2ToSet_Random = coefficientOfDetermination(classifiedValidationData_Random,PIPELINE_PARAMS['model_params']['response_var'],outputtedPropName_Random)
            rmseToSet_Random = RMSE(classifiedValidationData_Random,PIPELINE_PARAMS['model_params']['response_var'],outputtedPropName_Random)
            maeToSet_Random = MAE(classifiedValidationData_Random,PIPELINE_PARAMS['model_params']['response_var'],outputtedPropName_Random)

            r2ToSet_Spatial = coefficientOfDetermination(classifiedValidationData_Spatial,PIPELINE_PARAMS['model_params']['response_var'],outputtedPropName_Spatial)
            rmseToSet_Spatial = RMSE(classifiedValidationData_Spatial,PIPELINE_PARAMS['model_params']['response_var'],outputtedPropName_Spatial)
            maeToSet_Spatial = MAE(classifiedValidationData_Spatial,PIPELINE_PARAMS['model_params']['response_var'],outputtedPropName_Spatial)
            return foldFeature.set('R2_Random',r2ToSet_Random).set('RMSE_Random', rmseToSet_Random).set('MAE_Random', maeToSet_Random)\
                                .set('R2_Spatial',r2ToSet_Spatial).set('RMSE_Spatial', rmseToSet_Spatial).set('MAE_Spatial', maeToSet_Spatial)

    # Compute the mean and std dev of the accuracy values of the classifier across all folds
    kFoldAssignmentFC = ee.FeatureCollection(ee.List(list(range(1,PIPELINE_PARAMS['cv_params']['k']+1))).map(lambda n: ee.Feature(ee.Geometry.Point([0,0])).set('Fold',n)))
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
        meanRMSE_Random = ee.Number.sqrt(ee.Number(sumOfRMSEvalsSquared_Random).divide(PIPELINE_PARAMS['cv_params']['k']))
        RMSEvals_Spatial = accuracyFC.aggregate_array('RMSE_Spatial')
        RMSEvalsSquared_Spatial = RMSEvals_Spatial.map(lambda f: ee.Number(f).multiply(f))
        sumOfRMSEvalsSquared_Spatial = RMSEvalsSquared_Spatial.reduce(ee.Reducer.sum())
        meanRMSE_Spatial = ee.Number.sqrt(ee.Number(sumOfRMSEvalsSquared_Spatial).divide(PIPELINE_PARAMS['cv_params']['k']))

        RMSEdiff_Random = accuracyFC.aggregate_array('RMSE_Random').map(lambda f: ee.Number(ee.Number(f).subtract(meanRMSE_Random)).pow(2))
        sumOfRMSEdiff_Random = RMSEdiff_Random.reduce(ee.Reducer.sum())
        sdRMSE_Random = ee.Number.sqrt(ee.Number(sumOfRMSEdiff_Random).divide(PIPELINE_PARAMS['cv_params']['k']))
        RMSEdiff_Spatial = accuracyFC.aggregate_array('RMSE_Spatial').map(lambda f: ee.Number(ee.Number(f).subtract(meanRMSE_Spatial)).pow(2))
        sumOfRMSEdiff_Spatial = RMSEdiff_Spatial.reduce(ee.Reducer.sum())
        sdRMSE_Spatial = ee.Number.sqrt(ee.Number(sumOfRMSEdiff_Spatial).divide(PIPELINE_PARAMS['cv_params']['k']))

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