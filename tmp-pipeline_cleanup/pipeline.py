from config.config import *

from functions.determineBlockSizeForCV import *
from functions.helper_functions import *
from functions.cv_functions import *
from functions.pipeline_functions import *
from functions.gee_hyperparameter_tuning import *

PIPELINE_PARAMS = get_config(response_var = 'arbuscular_mycorrhizal_richness')

# Initialize ee API
if PIPELINE_PARAMS['cloud_params']['use_service_account'] == True:
    credentials = ee.ServiceAccountCredentials(PIPELINE_PARAMS['cloud_params']['service_account'], 
                                               PIPELINE_PARAMS['cloud_params']['service_account_key'])
    ee.Initialize(credentials, project = PIPELINE_PARAMS['cloud_params']['cloud_project'])
else:
    ee.Initialize()

# Composite of interest
composite = ee.Image.cat([
        ee.Image("projects/crowtherlab/Composite/CrowtherLab_bioComposite_30ArcSec"),
        ee.Image("projects/crowtherlab/Composite/CrowtherLab_climateComposite_30ArcSec"),
        ee.Image("projects/crowtherlab/Composite/CrowtherLab_geoComposite_30ArcSec"),
        ee.Image("projects/crowtherlab/Composite/CrowtherLab_processComposite_30ArcSec"),
        ])


# Perform the necessary folder initialization
gee_folder_initialization()

# Data processing
data_processing()

# Hyperparameter tuning
gee_hyperparameter_tuning()

# Model training and prediction
predicted_image = image_classification()

bootrapped_mean_image = bootstrapping().select('mean_image')
bootstrapped_lowerupperci_image = bootstrapping().select(['lower_ci', 'upper_ci'])
bootstrapped_std_dev_image = bootstrapping().select('std_dev')
bootstrapped_coefvar = bootstrapping().select('coef_var')

univariate_int_ext_image = interpolation_extrapolation().select('univariate_pct_int_ext')
pca_int_ext = interpolation_extrapolation().select('PCA_pct_int_ext')

if PIPELINE_PARAMS['model_params']['log_transform_response_var'] == True:
    img_to_export = ee.Image.cat(
        predicted_image.exp().subtract(1),
        bootrapped_mean_image.exp().subtract(1),
        bootstrapped_lowerupperci_image.select(0).exp().subtract(1),
        bootstrapped_lowerupperci_image.select(1).exp().subtract(1),
        bootstrapped_std_dev_image.exp().subtract(1),
        bootstrapped_coefvar.exp().subtract(1),
        univariate_int_ext_image,
        pca_int_ext)
if PIPELINE_PARAMS['model_params']['log_transform_response_var'] == False:
    img_to_export = ee.Image.cat(
        predicted_image,
        bootrapped_mean_image,
        bootstrapped_lowerupperci_image.select(0),
        bootstrapped_lowerupperci_image.select(1),
        bootstrapped_std_dev_image,
        bootstrapped_coefvar,
        univariate_int_ext_image,
        pca_int_ext)
    
# Rename bands
img_to_export = img_to_export.rename([PIPELINE_PARAMS['general_params']['response_var']+'_ensemble_mean',
                                        PIPELINE_PARAMS['general_params']['response_var']+'_bootstrapped_mean',
                                        PIPELINE_PARAMS['general_params']['response_var']+'_bootstrapped_lower',
                                        PIPELINE_PARAMS['general_params']['response_var']+'_bootstrapped_upper',
                                        PIPELINE_PARAMS['general_params']['response_var']+'_bootstrapped_stddev',
                                        PIPELINE_PARAMS['general_params']['response_var']+'_bootstrapped_coefvar',
                                        PIPELINE_PARAMS['general_params']['response_var']+'_univariate_pct_int_ext',
                                        PIPELINE_PARAMS['general_params']['response_var']+'_multivariate_pct_int_ext'])

# Export the image
image_export = ee.batch.Export.image.toAsset(
    image = img_to_export.toFloat(),
    description = PIPELINE_PARAMS['today']+'_'+PIPELINE_PARAMS['general_params']['response_var']+'_predicted',
    assetId = PIPELINE_PARAMS['general_params']['gee_project_folder']+'/'+PIPELINE_PARAMS['today']+'_'+PIPELINE_PARAMS['general_params']['response_var']+'_predicted',
    crs = PIPELINE_PARAMS['export_params']['crs'],
    crsTransform = PIPELINE_PARAMS['export_params']['crs_transform'],
    region = eval(PIPELINE_PARAMS['export_params']['export_geometry']),
    maxPixels = int(1e13),
    pyramidingPolicy = {".default": PIPELINE_PARAMS['export_params']['pyramiding_policy']}
)
image_export.start()

