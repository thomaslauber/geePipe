import ee
import yaml
import datetime

with open("config/pipeline_params.yaml", "r") as file:
    PIPELINE_PARAMS = yaml.safe_load(file)

# this function is used to access the config from anywhere
def get_config(response_var):
    today = datetime.datetime.now().strftime("%Y%m%d")

    # replace placeholders
    PIPELINE_PARAMS['model_params']['response_var'] = PIPELINE_PARAMS['model_params']['response_var'].replace("(RESPONSE_VAR)", response_var)

    PIPELINE_PARAMS['general_params']['training_data_filename'] = PIPELINE_PARAMS['general_params']['training_data_filename'].replace("(DATE)", today)
    PIPELINE_PARAMS['general_params']['bootstrap_filename'] = PIPELINE_PARAMS['general_params']['bootstrap_filename'].replace("(DATE)", today)

    PIPELINE_PARAMS['general_params']['gee_project_folder'] = PIPELINE_PARAMS['general_params']['gee_project_folder'].replace("(RESPONSE_VAR)", response_var)
    PIPELINE_PARAMS['general_params']['training_data_filename'] = PIPELINE_PARAMS['general_params']['training_data_filename'].replace("(RESPONSE_VAR)", response_var)
    PIPELINE_PARAMS['general_params']['bootstrap_filename'] = PIPELINE_PARAMS['general_params']['bootstrap_filename'].replace("(RESPONSE_VAR)", response_var)

    PIPELINE_PARAMS['today'] = today

    # PIPELINE_PARAMS['model_params']['composite'] = composite

    return PIPELINE_PARAMS





