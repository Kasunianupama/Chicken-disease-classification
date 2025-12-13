import os
import sys
from chicken_disease_prediction import logger
from chicken_disease_prediction.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from chicken_disease_prediction.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline

def _ensure_package_on_path(pkg_name='chicken_disease_prediction'):
    here = os.path.dirname(os.path.abspath(__file__))
    cur = here
    while True:
        # direct child (project_root/chicken_disease_prediction)
        if os.path.isdir(os.path.join(cur, pkg_name)):
            parent = cur
            if parent not in sys.path:
                sys.path.insert(0, parent)
            return True

        # package inside a common source folder (project_root/src/chicken_disease_prediction)
        src_candidate = os.path.join(cur, "src", pkg_name)
        if os.path.isdir(src_candidate):
            src_parent = os.path.join(cur, "src")
            if src_parent not in sys.path:
                sys.path.insert(0, src_parent)
            return True

        parent_cur = os.path.dirname(cur)
        if parent_cur == cur:
            break
        cur = parent_cur
    return False

import importlib
import importlib.util
import logging

# try to locate the package without using a static import (helps linters/IDE and runtime resolution)
pkg_spec = importlib.util.find_spec('chicken_disease_prediction')
if pkg_spec is None:
    # attempt to add project path and try again
    if _ensure_package_on_path('chicken_disease_prediction'):
        pkg_spec = importlib.util.find_spec('chicken_disease_prediction')

logger = None
if pkg_spec is not None:
    try:
        pkg = importlib.import_module('chicken_disease_prediction')
        logger = getattr(pkg, 'logger', None)
        if logger is None:
            # try submodule logger if package exposes it as a module
            try:
                logger_module = importlib.import_module('chicken_disease_prediction.logger')
                logger = getattr(logger_module, 'logger', None)
            except Exception:
                logger = None
    except Exception:
        logger = None

# fallback to standard logging if package logger not available
if logger is None:
    logger = logging.getLogger('chicken_disease_prediction')
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)

logger.info("This is a log message from main.py")

#from chicken_disease_prediction.pipeline.stage_01_data_ingestion import DataIngestionPipeline   

STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e 

STAGE_NAME = "Prepare Base Model Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    prepare_base_model = PrepareBaseModelPipeline()
    prepare_base_model.main()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

from chicken_disease_prediction.pipeline.stage_03_training import ModelTrainingPipeline

STAGE_NAME = "Training Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    model_trainer = ModelTrainingPipeline()
    model_trainer.main()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e