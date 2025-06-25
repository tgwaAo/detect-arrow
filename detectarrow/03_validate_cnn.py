#!/bin/env python3

from processing.model_handler import ModelHandler

if __name__ == '__main__':
    model_handler = ModelHandler()
    model_handler.load_dataset()
    model_handler.load_model()
    model_handler.prepare_validation()
    model_handler.classification_report()
    model_handler.saliency(import_hack=True)
