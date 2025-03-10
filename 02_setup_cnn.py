#!/bin/env python3

from main.processing.model_handler import ModelHandler

if __name__ == '__main__':
    model_handler = ModelHandler()
    model_handler.build_model()
    model_handler.load_datasets()
    model_handler.train_model()
    model_handler.save_model()
    model_handler.show_training_progress()
    model_handler.try_val_sample()
    model_handler.classification_report()
    model_handler.saliency(quick_hack=True)




