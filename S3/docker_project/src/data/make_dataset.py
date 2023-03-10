# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import torch
import numpy as np


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    Xtrain = np.load(input_filepath + "/train_0.npz")["images"]
    Ytrain = np.load(input_filepath + "/train_0.npz")["labels"]
    for i in range(1,5):
        Xtrain = np.concatenate((Xtrain,np.load(input_filepath + "/train_{}.npz".format(i))["images"]),axis=0)
        Ytrain = np.concatenate((Ytrain,np.load(input_filepath + "/train_{}.npz".format(i))["labels"]),axis=0)

    Xtest = np.load(input_filepath + "/test.npz")["images"]
    Ytest = np.load(input_filepath + "/test.npz")["labels"]

    # normalize images
    for image in Xtrain:
        image -= np.mean(image)
        image /= np.std(image)

    for image in Xtest:
        image -= np.mean(image)
        image /= np.std(image)

    torch.save(torch.from_numpy(Xtrain).float(),output_filepath+"/Xtrain.pt")
    torch.save(torch.from_numpy(Ytrain).long(),output_filepath+"/Ytrain.pt")
    torch.save(torch.from_numpy(Xtest).float(),output_filepath+"/Xtest.pt")
    torch.save(torch.from_numpy(Ytest).long(),output_filepath+"/Ytest.pt")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
