# Linear regression on the Boston house prices dataset


# Quick Start

## Clone the repository

```console
$ git clone git@github.com:caron14/BayesianOptimization.git 
```

## Setting up the Environment using Docker

The docker image can be created from Dockerfile easily.
You can easily build the same environment by specifying the python image and the version of the external library in the Dockerfile.

The contents of Dockerfile are below.
```Dockerfile
FROM python:3.9.2

WORKDIR /opt
RUN pip install --upgrade pip
RUN pip install numpy==1.20.1 \
				pandas==1.2.3 \
				matplotlib==3.3.4 \
				seaborn==0.11.1 \
				scikit-learn==0.24.1 

WORKDIR /work
```

To create the docker image, execute the following command in the directory where the Dockerfile exists.

```console
$ docker build .
```

Build the docker container from the docker image. The "-v" option is used to synchronize the local directory with the directory in the docker container.

```console
$ docker run -it -v ~/**/LinearRegression_BostonHousePrices:/work <Image ID> bash
```

It should be noted here that you can confirm "<Image ID>" by the following docker command.

```console
$ docker images
```

## Execution

```console
$ python linear-regression.py
```

## Dependencies
* Numpy
* Pandas
* Scikit-learn
* Matplotlib
* seaborn