#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT OR CC-BY-4.0

import argparse
import itertools
import json
import multiprocessing
import sys
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import numpy as np
import pyximport
import scipy.io as spio

from broad_utils import quatFromAccMag, quatmult, calculateRMSE, loadResults, getAveragedRmseValues

pyximport.install()
# The orientation estimation algorithms are implemented in C++ and will automatically be compiled using pyximport and
# the .pyxbld file. This will require a C++ compiler to be available.
from madgwick_mahony import MadgwickAHRS, MahonyAHRS


def runMadgwick(data, beta):
    acc = data['imu_acc']
    gyr = data['imu_gyr']
    mag = data['imu_mag']
    sampling_rate = data['sampling_rate']

    obj = MadgwickAHRS(beta, sampling_rate)
    obj.setState(quatFromAccMag(acc[0], mag[0]))  # set initial state based on the first sample
    quat = obj.updateBatch(gyr, acc, mag)  # run the orientation estimation algorithm
    quat = quatmult(np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], float), quat)  # make sure earth frame is ENU
    return quat


def runMahony(data, Kp, Ki):
    acc = data['imu_acc']
    gyr = data['imu_gyr']
    mag = data['imu_mag']
    sampling_rate = data['sampling_rate']

    obj = MahonyAHRS(Kp, Ki, sampling_rate)
    obj.setState(quatFromAccMag(acc[0], mag[0]))  # set initial state based on the first sample
    quat, _ = obj.updateBatch(gyr, acc, mag)  # run the orientation estimation algorithm
    quat = quatmult(np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], float), quat)  # make sure earth frame is ENU
    return quat


def gridProcess(trialName, dataPath, fn, paramGrid):
    """Processes one trial with one single orientation estimation algorithm and a grid of parameters."""

    # load the data
    data = spio.loadmat(dataPath / f'{trialName}.mat')
    for k in ('imu_gyr', 'imu_acc', 'imu_mag'):  # ensure the IMU data is stored in C order
        data[k] = np.ascontiguousarray(data[k])
    data['movement'] = data['movement'].squeeze().astype(bool)  # convert to 1D boolean so it can be used for indexing

    paramNames = list(paramGrid.keys())  # list of parameter names of the grid to search
    paramValues = list(paramGrid.values())  # list of parameter values of the grid to search

    shape = [len(p) for p in paramValues]  # shape of the result matrices
    out = dict(total_rmse_deg=np.zeros(shape), heading_rmse_deg=np.zeros(shape), inclination_rmse_deg=np.zeros(shape))

    print('starting', 'x'.join(str(n) for n in shape) + '=' + str(np.prod(shape)), 'grid search for', trialName)
    sys.stdout.flush()

    # loop over the grid of parameter values
    for params in itertools.product(*[enumerate(p) for p in paramValues]):
        ind = tuple(p[0] for p in params)  # index of the current parameter combination in the output matrix
        p = dict(zip(paramNames, [p[1] for p in params]))  # parameter dictionary containing the current combination

        # run the orientation estimation algorithm
        imu_quat = fn(data, **p)

        # calculate total/heading/inclination RMSE during the motion phase
        errors = calculateRMSE(imu_quat, data['opt_quat'], data['movement'])
        for k in out:
            out[k][ind] = errors[k]

    return trialName, out


def run(dataPath, outPath, trials, name, fn, params, workerCount, force):
    """
    Processes all trials with one single orientation estimation algorithm for a grid of parameters.

    The work is split by trial and run in mutliple processes (if workerCount > 1).
    """
    outFilename = outPath / f'results_{name}.mat'
    if not force and outFilename.exists():
        print(f'skipping grid search for {name} (results_{name}.mat already exists, use -f to overwrite)')
        return

    workerFn = partial(gridProcess, dataPath=dataPath, fn=fn, paramGrid=params)
    startTime = time.time()
    results = dict(total_rmse_deg={}, heading_rmse_deg={}, inclination_rmse_deg={})

    with multiprocessing.Pool(workerCount) if workerCount != 1 else nullcontext() as pool:
        # start processing
        if workerCount == 1:  # do not use multiprocessing to facilitate debugging/profiling
            iterator = map(fn, trials.keys())
        else:
            iterator = pool.imap_unordered(workerFn, trials.keys(), chunksize=1)

        # collect results
        for i, (trialName, result) in enumerate(iterator, start=1):
            print('done: {} of {}, {} %'.format(i, len(trials), int(round(100 * i / len(trials)))))
            sys.stdout.flush()

            for k in results:
                assert len('trial_' + trialName) <= 63  # maximum field name length in .mat
                results[k]['trial_' + trialName] = result[k]

    duration = time.time() - startTime
    assert i == len(trials)

    vals = np.prod([len(v) for v in params.values()])
    print(f'tested {vals} parameter values for {len(trials)} trials in {duration:.0f} s, '
          f'{1000 * duration / vals:.0f} ms/val, {1000 * duration / vals / len(trials):.2f} ms/val/trial')

    results['param_names'] = np.array(list(params.keys()), object)
    results['param_values'] = params
    spio.savemat(outFilename, results, long_field_names=True, do_compression=True, oned_as='column')
    print(f'full results written to {outFilename}')


def createAverageRmseJson(outPath, trialInfo, name):
    """Writes averaged RMSE values for each trial and each group of trials to a json file."""
    results = loadResults(outPath / f'results_{name}.mat')
    outFilename = outPath / f'results_{name}_average_rmse.json'
    avgRmse = getAveragedRmseValues(trialInfo, results)
    with open(outFilename, 'w') as f:
        json.dump(avgRmse, f, indent=2)
    print(f'averaged RMSE values written to {outFilename}')


def main():
    parser = argparse.ArgumentParser(description='Runs orientation estimation algorithms on the BROAD dataset with '
                                                 'linearly spaced parameters.')
    parser.add_argument('-f', '--force', action='store_true', help='overwrite existing files')
    parser.add_argument('-j', '--jobs', action='store_true', default=multiprocessing.cpu_count(),
                        help='number of worker processes to use, default: %(default)s (cpu count)')
    args = parser.parse_args()

    basePath = Path(__file__).resolve().parent
    dataPath = basePath / '..' / 'data_mat'
    outPath = basePath / 'out'
    outPath.mkdir(exist_ok=True)

    with open(dataPath / 'trials.json') as f:
        trialInfo = json.load(f)

    madgwickParams = {
        'beta': np.round(np.arange(0.01, 0.300001, 0.01), 8),
    }
    run(dataPath, outPath, trialInfo['trials'], 'madgwick', runMadgwick, madgwickParams, args.jobs, args.force)
    createAverageRmseJson(outPath, trialInfo, 'madgwick')

    mahonyParams = {
        'Kp': np.round(np.arange(0.02, 2.000001, 0.02), 8),
        'Ki': np.round(np.arange(0.0, 0.0040001, 0.0001), 8),
    }
    run(dataPath, outPath, trialInfo['trials'], 'mahony', runMahony, mahonyParams, args.jobs, args.force)
    createAverageRmseJson(outPath, trialInfo, 'mahony')


if __name__ == '__main__':
    main()
