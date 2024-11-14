# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT OR CC-BY-4.0

import numpy as np
import scipy.io as spio


def calculateErrorQuatEarth(imu_quat, opt_quat):
    """
    Calculates quaternion that represents the orientation estimation error in the global coordinate system.

    :param imu_quat: IMU orientation, shape (N, 4)
    :param opt_quat: OMC orientation, shape (N, 4)
    :return: error quaternion, shape (N, 4)
    """
    # normalize the input quaternions just in case
    imu_quat = imu_quat / np.linalg.norm(imu_quat, axis=1)[:, None]
    opt_quat = opt_quat / np.linalg.norm(opt_quat, axis=1)[:, None]
    # calculate the relative orientation expressed in the global coordinate system
    # imu_quat * (inv(opt_quat) * imu_quat) * inv(imu_quat) = imu_quat * inv(opt_quat)
    out = quatmult(imu_quat, invquat(opt_quat))
    # normalize the output quaternion
    out = out / np.linalg.norm(out, axis=1)[:, None]
    return out


def calculateTotalError(q_diff):
    """
    Calculates the total error, i.e. the total absolute rotation angle of the quaternion.

    :param q_diff: error quaternion, shape (N, 4)
    :return: error in rad, shape (N,)
    """
    return 2 * np.arccos(np.clip(np.abs(q_diff[:, 0]), 0, 1))


def calculateHeadingError(q_diff_earth):
    """
    Calculates the heading error.

    :param q_diff_earth: error quaternion in global coordinates (c.f. calculateErrorQuatEarth), shape (N, 4)
    :return: error in rad, shape (N,)
    """
    return 2 * np.arctan(np.abs(q_diff_earth[:, 3] / q_diff_earth[:, 0]))


def calculateInclinationError(q_diff_earth):
    """
    Calculates the inclination error.

    :param q_diff_earth: error quaternion in global coordinates (c.f. calculateErrorQuatEarth), shape (N, 4)
    :return: error in rad, shape (N,)
    """
    return 2 * np.arccos(np.clip(np.sqrt(q_diff_earth[:, 0] ** 2 + q_diff_earth[:, 3] ** 2), 0, 1))


def calculateRMSE(imu_quat, opt_quat, movement):
    """
    Calculates total/heading/inclination errors in degrees (only considering movement phases).

    :param imu_quat: IMU orientation, shape (N, 4)
    :param opt_quat: OMC orientation, shape (N, 4)
    :param movement: boolean indexing array that denotes motion phases, shape (N,)
    :return: dict containing total, heading and inclination errors in degrees
    """
    assert movement.dtype == bool

    q_diff_earth = calculateErrorQuatEarth(imu_quat, opt_quat)

    totalError = calculateTotalError(q_diff_earth)[movement]
    headingError = calculateHeadingError(q_diff_earth)[movement]
    inclError = calculateInclinationError(q_diff_earth)[movement]

    return dict(
        total_rmse_deg=np.rad2deg(rmse(totalError)),
        heading_rmse_deg=np.rad2deg(rmse(headingError)),
        inclination_rmse_deg=np.rad2deg(rmse(inclError))
    )


def quatmult(q1, q2):
    """
    Quaternion multiplication.

    If two Nx4 arrays are given, they are multiplied row-wise. Alternative one of the inputs can be a single
    quaternion which is then multiplied to all rows of the other input array.
    """

    q1 = np.asarray(q1, float)
    q2 = np.asarray(q2, float)

    # if both input quaternions are 1D arrays, we also want to return a 1D output
    is1D = max(len(q1.shape), len(q2.shape)) < 2

    # but to be able to use the same indexing in all cases, make sure everything is in 2D arrays
    if q1.shape == (4,):
        q1 = q1.reshape((1, 4))
    if q2.shape == (4,):
        q2 = q2.reshape((1, 4))

    # check the dimensions
    N = max(q1.shape[0], q2.shape[0])
    assert q1.shape == (N, 4) or q1.shape == (1, 4)
    assert q2.shape == (N, 4) or q2.shape == (1, 4)

    # actual quaternion multiplication
    q3 = np.zeros((N, 4), float)
    q3[:, 0] = q1[:, 0] * q2[:, 0] - q1[:, 1] * q2[:, 1] - q1[:, 2] * q2[:, 2] - q1[:, 3] * q2[:, 3]
    q3[:, 1] = q1[:, 0] * q2[:, 1] + q1[:, 1] * q2[:, 0] + q1[:, 2] * q2[:, 3] - q1[:, 3] * q2[:, 2]
    q3[:, 2] = q1[:, 0] * q2[:, 2] - q1[:, 1] * q2[:, 3] + q1[:, 2] * q2[:, 0] + q1[:, 3] * q2[:, 1]
    q3[:, 3] = q1[:, 0] * q2[:, 3] + q1[:, 1] * q2[:, 2] - q1[:, 2] * q2[:, 1] + q1[:, 3] * q2[:, 0]

    if is1D:
        q3 = q3.reshape((4,))

    return q3


def invquat(q):
    """Calculates the inverse of unit quaternions."""

    q = np.asarray(q, float)
    if len(q.shape) != 2:
        assert q.shape == (4,)
        qConj = q.copy()
        qConj[1:] *= -1
        return qConj
    else:
        assert q.shape[1] == 4
        qConj = q.copy()
        qConj[:, 1:] *= -1
        return qConj


def quatFromRotMat(R):
    """Gets a quaternion from a rotation matrix."""
    assert R.shape == (3, 3)

    w_sq = (1 + R[0, 0] + R[1, 1] + R[2, 2]) / 4
    x_sq = (1 + R[0, 0] - R[1, 1] - R[2, 2]) / 4
    y_sq = (1 - R[0, 0] + R[1, 1] - R[2, 2]) / 4
    z_sq = (1 - R[0, 0] - R[1, 1] + R[2, 2]) / 4

    q = np.zeros((4,), float)
    q[0] = np.sqrt(w_sq)
    q[1] = np.copysign(np.sqrt(x_sq), R[2, 1] - R[1, 2])
    q[2] = np.copysign(np.sqrt(y_sq), R[0, 2] - R[2, 0])
    q[3] = np.copysign(np.sqrt(z_sq), R[1, 0] - R[0, 1])
    return q


def quatFromAccMag(acc, mag):
    """Calculates an initial orientation from a accelerometer and magnetomter sample."""
    assert acc.shape == (3,)
    assert mag.shape == (3,)
    z = acc
    x = np.cross(np.cross(z, -mag), z)
    y = np.cross(z, x)
    R = np.column_stack([x/np.linalg.norm(x), y/np.linalg.norm(y), z/np.linalg.norm(z)])
    return quatFromRotMat(R)


def rmse(diff):
    """Calculates the RMS of the input signal."""
    return np.sqrt(np.nanmean(diff**2))


def loadResults(filename):
    """Loads result files created by process_data.py"""
    results = spio.loadmat(filename, squeeze_me=True)
    # recover params dictionary
    paramNames = results['param_names']
    if not isinstance(paramNames, np.ndarray):
        paramNames = [paramNames]  # undo squeeze if there is only one parameter
    results['params'] = {p: results['param_values'][p].item() for p in paramNames}
    return results


def getAveragedRmseValues(trialInfo, results):
    """
    Determines averaged RMSE values for each trial and for each group of trials.

    :param trialInfo: trial and group information (loaded from trials.json)
    :param results: result data structure (see loadResults)
    :return: nested dict with averaged rmse results
    """

    metrics = ('total_rmse_deg', 'heading_rmse_deg', 'inclination_rmse_deg')

    # determine parameter values that minimize the average total RMSE over all trials (i.e. the TAGP)
    tagpParams, tagpParamInd = getTagpParams(results)

    # determine RMSE values by group (with TAGP parameters and the minimum error in the search grid)
    trials = {}
    for trialName in trialInfo['trials']:
        trials[trialName] = dict(
            tagp_parameters={metric: results[metric][f'trial_{trialName}'].item()[tagpParamInd] for metric in metrics},
            minimum_value={metric: np.min(results[metric][f'trial_{trialName}'].item()) for metric in metrics},
        )

    # combine errors for all groups
    groups = {}
    for groupInfo in trialInfo['groups']:
        groupName = groupInfo['name']
        groupTrials = [trialName for trialName, info in trialInfo['trials'].items() if groupName in info['groups']]
        groups[groupName] = dict(tagp_parameters={}, minimum_value={})
        for params in groups[groupName].keys():
            for metric in metrics:
                values = [trials[n][params][metric] for n in groupTrials]
                groups[groupName][params][metric] = np.mean(values)

    return dict(
        tagp_parameters=tagpParams,
        tagp_parameter_ind=tagpParamInd,
        groups=groups,
        trials=trials,
    )


def getTagpParams(results):
    """
    Determines the parameter setting that is associated with the TAGP (i.e. the lowest average error achievable when
    using a common parameter set for all trials).

    :param results: result data structure (see loadResults)
    :return: tagpParams (dict containing the parameter values associated with the TAGP), tagpInd (indexing tuple into
        the cost array)
    """
    # the TAGP is defined by the mean over all 39 trials
    trialNames = [n.lstrip('trial_') for n in sorted(results['total_rmse_deg'].dtype.names)]
    assert len(trialNames) == 39

    cost = getMeanError(results, trialNames, 'total_rmse_deg')
    tagpParamInd = np.unravel_index(np.argmin(cost), cost.shape)
    tagpParamInd = tuple([ind.item() for ind in tagpParamInd])  # convert from np.int64 to regular int
    tagpParams = {paramName: values[ind] for (paramName, values), ind in zip(results['params'].items(), tagpParamInd)}
    return tagpParams, tagpParamInd


def getMeanError(results, trialNames, errorMetric):
    """
    Determines the averaged error matrix for the given metric and the given list of trials.

    :param results: result data structure (see loadResults)
    :param trialNames: list of trial names
    :param errorMetric: error metric to consider
    :return: averaged error matrix
    """
    trialNames = list(trialNames)
    assert len(trialNames) == len(set(trialNames))
    vals = [results[errorMetric][f'trial_{name}'].item() for name in trialNames]
    return np.mean(vals, axis=0)
