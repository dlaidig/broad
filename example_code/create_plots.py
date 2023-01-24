#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT OR CC-BY-4.0

import json
from pathlib import Path

import matplotlib.transforms
import numpy as np
from matplotlib import pyplot as plt, cm

from broad_utils import loadResults, getAveragedRmseValues, getTagpParams, getMeanError


def paramErrorLinePlot(ax, trialNames, results):
    param = 'beta'
    assert results['params'].keys() == {param}
    vals = results['params'][param]

    err_incl = getMeanError(results, trialNames, 'inclination_rmse_deg')
    err_heading = getMeanError(results, trialNames, 'heading_rmse_deg')
    err_total = getMeanError(results, trialNames, 'total_rmse_deg')

    def mark(x, y, col, offsetX, offsetY, tagp):
        ax.plot(x, y, col+'o', markersize=4)
        ax.plot([0, x], [y, y], col, lw=1)
        ax.plot([x, x], [0, y], col, lw=1)
        text = f'$\\beta={x}$'
        if tagp:
            text = f'$\\mathrm{{TAGP}}={y:.2f}^\\circ$\n' + text
        ax.text(x+offsetX, y+offsetY, text, color=col, ha='left', va='top', size=8)

    def plot(vals, y, col, label, offsetX, offsetY, tagp=False):
        ax.plot(vals, y, col,  label=label, zorder=100)
        mark(vals[np.argmin(y)], np.min(y), col, offsetX, offsetY, tagp)

    plot(vals, err_total, 'C0', '$e$', -0.038, 1.45, tagp=True)
    plot(vals, err_heading, 'C1', '$e_\\mathrm{h}$', -0.02, 0.8)
    plot(vals, err_incl, 'C2', '$e_\\mathrm{i}$', -0.02, 0.8)

    ax.legend()

    ax.grid()
    ax.set_xlabel('algorithm parameter (gain $\\beta$)')
    ax.set_ylabel('RMSE averaged over all trials [°]')
    ax.set_ylim(0, 8)
    ax.set_xlim(min(vals), max(vals))


def contourPlot(ax, trialNames, results):
    paramX = 'Kp'
    paramY = 'Ki'
    minLevel = 0.0
    maxLevel = 12.0

    assert paramX in results['params']
    assert paramY in results['params']
    assert len(results['params']) == 2
    indX = list(results['params'].keys()).index(paramX)
    valX = results['params'][paramX]
    indY = list(results['params'].keys()).index(paramY)
    valY = results['params'][paramY]
    assert indX != indY

    cost = getMeanError(results, trialNames, 'total_rmse_deg')
    tagpParams, tagpParamInd = getTagpParams(results)

    h = ax.contourf(valX, valY, cost.T, levels=np.linspace(minLevel, maxLevel, 101), origin='lower', cmap=cm.jet)
    ax.contour(h, levels=h.levels[6::10], origin='lower', colors='0.5', linewidths=0.5)
    cbar = plt.colorbar(h, ax=ax)
    cbar.set_ticks([0, 2, 4, 6, 8, 10, 12])
    cbar.set_label('RMSE averaged over all trials [°]')

    # manually generate grid at actual search locations
    for v in valX:
        ax.axvline(v, color='k', alpha=0.1, lw=0.2)
    for v in valY:
        ax.axhline(v, color='k', alpha=0.1, lw=0.2)

    ax.plot(tagpParams[paramX], tagpParams[paramY], 'C1o', markersize=4)
    ax.text(tagpParams[paramX]-0.2, tagpParams[paramY]+0.0001,
            f'$\\mathrm{{TAGP}}={np.min(cost):.2f}^\\circ$\n'
            f'$K_\\mathrm{{p}}={tagpParams[paramX]}$\n$K_\\mathrm{{i}}={tagpParams[paramY]}$', size=8)

    ax.set_xlabel('first parameter (gain $K_\\mathrm{p}$)')
    ax.set_ylabel('second parameter (bias est. gain $K_\\mathrm{i}$)')
    ax.set_yticks([0.0, 0.001, 0.002, 0.003, 0.004])


def createParameterErrorPlot(fig, trialInfo, results):
    axes = fig.subplots(1, 2)

    trialNames = trialInfo['trials'].keys()
    paramErrorLinePlot(axes[0], trialNames, results['madgwick'])
    contourPlot(axes[1], trialNames, results['mahony'])

    axes[0].set_title('(a) Algorithm A')
    axes[1].set_title('(b) Algorithm B')

    fig.tight_layout()


def extractTreeInfo(trialInfo):
    y = []
    levels = []
    labels = []

    lastCategory = ''
    for groupInfo in trialInfo['groups']:
        if groupInfo['level'] == 1:
            spacing = 1
        elif groupInfo['level'] == 2 and groupInfo['category'] != lastCategory:
            spacing = 0.3
        else:
            spacing = 0
        lastCategory = groupInfo['category']

        y.append((y[-1] if y else 0) + 0.8 + spacing)
        levels.append(groupInfo['level'])

        label = groupInfo['name'].replace('_', ' ')
        if groupInfo['level'] < 2:
            label = '$\\mathbf{' + label.replace(' ', '~') + '}$'
        labels.append(label)

    return np.array(y, float), np.array(levels, int), labels


def createTree(fig, ax, y, levels, xPos=-16/72, step=6/72, lw=0.75, markersize=2, trans=None):
    if trans is None:
        # creates a transformation that uses inches for the x coordinate and data coordinates for the y axis.
        # as 1 inch equals 1/72 point, with the default parameters, the tree levels are rendered at -16, -10 and -4
        # points, which fits well if pad=20 is passed to ax.tick_params.
        scaled = fig.dpi_scale_trans + matplotlib.transforms.ScaledTranslation(0, 0, ax.transData)
        trans = matplotlib.transforms.blended_transform_factory(scaled, ax.transData)

    ax.plot(xPos, y[0], 'ko', markersize=markersize, clip_on=False, transform=trans)

    downPos = None
    for i in range(1, len(y)):
        if levels[i] <= levels[0]:
            break
        if levels[i] == levels[0] + 1:
            createTree(fig, ax, y[i:], levels[i:], xPos+step, step, lw, markersize, trans)
            downPos = y[i]

    if downPos is not None:
        ax.plot([xPos, xPos], [y[0], downPos], 'k', lw=lw,  clip_on=False, transform=trans)

    if levels[0] != 0:
        ax.plot([xPos-step, xPos], [y[0], y[0]], 'k', lw=lw, clip_on=False, transform=trans)


def createGroupBarPlot(fig, trialInfo, results):
    maxErr = 16  # x axis scaling (in degrees)
    reverse = [9, 13]  # break up axes and reverse the labels at those error values

    axes = fig.subplots(1, 2, sharey=True)

    avgRmseA = getAveragedRmseValues(trialInfo, results['madgwick'])
    avgRmseB = getAveragedRmseValues(trialInfo, results['mahony'])

    # combine error values for all groups into one numpy array for easy plotting
    params = ['tagp_parameters', 'minimum_value']
    metrics = ['total_rmse_deg', 'heading_rmse_deg', 'inclination_rmse_deg']
    resA = {p: {m: np.array([g[p][m] for g in avgRmseA['groups'].values()], float) for m in metrics} for p in params}
    resB = {p: {m: np.array([g[p][m] for g in avgRmseB['groups'].values()], float) for m in metrics} for p in params}

    # axes setup
    y, levels, labels = extractTreeInfo(trialInfo)
    axes[0].set_xlim(0, maxErr)
    axes[1].set_xlim(maxErr, 0)
    for ax, rev, shift in zip(axes, reverse, [0.1, -0.1]):
        pos = np.arange(0, maxErr + shift, 2, int)
        val = [p if p < rev else maxErr - p for p in pos]
        ax.set_xticks(pos)
        ax.set_xticklabels([f'{v:d}' for v in val])
    axes[0].invert_yaxis()
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].tick_params(axis='y', which='major', pad=20)
    axes[0].tick_params(axis='y', which='both', length=0)
    axes[1].tick_params(axis='y', which='both', length=0)
    for i, ax in enumerate(axes):
        ax.grid(axis='x', color='gray', alpha=0.3)
        ax.set_axisbelow(True)

    # strange fix for y axis label alignment in PGF export...
    # https://stackoverflow.com/questions/65243861/matplotlib-python-y-axis-labels-not-aligned-in-pgf-format
    for lab in axes[0].yaxis.get_ticklabels():
        lab.set_verticalalignment('center')

    # broken axis
    for ax, rev in zip(axes, reverse):
        trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.plot((rev, rev), (-0.01, 0.01), transform=trans, color='white', lw=5, clip_on=False, zorder=10)

    # custom horizontal grid
    for y_val, ydiff in zip(y[:-1], np.diff(y)):
        if ydiff < 0.81:
            continue
        for ax in axes:
            ax.axhline(y_val + ydiff / 2, color='gray', alpha=0.3, linewidth=0.5)

    # bar plots
    h = 0.25
    shift = h * 3 / 4 + h / 8
    for ax, res in zip(axes, [resA['tagp_parameters'], resB['tagp_parameters']]):
        ax.barh(y - shift, res['inclination_rmse_deg'], height=h / 2, label='$e_\\mathrm{i}$', color='C2', alpha=0.8)
        ax.barh(y, res['total_rmse_deg'], height=h, label='$e$', color='C0')
        ax.barh(y + shift, res['heading_rmse_deg'], height=h / 2, label='$e_\\mathrm{h}$', color='C1', alpha=0.8)

    # black dots
    for ax, res in zip(axes, [resA['minimum_value'], resB['minimum_value']]):
        for i in range(len(y)):
            ax.plot(res['inclination_rmse_deg'][i], y[i] - shift, 'ok', markersize=1)
            ax.plot(res['total_rmse_deg'][i], y[i], 'ok', markersize=2)
            ax.plot(res['heading_rmse_deg'][i], y[i] + shift, 'ok', markersize=1)

    def stemH(ax, pos, val, lineStyle, lineArgs, markerStyle, markerArgs):
        # will be easier in matplotlib 3.4.0, cf. https://github.com/matplotlib/matplotlib/pull/18187
        for p, v in zip(pos, val):
            ax.plot([maxErr, maxErr - v], [p, p], lineStyle, **lineArgs)
            ax.plot([maxErr - v], [p], markerStyle, **markerArgs)

    # stem comparsion plot
    for metric, col, shiftVal, lineArgs, markerArgs in (
        ['total_rmse_deg', 'C0', 0, dict(lw=2), dict(markersize=4)],
        ['inclination_rmse_deg', 'C2', -shift, dict(lw=1, alpha=0.8), dict(markersize=2, alpha=0.8)],
        ['heading_rmse_deg', 'C1', shift, dict(lw=1, alpha=0.8), dict(markersize=2, alpha=0.8)]):
        valA = resA['tagp_parameters'][metric]
        valB = resB['tagp_parameters'][metric]
        indA = valA < valB
        stemH(axes[0], y[indA] + shiftVal, (valB - valA)[indA], col, lineArgs, 'o' + col, markerArgs)
        stemH(axes[1], y[~indA] + shiftVal, (valA - valB)[~indA], col, lineArgs, 'o' + col, markerArgs)

    # draw tree next to y axis to illustrate nested group structure
    createTree(fig, axes[0], y, levels)

    # add text labels for the main errors
    for i in range(len(y)):
        for ax, res in zip(axes, [resA, resB]):
            if levels[i] > 1:
                continue
            val = res['tagp_parameters']['total_rmse_deg'][i]
            size = 7 if levels[i] > 0 else 9
            text = f'{val:.2f}°'
            if levels[i] == 0:
                text = 'TAGP: ' + text
            ax.text(val + 0.2, y[i], text, ha='left' if ax == axes[0] else 'right', va='center', color='C0', size=size)

            val = res['minimum_value']['total_rmse_deg'][i]
            size = 7 if levels[i] > 0 else 8
            text = f'{val:.2f}°'
            if levels[i] == 0:
                text = 'ITOP: ' + text
            ax.text(val, y[i] - 1.5 * shift, text, ha='left' if ax == axes[0] else 'right', va='bottom', color='k',
                    size=size)

    axes[0].set_title('Algorithm A')
    axes[1].set_title('Algorithm B')
    axes[0].set_xlabel('RMSE averaged over group of trials [°]', x=1)

    legend = axes[1].legend(loc='upper left', fontsize=7)
    for i in 0, 2:  # adjust height of bars in legend
        legend.get_patches()[i].set_height(legend.get_patches()[i].get_height() / 2)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0)

    # use a dummy subplot to show "difference" title in the middle where the stem plots originate
    pos = axes[0].get_position()
    titleax = fig.add_axes([pos.x0 + pos.width, pos.y0, 0, pos.height])
    titleax.set_title('difference')
    titleax.axis('off')  # also removes xlabel
    titleax.xaxis.set_visible(False)


def main():
    basePath = Path(__file__).resolve().parent
    dataPath = basePath / '..' / 'data_mat'
    outPath = basePath / 'out'
    outPath.mkdir(exist_ok=True)

    # load trial information and results
    with open(dataPath / 'trials.json') as f:
        trialInfo = json.load(f)

    results = dict(
        madgwick=loadResults(outPath / 'results_madgwick.mat'),
        mahony=loadResults(outPath / 'results_mahony.mat'),
    )

    # create first plot
    fig = plt.figure(figsize=(6.5, 3))
    createParameterErrorPlot(fig, trialInfo, results)
    outFilename = outPath / 'parameter_error_plot.pdf'
    fig.savefig(outFilename)
    print(f'plot written to {outFilename}')

    # create second plot
    fig = plt.figure(figsize=(8, 5))
    createGroupBarPlot(fig, trialInfo, results)
    outFilename = outPath / 'group_bar_plot.pdf'
    fig.savefig(outFilename)
    print(f'plot written to {outFilename}')

    plt.show()


if __name__ == '__main__':
    main()
