# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent SciPy SMRF stage oracle; optionally runs the actual PDAL CLI too.

Requires numpy==2.2.6 scipy==1.15.3. Pass --pdal /path/to/pdal for differential labels.
Algorithm source: https://github.com/PDAL/PDAL/blob/2.8.4/filters/SMRFilter.cpp
"""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import scipy
from scipy.ndimage import minimum_filter, maximum_filter

assert np.__version__ == '2.2.6' and scipy.__version__ == '1.15.3'
parser = argparse.ArgumentParser()
parser.add_argument('--pdal')
arguments = parser.parse_args()

def diamond(radius):
    vertical, horizontal = np.mgrid[-radius:radius+1,-radius:radius+1]
    return np.abs(vertical) + np.abs(horizontal) <= radius

def erode(surface, radius):
    return minimum_filter(surface, footprint=diamond(radius), mode='nearest')

def dilate(surface, radius):
    return maximum_filter(surface, footprint=diamond(radius), mode='nearest')

def fill(surface):
    result = surface.copy()
    candidates = np.argwhere(np.isfinite(surface))
    if not len(candidates):
        return result
    for row, column in np.argwhere(~np.isfinite(surface)):
        distances = np.sum((candidates - [row,column]) ** 2, axis=1)
        order = candidates[:,1] * surface.shape[0] + candidates[:,0]
        nearest = np.lexsort((order,distances))[:8]
        selected = candidates[nearest]
        result[row,column] = surface[selected[:,0],selected[:,1]].mean()
    return result

def progressive(surface, slope, radius):
    previous = surface.copy()
    erosion = surface.copy()
    objects = np.zeros(surface.shape,dtype=bool)
    for current in range(1,radius+1):
        erosion = erode(erosion,1)
        opening = dilate(erosion,current)
        objects |= np.abs(previous-opening) > slope * current
        previous = opening
    return objects

def make_fixture(name, slope=.15, scalar=1.25, threshold=.5, window=3, cut=0, steepness=0, holes=False):
    size = 11
    minimum = np.full((size,size),np.nan)
    points = []
    for row in range(size):
        for column in range(size):
            if holes and (row,column) in [(2,3),(7,8)]:
                continue
            elevation = steepness * column
            if 4<=row<=6 and 4<=column<=6:
                elevation += 4
            if (row,column)==(2,7):
                elevation -= 8
            points.append([column,row,elevation])
            minimum[row,column] = elevation
    points += [[1,5,steepness+.8],[1,6,steepness-.8]]
    # A masked high return must retain its input class and not alter the minimum surface.
    points.append([5,5,-100])
    selection = [1]*(len(points)-1)+[0]
    classes = [6]*len(points)
    # Rasterize every selected return, including the extra signed-residual probes.
    for index, (column,row,elevation) in enumerate(points):
        if selection[index]:
            minimum[int(row),int(column)] = min(minimum[int(row),int(column)], elevation)
    minimum = fill(minimum)
    low = progressive(-minimum,5,1)
    net = np.zeros_like(low)
    cut_surface = minimum.copy()
    if cut:
        spacing = math.ceil(cut)
        net[::spacing,:] = True
        net[:,::spacing] = True
        opened = dilate(erode(minimum,2*spacing),2*spacing)
        cut_surface[net] = opened[net]
    objects = progressive(cut_surface,slope,math.ceil(window))
    provisional = minimum.copy()
    provisional[low | net | objects] = np.nan
    provisional = fill(provisional)
    vertical,horizontal = np.gradient(provisional)
    tolerance = threshold + scalar*np.hypot(horizontal,vertical)
    expected = []
    for index,(column,row,elevation) in enumerate(points):
        expected.append(6 if not selection[index] else (2 if abs(elevation-provisional[int(row),int(column)]) <= tolerance[int(row),int(column)] else 1))
    fixture = dict(name=name,width=size,height=size,positions=np.asarray(points,dtype=np.float32).ravel().tolist(),classifications=classes,selection=selection,slope=slope,scalar=scalar,threshold=threshold,window=window,cut=cut,groundSurface=provisional.ravel().tolist(),expectedClasses=expected)
    if arguments.pdal:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory)/'input.csv'
            output_path = Path(directory)/'output.csv'
            with input_path.open('w') as output:
                writer = csv.writer(output)
                writer.writerow(['X','Y','Z','Classification','Selected','RowId'])
                writer.writerows([*point,classes[index],selection[index],index] for index,point in enumerate(points))
            pipeline = [str(input_path),dict(type='filters.smrf',cell=1,slope=slope,scalar=scalar,threshold=threshold,window=window,cut=cut,where='Selected == 1'),dict(type='writers.text',filename=str(output_path),format='csv',order='X,Y,Z,Classification,RowId',keep_unspecified=False)]
            subprocess.run([arguments.pdal,'pipeline','--stdin'],input=json.dumps(pipeline),text=True,check=True,capture_output=True)
            actual = list(csv.DictReader(io.StringIO(output_path.read_text())))
            # where_merge can append excluded rows; preserve source correspondence by exact position.
            lookup = {int(float(row['RowId'])):int(float(row['Classification'])) for row in actual}
            fixture['pdalClasses'] = [lookup[index] for index in range(len(points))]
            fixture['pdalAgreement'] = sum(first==second for first,second in zip(expected,fixture['pdalClasses'])) / len(expected)
    return fixture

fixtures = [make_fixture('flat-object-low-outlier-holes',holes=True),make_fixture('slope-scalar',slope=.5,steepness=.4),make_fixture('slope-zero-scalar',slope=.5,steepness=.4,scalar=0),make_fixture('net-cut',cut=5,holes=True)]
output = dict(generator='NumPy 2.2.6 / SciPy 1.15.3 diamond morphology and numerical gradient',source='PDAL 2.8.4 SMRFilter algorithm',fixtures=fixtures)
if arguments.pdal:
    output['pdalVersion'] = subprocess.check_output([arguments.pdal,'--version'],text=True).strip()
Path(__file__).with_name('smrf.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
