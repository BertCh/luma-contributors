# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/geometry-measures/geometry-output-oracles.oracle.py
from pyproj import Proj, Geod
for kind in ['aeqd','laea']:
    projection = Proj(proj=kind,lat_0=0,lon_0=0,**({'R':6371008.8} if kind=='aeqd' else {'ellps':'WGS84'}))
    print(kind, [projection(*point) for point in [(0.5,0.25),(1,1)]])
print('geodesic',Geod(ellps='WGS84').inv(-73.75,40.5,2.25,48.75))
print('one-iteration spherical fallback',Geod(a=6371008.8,b=6371008.8).inv(0,30,100,-45)[2])
print('two/three iteration ellipsoid',Geod(ellps='WGS84').inv(0,20,170,60)[2])
print('two/three iteration sphere',Geod(a=6371008.8,b=6371008.8).inv(0,20,170,60)[2])

# Independent NumPy Vincenty convergence calculation in two precisions.
import numpy as np
for dtype in [np.float64, np.float32]:
    one, two, three, four, sixteen = [dtype(value) for value in [1,2,3,4,16]]
    radians = dtype(np.pi / 180)
    flattening = dtype(1 / 298.257223563)
    first, second = dtype(20)*radians, dtype(60)*radians
    def gap(phi):
        sine, cosine = np.sin(phi), np.cos(phi)
        return np.arctan(flattening*sine*cosine/(cosine*cosine+(one-flattening)*sine*sine))
    sine_first, sine_second = np.sin(first-gap(first)), np.sin(second-gap(second))
    cosine_first = np.sin(dtype(70)*radians+gap(first))
    cosine_second = np.sin(dtype(30)*radians+gap(second))
    sine_delta = np.sin(dtype(40)*radians-(gap(second)-gap(first)))
    longitude = dtype(170)*radians
    auxiliary = longitude
    for iteration in range(3):
        versine = two*np.sin(dtype(.5)*auxiliary)*np.sin(dtype(.5)*auxiliary)
        north = sine_delta+sine_first*cosine_second*versine
        east = cosine_second*np.sin(auxiliary)
        sine_sigma = np.sqrt(east*east+north*north)
        cosine_sigma = sine_first*sine_second+cosine_first*cosine_second*(one-versine)
        sigma = np.arctan2(sine_sigma,cosine_sigma)
        sine_alpha = cosine_first*cosine_second*np.sin(auxiliary)/sine_sigma
        cosine_alpha_squared = one-sine_alpha*sine_alpha
        double_sigma = cosine_sigma-two*sine_first*sine_second/cosine_alpha_squared
        correction = flattening/sixteen*cosine_alpha_squared*(four+flattening*(four-three*cosine_alpha_squared))
        previous = auxiliary
        auxiliary = longitude+(one-correction)*flattening*sine_alpha*(sigma+correction*sine_sigma*(double_sigma+correction*cosine_sigma*(-one+two*double_sigma*double_sigma)))
        difference = abs(auxiliary-previous)
        threshold = dtype(4.8e-7)*abs(auxiliary)
        print(dtype.__name__,iteration+1,'difference',difference,'threshold',threshold,'converged',difference<=threshold)
