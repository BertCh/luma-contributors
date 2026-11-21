# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent full-matrix Mantel reference: skbio.stats.distance.mantel and scipy.

Pearson, strict upper triangle, greater tail; no transforms of distances/times.
All 48636 Chicago rat reports are considered, then lexicographic geographic/time
strata select 2048 and 128 rows at fixed quantiles. EPSG:26916 km, local origin;
coordinates snapped to 1/1024 km and uploaded f32. Distances are scipy pdist
rounded f32 (the API consumes distances); TS reconstructs these INPUT distances
with Math.hypot, not any Mantel statistic. Times are integer days from source
local-clock epoch, exact f32. Full matrices, no truncated neighbor bands.

The API cannot accept supplied permutations: independently generated NumPy
assignments must NOT equal GPU same-seed draws. Store every independent reference
correlation (4095 draws) and use two-sample DKW CDF confidence bounds over all
reference and GPU values, plus tail/binomial uncertainty over 3 GPU seeds.
Observed r uses S. The 2048-row scale case has one draw and checks observed r;
the 128-row case tests inference. A six-row tied/duplicate case uses all 720
permutations (exact null). Undefined Pearson coefficients and inference serialize
as null; requested draws remain separate from usable draws. No false claim of
identical null draws. The deprecated observed spelling equals observedCorrelation.
https://scikit.bio/docs/latest/generated/skbio.stats.distance.mantel.html
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT/'examples/deck/spatial-analysis-showcase/public/data/chicago-311-rats'
# Correlations lie in [-1,1]. Treat binary64 differences within sixteen ulps at
# unit scale as numerical ties, preserving the unmodified independent draws.
MANTEL_CORRELATION_TIE_TOLERANCE = 16 * math.ulp(1.)
MANTEL_CASE_FIELDS = ('name', 'reference', 'n', 'positions', 'times', 'sourceRows',
                     'observedCorrelation', 'observed', 'pairCount',
                     'referencePermutationCount', 'usablePermutationCount', 'null',
                     'inference', 'constant', 'exactNull', 'permutationCount', 'seeds',
                     'inputFloat32Sha256')


def serializeMantelFixture(fixture):
    """Use makeCase's field order even when retaining older finite sample mappings."""
    ordered = {**fixture, 'cases': [{key: case[key] for key in MANTEL_CASE_FIELDS}
                                  for case in fixture['cases']]}
    return json.dumps(ordered, separators=(',', ':'), allow_nan=False)+'\n'


def calculateMantelCorrelation(distances, timeDistances, coefficient):
    """Pearson is unavailable below two pairs or with either constant vector.

    The coefficient callback is the independent scientific library, not a CPU twin.
    Domain checks and null-draw filtering need only the Python standard library.
    """
    if len(distances) < 2 or len(distances) != len(timeDistances):
        return None
    if not all(math.isfinite(value) for value in itertools.chain(distances, timeDistances)):
        return None
    if min(distances) == max(distances) or min(timeDistances) == max(timeDistances):
        return None
    observed = float(coefficient(distances, timeDistances))
    return observed if math.isfinite(observed) else None


def summarizeMantelInference(observed, null, exhaustive=False):
    """Exact tails count/N; Monte Carlo tails use (count+1)/(N+1); no draws means null."""
    inference = None
    if observed is not None and null:
        # Subtract an exactly represented sample first. Rounding the mean itself
        # before centering can give an identical nonzero sample positive variance.
        origin = null[0]
        deviations = [statistic-origin for statistic in null]
        meanDeviation = math.fsum(deviations) / len(null)
        mean = origin + meanDeviation
        variance = (math.fsum((deviation-meanDeviation)**2 for deviation in deviations) / (len(null)-1)
                    if len(null) > 1 else 0.)
        pseudocount = 0 if exhaustive else 1
        greaterCount = sum(statistic >= observed or
            math.isclose(statistic, observed, rel_tol=0., abs_tol=MANTEL_CORRELATION_TIE_TOLERANCE)
            for statistic in null)
        lesserCount = sum(statistic <= observed or
            math.isclose(statistic, observed, rel_tol=0., abs_tol=MANTEL_CORRELATION_TIE_TOLERANCE)
            for statistic in null)
        greater = (greaterCount+pseudocount) / (len(null)+pseudocount)
        lesser = (lesserCount+pseudocount) / (len(null)+pseudocount)
        inference = {'permutationMean': mean, 'permutationVariance': variance,
                     'greaterProbability': greater, 'lesserProbability': lesser,
                     'twoSidedProbability': min(1., 2*min(greater, lesser)),
                     'zScore': ((observed-origin)-meanDeviation)/math.sqrt(variance) if variance > 0 else None}
    return inference


def summarizeMantelReference(observed, distances, permutedTimeDistances, coefficient,
                            referencePermutationCount, exhaustive=False):
    """Filter individually undefined draws; an undefined observation has no inference.

    referencePermutationCount is the requested independent reference budget, whereas
    permutationCount on the containing case is the GPU comparison's requested budget.
    """
    null = []
    if observed is not None:
        for difference in permutedTimeDistances:
            statistic = calculateMantelCorrelation(distances, difference, coefficient)
            if statistic is not None:
                null.append(statistic)
    # observed is the deprecated spelling; use observedCorrelation.
    return {'observedCorrelation': observed, 'observed': observed,
            'pairCount': len(distances), 'referencePermutationCount': referencePermutationCount,
            'usablePermutationCount': len(null), 'null': null,
            'inference': summarizeMantelInference(observed, null, exhaustive)}


def makeCase(name, positions, times, sourceRows, draws, exhaustive=False):
    import numpy as numpy
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import pearsonr

    positions = numpy.asarray(positions, dtype=numpy.float32).astype(float)
    times = numpy.asarray(times, dtype=numpy.float32).astype(float)
    distances = pdist(positions).astype(numpy.float32).astype(float)
    timeDistances = pdist(times[:, None])
    def observedCoefficient(spatial, temporal):
        # Import only for a defined full-matrix observation. skbio itself rejects
        # fewer than three rows, even though those cases still have a pair count.
        from skbio.stats.distance import mantel
        return mantel(squareform(spatial), squareform(temporal), permutations=0, alternative='greater')[0]

    observed = calculateMantelCorrelation(distances, timeDistances, observedCoefficient)
    random = numpy.random.default_rng(31015)
    assignments = itertools.permutations(range(len(times))) if exhaustive else (random.permutation(len(times)) for _ in range(draws))
    def permutationDistances():
        for index, assignment in enumerate(assignments):
            difference = pdist(times[list(assignment), None])
            if index < 8:
                check = calculateMantelCorrelation(distances, difference, observedCoefficient)
                statistic = calculateMantelCorrelation(distances, difference,
                    lambda spatial, temporal: pearsonr(spatial, temporal).statistic)
                assert (check is None and statistic is None) or abs(check-statistic) < 1e-12
            yield difference

    reference = summarizeMantelReference(observed, distances, permutationDistances(),
        lambda spatial, temporal: pearsonr(spatial, temporal).statistic, draws, exhaustive)
    return {'name': name, 'reference': ('skbio.stats.distance.mantel; scipy.stats.pearsonr'
                if observed is not None else 'Pearson domain: fewer than two pairs or zero spread'),
            'n': len(times), 'positions': positions.ravel().tolist(), 'times': times.tolist(),
            'sourceRows': numpy.asarray(sourceRows).tolist(), **reference, 'constant': observed is None,
            'exactNull': exhaustive and observed is not None and len(reference['null']) == draws,
            'permutationCount': 511 if draws else 1,
            'seeds': [7, 123, 901] if draws else [7],
            'inputFloat32Sha256': {key: hashlib.sha256(array.astype('<f4').tobytes()).hexdigest() for key,array in
                                  [('positions',positions),('times',times),('distances',distances)]}}


def main():
    import numpy as numpy
    import scipy
    import skbio
    import pyproj

    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).parent/'fixtures')
    arguments=parser.parse_args()
    positions=numpy.fromfile(DATA/'position.bin',dtype='<f4').reshape(-1,2)
    times=numpy.fromfile(DATA/'time.bin',dtype='<u4')//86400
    order=numpy.lexsort((times, positions[:,1], numpy.floor(positions[:,0]*100)))
    transform=pyproj.Transformer.from_crs(4326,26916,always_xy=True)
    projected=numpy.column_stack(transform.transform(positions[:,0],positions[:,1]))/1000-[440,4600]
    projected=numpy.rint(projected*1024)/1024
    cases=[]
    for count,draws in [(2048,0),(128,4095)]:
        selected=order[numpy.linspace(0,len(order)-1,count,dtype=int)]
        cases.append(makeCase(f'chicago-rats-{count}',projected[selected],times[selected],selected,draws))
    smallPositions=[[0,0],[0,0],[1,0],[3,1],[4,4],[7,2]]
    cases.append(makeCase('ties-duplicates-exact-null',smallPositions,[0,0,1,3,3,8],[],720,True))
    cases.append(makeCase('constant-time',smallPositions,[3]*6,[],511))
    cases.append(makeCase('constant-spatial-distances',[[0,0]]*3,[0,1,3],[],511))
    cases.append(makeCase('one-pair',[[0,0],[1,0]],[0,1],[],511))
    fixture={'generator':Path(__file__).name,'versions':{'python':platform.python_version(),'numpy':numpy.__version__,'scipy':scipy.__version__,'scikit-bio':skbio.__version__,'pyproj':pyproj.__version__},
             'provenance':{'source':str(DATA.relative_to(ROOT)),'sourceCount':len(times),'sourceSha256':{name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ['manifest.json','position.bin','time.bin']},'crs':'EPSG:26916','units':'km; integer days since 2023-01-01 local clock','selection':'lexsort(floor(longitude*100),latitude,time), equally spaced ranks','referenceSeed':31015},'cases':cases}
    output=arguments.output_dir/'parity-wp1-mantel.json'
    serialized=serializeMantelFixture(fixture)
    assert len(serialized)<1500000
    output.write_text(serialized)
    print(output,len(serialized),[(case['name'],case['observed'],len(case['null'])) for case in cases])

if __name__=='__main__': main()
