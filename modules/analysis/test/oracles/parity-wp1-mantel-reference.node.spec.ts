// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {spawnSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {mkdtempSync, readFileSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {expect, it} from 'vitest';
import fixture from './fixtures/parity-wp1-mantel.json';
import bandFixture from './fixtures/parity-wp1-mantel-band.json';

function runReferenceProbe(probe: string): unknown {
  // Exercise the generator's actual domain/summary functions under standard-library
  // Python. Scientific packages remain the full generator's explicit environment.
  const script = `
import json, runpy, sys
from pathlib import Path
globals().update(runpy.run_path(sys.argv[1]))
${probe}
`;
  const run = spawnSync(
    process.env['ORACLE_TEST_PYTHON'] ?? 'python3',
    ['-I', '-S', '-c', script, fileURLToPath(new URL('./parity-wp1-mantel.py', import.meta.url))],
    {encoding: 'utf8'}
  );
  expect(run.status, run.stderr).toBe(0);
  return JSON.parse(run.stdout);
}

it('NUM-03 leaves correlation and inference unavailable for constant or insufficient pairs', () => {
  const result = runReferenceProbe(`
def unexpected_coefficient(*arguments):
    raise AssertionError('An undefined coefficient must not reach the numerical library')
def unexpected_permutations():
    raise AssertionError('An undefined observation must not manufacture null draws')
    yield []
cases = []
for distances, temporal in [([1,2,3], [0,0,0]), ([1,1,1], [1,3,2]),
                            ([1], [2]), ([], [])]:
    observed = calculateMantelCorrelation(distances, temporal, unexpected_coefficient)
    cases.append(summarizeMantelReference(observed, distances, unexpected_permutations(),
                                         unexpected_coefficient, 19))
print(json.dumps(cases, allow_nan=False))
`);
  // Pearson needs at least two pairs and nonzero spread in BOTH vectors.
  expect(result).toEqual(
    [3, 3, 1, 0].map(pairCount => ({
      observedCorrelation: null,
      observed: null,
      pairCount,
      referencePermutationCount: 19,
      usablePermutationCount: 0,
      null: [],
      inference: null
    }))
  );
});

it('NUM-03 excludes individually degenerate draws without discarding a defined observation', () => {
  const result = runReferenceProbe(`
def coefficient(first, second):
    # For two nonconstant pairs Pearson is exactly the sign of the two slopes.
    assert first == [1,3] and second in ([1,2], [2,1])
    return 1.0 if second == [1,2] else -1.0
pairs = [(0,1), (0,2)]
def gaps(times):
    return [abs(times[first]-times[second]) for first, second in pairs]
observed = calculateMantelCorrelation([1,3], gaps([0,1,2]), coefficient)
# Real row permutations: gaps [1,1] are constant; gaps [2,1] reverse the slope.
draws = [gaps(times) for times in [[1,0,2], [0,2,1]]]
result = summarizeMantelReference(observed, [1,3], draws, coefficient, 2)
empty = summarizeMantelReference(observed, [1,3], draws[:1], coefficient, 1)
print(json.dumps([result, empty], allow_nan=False))
`);
  // Restricted pairs allow a constant permuted gap vector despite a defined
  // observation. One usable draw at -1 gives pseudo tails 1/2 and 1, no z score.
  expect(result).toEqual([
    {
      observedCorrelation: 1,
      observed: 1,
      pairCount: 2,
      referencePermutationCount: 2,
      usablePermutationCount: 1,
      null: [-1],
      inference: {
        permutationMean: -1,
        permutationVariance: 0,
        greaterProbability: 0.5,
        lesserProbability: 1,
        twoSidedProbability: 1,
        zScore: null
      }
    },
    {
      observedCorrelation: 1,
      observed: 1,
      pairCount: 2,
      referencePermutationCount: 1,
      usablePermutationCount: 0,
      null: [],
      inference: null
    }
  ]);
});

it('NUM-03 preserves finite independent references and refreshes band provenance', () => {
  expect(fixture.cases[3].observed).toBeNull();
  expect(fixture.cases.slice(0, 3).map(testCase => testCase.observed)).toEqual([
    0.015770371910276793, -0.022810951629205475, 0.8153860626783852
  ]);
  expect(fixture.cases.slice(0, 3).map(testCase => testCase.null.length)).toEqual([0, 4095, 720]);
  // Freeze the independently generated finite null draws, not a regenerated CPU/GPU twin.
  expect(
    fixture.cases
      .slice(0, 3)
      .map(testCase => createHash('sha256').update(JSON.stringify(testCase.null)).digest('hex'))
  ).toEqual([
    '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945',
    '223edb3fccec0d4080d678fb0777d825832d1df48c8eb4d020ba81cb579ad4d4',
    'f3d3fb97771851cda66282c19c02ac38a87991175f1f9aa76cca3f3650bbc849'
  ]);
  for (const testCase of fixture.cases) {
    expect(testCase.observedCorrelation).toBe(testCase.observed);
    expect(testCase.pairCount).toBe((testCase.n * (testCase.n - 1)) / 2);
    expect(testCase.usablePermutationCount).toBe(testCase.null.length);
  }
  for (const name of ['constant-time', 'constant-spatial-distances', 'one-pair']) {
    const testCase = fixture.cases.find(candidate => candidate.name === name);
    expect(testCase, name).toBeDefined();
    expect(testCase).toMatchObject({
      observedCorrelation: null,
      observed: null,
      referencePermutationCount: 511,
      usablePermutationCount: 0,
      null: [],
      inference: null,
      exactNull: false
    });
  }
  expect(fixture.cases[3].pairCount).toBe(15);
  const source = readFileSync(new URL('./fixtures/parity-wp1-mantel.json', import.meta.url));
  expect(bandFixture.sourceSha256).toBe(createHash('sha256').update(source).digest('hex'));
});

it('NUM-03 distinguishes exact tails and unavailable zero-variance z scores', () => {
  const result = runReferenceProbe(`
print(json.dumps([
    summarizeMantelInference(1, [1, -1], exhaustive=True),
    summarizeMantelInference(1, [1]),
], allow_nan=False))
`);
  // Exact enumeration has one of two draws >= 1, without Monte Carlo's pseudo-count.
  // A single usable draw can define the tail probability but cannot define a z score.
  expect(result).toEqual([
    {
      permutationMean: 0,
      permutationVariance: 2,
      greaterProbability: 0.5,
      lesserProbability: 1,
      twoSidedProbability: 1,
      zScore: 0.7071067811865475
    },
    {
      permutationMean: 1,
      permutationVariance: 0,
      greaterProbability: 1,
      lesserProbability: 1,
      twoSidedProbability: 1,
      zScore: null
    }
  ]);
});

it('NUM-03 independently regenerates every finite draw of the tied exact null', () => {
  const scene = fixture.cases[2];
  const result = runReferenceProbe(`
from struct import pack, unpack

def float32(value):
    return unpack('<f', pack('<f', value))[0]

def center(values):
    mean = math.fsum(values) / len(values)
    return [value - mean for value in values]

def pearson(first, second):
    dot = math.fsum(left * right for left, right in zip(first, second))
    squared_norms = math.fsum(value ** 2 for value in first) * math.fsum(value ** 2 for value in second)
    return dot / math.sqrt(squared_norms)

positions = ${JSON.stringify(scene.positions)}
times = ${JSON.stringify(scene.times)}
points = list(zip(positions[::2], positions[1::2]))
pairs = list(itertools.combinations(range(len(points)), 2))
spatial = center([float32(math.dist(points[first], points[second])) for first, second in pairs])

def correlation(assignment):
    temporal = center([abs(assignment[first] - assignment[second]) for first, second in pairs])
    return pearson(spatial, temporal)

print(json.dumps(dict(observed=correlation(times),
                      draws=[correlation(assignment) for assignment in itertools.permutations(times)])))
`);
  expect(result).toEqual({
    observed: expect.closeTo(scene.observedCorrelation!, 14),
    draws: scene.null.map(value => expect.closeTo(value, 14))
  });
});

it('B1 serializes every case in generator order and matches the committed bytes', () => {
  // The final three scenes were produced by makeCase; the first three retained
  // independent finite samples and must use the same field order for byte comparison.
  const keys = Object.keys(fixture.cases[3]);
  for (const testCase of fixture.cases) expect(Object.keys(testCase)).toEqual(keys);
  const result = runReferenceProbe(`
import hashlib
path = Path(sys.argv[1]).with_name('fixtures') / 'parity-wp1-mantel.json'
fixture = json.loads(path.read_text())
# Input mapping order cannot change the generator's serialized case layout.
fixture['cases'] = [dict(reversed(list(case.items()))) for case in fixture['cases']]
serialized = serializeMantelFixture(fixture).encode()
print(json.dumps(dict(matches=serialized == path.read_bytes(),
                      sha256=hashlib.sha256(serialized).hexdigest())))
`);
  expect(result).toEqual({matches: true, sha256: bandFixture.sourceSha256});
});

const mantelReferencePython = process.env['MANTEL_REFERENCE_PYTHON'];
it.skipIf(!mantelReferencePython)(
  'NUM-03 regenerates the complete Mantel fixture with the recorded scientific stack',
  () => {
    const directory = mkdtempSync(join(tmpdir(), 'mantel-reference-'));
    try {
      const generator = fileURLToPath(new URL('./parity-wp1-mantel.py', import.meta.url));
      const script = `
import importlib.metadata, json, platform, runpy, sys
expected = json.loads(sys.argv[1])
actual = {'python': platform.python_version()}
for package in expected.keys() - actual.keys():
    try:
        actual[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        actual[package] = 'missing'
if actual != expected:
    raise SystemExit(f'Expected Mantel reference versions {expected}; found {actual}')
sys.argv = sys.argv[2:]
runpy.run_path(sys.argv[0], run_name='__main__')
`;
      const run = spawnSync(
        mantelReferencePython!,
        [
          '-I',
          '-c',
          script,
          JSON.stringify(fixture.versions),
          generator,
          '--output-dir',
          directory
        ],
        {encoding: 'utf8', timeout: 45_000}
      );
      expect(run.error).toBeUndefined();
      expect(run.status, run.stderr).toBe(0);
      const regenerated = readFileSync(join(directory, 'parity-wp1-mantel.json'));
      const committed = readFileSync(new URL('./fixtures/parity-wp1-mantel.json', import.meta.url));
      expect(createHash('sha256').update(regenerated).digest('hex')).toBe(bandFixture.sourceSha256);
      expect(regenerated.equals(committed)).toBe(true);
    } finally {
      rmSync(directory, {recursive: true, force: true});
    }
  },
  60_000
);

it('B2 S2 counts both mathematical ties in a six-assignment exact null', () => {
  const result = runReferenceProbe(`
# SciPy pdist/pearsonr on collinear x=[0,1,2], times=[0,1,4], in itertools
# permutation order. Identity/reversal both equal sqrt(4/7) mathematically;
# the one-ulp difference is numerical roundoff, not a distinct tail event.
draws = [0.7559289460184543, -0.9449111825230679, 0.18898223650461354,
         -0.944911182523068, 0.18898223650461354, 0.7559289460184544]
result = []
for observed in [draws[0], draws[-1]]:
    summary = summarizeMantelInference(observed, draws, exhaustive=True)
    result.append([summary['greaterProbability'] * 6, summary['lesserProbability'] * 6,
                   summary['twoSidedProbability']])
print(json.dumps(result, allow_nan=False))
`);
  // Two maxima among six draws, all six <= the maximum: two-sided=2*(2/6)=2/3.
  expect(result).toEqual([
    [2, 6, 2 / 3],
    [2, 6, 2 / 3]
  ]);
});

it('B3 keeps repeated nonzero draws at zero variance and unavailable z score', () => {
  const result = runReferenceProbe(`
value = 0.7559289460184544
summary = summarizeMantelInference(value, [value, value, value])
print(json.dumps([summary['permutationMean'], summary['permutationVariance'], summary['zScore']],
                 allow_nan=False))
`);
  expect(result).toEqual([0.7559289460184544, 0, null]);
});
