// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getLensScenario, listLensScenarios} from '../../../src/computation-lenses/index';
import {validateSpatialWeightsCaptures} from '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-scenario-utils';
import {declareLensScenario} from '../scenario-declaration.test-utils';
import '../../../src/computation-lenses/scenarios/index';

for (const slug of [
  'spatial-weights-traces',
  'spatial-error-gm',
  'spatial-two-stage-least-squares',
  'spatial-lag-maximum-likelihood',
  'spatial-error-maximum-likelihood'
]) {
  it(`${slug} actual prepare capture list refuses candidate-only loss`, async () => {
    const scenarioId = `gpu-spatial-analysis.${slug}`;
    const scenario = getLensScenario(scenarioId);
    expect(scenario, `${scenarioId} must have its real registered definition`).toBeDefined();
    const registrations = listLensScenarios();
    // Run the real load/prepare/declare closure on a null device, exactly as the
    // registry-wide declaration gate does. No registration stubs or cache resets.
    // A second caller shares the loaded modules and registry. Repeat without resetting either:
    // test isolation must not conceal a cached definition or a consumed prepare closure.
    for (let caller = 0; caller < 2; caller++) {
      const declared = await declareLensScenario(scenario!);
      expect(listLensScenarios()).toEqual(registrations);
      expect(declared.captureNames).toContain('weightsOverflow');
      expect(declared.captureNames).toContain('weightsCandidateOverflow');
      const captures = declared.captureNames.map(name => ({
        name,
        values: Uint32Array.of(name === 'weightsCandidateOverflow' ? 1 : 0)
      }));
      expect(() => validateSpatialWeightsCaptures(captures)).toThrow(/weightsCandidateOverflow/);
      for (const capture of captures) capture.values.fill(0);
      expect(() => validateSpatialWeightsCaptures(captures)).not.toThrow();
    }
  });
}
