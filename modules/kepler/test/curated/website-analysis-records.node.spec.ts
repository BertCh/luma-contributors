// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {KEPLER_CURATED_ANALYSES} from '../../src/curated/curated-analyses';
import {
  WORKFLOWS,
  WORKFLOW_TEMPLATES
} from '../../../../website/src/components/meta/pages/workflows/workflows';
import {getCatalogAnalysisMetadata} from '../../../../website/src/catalog/analysis-metadata';

describe('website curated workflow records', () => {
  it('uses all shared templates by identity, rather than copying the question and method', () => {
    expect(WORKFLOW_TEMPLATES).toBe(KEPLER_CURATED_ANALYSES);
    expect(WORKFLOW_TEMPLATES).toHaveLength(28);
    expect(WORKFLOWS).toHaveLength(5);
    for (const workflow of WORKFLOWS) {
      const template = KEPLER_CURATED_ANALYSES.find(
        candidate => candidate.id === workflow.templateId
      );
      expect(workflow.template, workflow.id).toBe(template);
      expect(workflow.title, workflow.id).toBe(template!.name);
      expect(workflow.question, workflow.id).toBe(template!.title);
      expect(workflow.start, workflow.id).toBe(template!.slots.map(slot => slot.label).join('; '));
      expect(getCatalogAnalysisMetadata(workflow.recipe).questions.length).toBeGreaterThan(0);
    }
  });

  it('keeps recording evidence distinct from template execution and labels method differences', () => {
    const access = WORKFLOWS.find(workflow => workflow.id === 'drive-time')!;
    expect(access.template.execution).toMatchObject({
      status: 'available',
      bindingId: 'network-accessibility'
    });
    expect(access.recipe).toBe('addDriveTimeCatchmentRecipe');
    expect(access.recordNote).toContain('without bands or isochrones');
    const hotSpots = WORKFLOWS.find(workflow => workflow.id === 'hot-spots')!;
    expect(hotSpots.recordNote).toContain('not the template execution');
    const drainage = WORKFLOWS.find(workflow => workflow.id === 'drainage')!;
    expect(drainage.template.execution).toMatchObject({
      status: 'available',
      bindingId: 'curated-drainage-network'
    });
    expect(drainage.recordNote).toContain('3 by 3 test grid');
    for (const workflow of WORKFLOWS) {
      expect(workflow.record.kernels).toBeGreaterThan(0);
      expect(workflow.steps.length).toBeGreaterThan(0);
      expect(workflow.stages.length).toBeGreaterThan(0);
    }
  });
});
