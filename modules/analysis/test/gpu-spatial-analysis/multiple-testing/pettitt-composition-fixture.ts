// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';

/** Count destinations shared by the null-device and executable GPU composition fixtures. */
export function createPettittCompositionCountOutputs(
  createCountView: (name: string) => GraphDataView<'uint32'>
) {
  const familyCount = createCountView('effective');
  return {
    finiteTestCount: createCountView('finite'),
    effectiveFamilySize: familyCount
  };
}
