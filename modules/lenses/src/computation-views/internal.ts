// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// `@luma-contributors/lenses/computation-views`: the ladder-of-abstraction view runtime. Vanilla
// TypeScript; no DOM is touched at import time (only inside mount functions and the element).

// Shared vocabulary and the view tree
export {findViewNodes} from './view-node';
export {createViewContext} from './registry/view-registry';
export {renderResultView, buildResultViewModel} from './views/result-view';
export {LENS_RUNG_IDS, type LensRungId, type OperatorReduction, type OperatorScope} from './types';

// Registry and author contracts
export type {
  ViewContext,
  ViewPackData,
  ViewPackRegistration,
  ViewRegistration
} from './registry/view-registration';
export {ViewRegistry} from './registry/view-registry';
export {createDefaultViewRegistry} from './registry/default-registry';

// Coordinator (CPU backend + GPU backend interface)

export {LadderCoordinator} from './coordinator/ladder-coordinator';

// Static twins and the one description generator

export {
  type LensTwin,
  buildDecodedLensTwin,
  buildLensTwin,
  parseLensDirective,
  renderLensTwinHtml,
  validateLensTwin
} from './twin/static-twin';

// Interactive hosts

export {
  type LensAction,
  LensController,
  type LensControllerOptions,
  type LensFrame
} from './hosts/lens-controller';
export {
  type MountLadderLensOptions,
  type MountedLadderLens,
  mountLadderLens
} from './hosts/mount-lens';
export {defineLadderLensElement} from './hosts/ladder-lens-element';
