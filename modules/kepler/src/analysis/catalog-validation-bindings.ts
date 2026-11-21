// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {VALIDATION_REGRESSION_BINDINGS} from './catalog-validation-regression';
import {VALIDATION_VECCHIA_BINDINGS} from './catalog-validation-vecchia';
import {VALIDATION_KRIGING_BINDING} from './catalog-validation-kriging';
import {VALIDATION_PREDICTION_BINDINGS} from './catalog-validation-prediction';
import {VALIDATION_SCORE_BINDINGS} from './catalog-validation-scores';

export const VALIDATION_BINDINGS = [
  ...VALIDATION_REGRESSION_BINDINGS,
  ...VALIDATION_VECCHIA_BINDINGS,
  VALIDATION_KRIGING_BINDING,
  ...VALIDATION_PREDICTION_BINDINGS,
  ...VALIDATION_SCORE_BINDINGS
];
