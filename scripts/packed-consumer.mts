// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_LINE_DENSITY_PARAMETER_SCHEMA} from '@luma-contributors/analysis/gpu-spatial-analysis';

const schemaWordLength: number = GPU_LINE_DENSITY_PARAMETER_SCHEMA.wordLength;
void schemaWordLength;
// @ts-expect-error An invented field must remain a type error in a native consumer.
GPU_LINE_DENSITY_PARAMETER_SCHEMA.notARealField;
