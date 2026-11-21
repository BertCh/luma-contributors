// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Common spelling check, including named operations before deferred lowering. @internal */
export function validateSinkhornTransportRadiusOption(id: string, options: object): void {
  validateRequirement(
    id,
    !('blurRadius' in options),
    'blurRadius was removed. Use transportRadius'
  );
}
