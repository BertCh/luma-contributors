// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getFormatter} from '@kepler.gl/utils';
import {setPointLabelFormatterFactory} from './label-format';

setPointLabelFormatterFactory(
  (format, field) => getFormatter(format, field as never) as (value: unknown) => unknown
);
