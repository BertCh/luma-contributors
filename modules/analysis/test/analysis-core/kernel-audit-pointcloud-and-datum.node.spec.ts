// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {POINTCLOUD_AND_DATUM_CASES} from './kernel-audit/cases-pointcloud-and-datum';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('pointcloud-and-datum', POINTCLOUD_AND_DATUM_CASES);
