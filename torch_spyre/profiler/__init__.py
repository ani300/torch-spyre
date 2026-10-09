# Copyright 2025-2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Spyre profiling package.

Public APIs include FFDC retrieval (``get_diagnostic_report``, also bound as
``torch.spyre.get_diagnostic_report``) and ``export_chrome_trace``, which adds
CPU-to-submission flows to an upstream ``torch.profiler`` capture. Device presence is
``torch.spyre.is_available()``, not a flag on this package.
"""

from torch_spyre.profiler._ffdc import get_diagnostic_report
from torch_spyre.profiler._trace import export_chrome_trace

__all__ = ["export_chrome_trace", "get_diagnostic_report"]
