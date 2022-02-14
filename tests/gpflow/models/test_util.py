# Copyright 2022 The GPflow Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Type
from unittest.mock import MagicMock

import pytest

from gpflow.models import (
    ExternalDataTrainingLossMixin,
    GPModel,
    InternalDataTrainingLossMixin,
    on_data_change,
)


@pytest.mark.parametrize(
    "model_class,call_expected",
    [
        (GPModel, False),
        (InternalDataTrainingLossMixin, True),
        (ExternalDataTrainingLossMixin, False),
    ],
)
def test_on_data_change(model_class: Type[GPModel], call_expected: bool) -> None:
    model = MagicMock(model_class)
    on_data_change(model)
    if call_expected:
        assert 1 == model.on_data_change.call_count
    else:
        assert not hasattr(model_class, "on_data_change")
