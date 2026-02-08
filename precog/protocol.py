# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import List, Optional

import bittensor as bt
import pydantic


class Challenge(bt.Synapse):
    """ """

    # Required request input, filled by sending dendrite caller.
    timestamp: str = pydantic.Field(
        ...,
        title="Timestamps",
        description="The timestamp to predict from",
        allow_mutation=False,
    )

    # Optional request output, filled by recieving axon.
    prediction: Optional[float] = pydantic.Field(
        default=None,
        title="Predictions",
        description="The predictions to send to the dendrite caller",
    )

    # Optional request output, filled by recieving axon.
    interval: Optional[List[float]] = pydantic.Field(
        default=None,
        title="Interval",
        description="The predicted interval for the next hour. Formatted as [min, max]",
    )

    def deserialize(self) -> int:
        """
        Deserialize the dummy output. This method retrieves the response from
        the miner in the form of dummy_output, deserializes it and returns it
        as the output of the dendrite.query() call.

        Returns:
        - int: The deserialized response, which in this case is the value of dummy_output.

        Example:
        Assuming a Dummy instance has a dummy_output value of 5:
        >>> dummy_instance = Dummy(dummy_input=4)
        >>> dummy_instance.dummy_output = 5
        >>> dummy_instance.deserialize()
        5
        """
        return self.prediction
