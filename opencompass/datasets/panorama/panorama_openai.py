"""Custom OpenAI model wrapper for PANORAMA benchmark.

This module provides a custom OpenAI model class that skips the max_seq_len
check to handle the very long prompts in PANORAMA dataset without slow token
length calculations.
"""

from typing import Dict, List, Optional, Union

from opencompass.models import OpenAI
from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]


@MODELS.register_module()
class PanoramaOpenAI(OpenAI):
    """OpenAI model wrapper optimized for PANORAMA benchmark.

    This class overrides the _preprocess_messages method to skip the
    max_seq_len check when max_seq_len is None, allowing very long prompts
    to be processed without errors or slow token length calculations.
    """

    def _preprocess_messages(
        self,
        input: Union[str, PromptList],
        max_out_len: int,
        max_seq_len: int,
        mode: str,
        get_token_len_func,
    ) -> tuple:
        """Preprocess input into messages format.

        When max_seq_len is None, skip all length checks to allow very long
        prompts (like those in PANORAMA) to be processed efficiently.
        """
        # Skip length check if max_seq_len is None
        if max_seq_len is None:
            # Convert input to messages format directly without length checks
            if isinstance(input, str):
                messages = [{'role': 'user', 'content': input}]
            else:
                messages = []
                for item in input:
                    input_content = item['prompt']
                    msg = {'content': input_content}
                    if item['role'] == 'HUMAN':
                        msg['role'] = 'user'
                    elif item['role'] == 'BOT':
                        msg['role'] = 'assistant'
                    elif item['role'] == 'SYSTEM':
                        msg['role'] = 'system'
                    messages.append(msg)
            return messages, max_out_len

        # Otherwise, use parent class implementation
        return super()._preprocess_messages(
            input, max_out_len, max_seq_len, mode, get_token_len_func
        )
