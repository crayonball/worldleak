"""OpenUnlearning eval with an in-context unlearning instruction (UNL_ICU_PREFIX) prepended to every user prompt."""

import os
import runpy
import sys

import data.utils as du

_orig = du.preprocess_chat_instance
PREFIX = os.environ["UNL_ICU_PREFIX"]


def preprocess_chat_instance(tokenizer, template_config, prompt_msgs, response_msgs, *args, **kwargs):
    if isinstance(prompt_msgs, str):
        prompt_msgs, response_msgs = [prompt_msgs], [response_msgs]
    prompt_msgs = list(prompt_msgs[:-1]) + [PREFIX + prompt_msgs[-1]]
    return _orig(tokenizer, template_config, prompt_msgs, response_msgs, *args, **kwargs)


for m in list(sys.modules.values()):
    if getattr(m, "preprocess_chat_instance", None) is _orig:
        m.preprocess_chat_instance = preprocess_chat_instance
sys.argv = ["src/eval.py", *sys.argv[1:]]
runpy.run_path("src/eval.py", run_name="__main__")
