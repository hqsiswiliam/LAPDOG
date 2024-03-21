

from . import base, fever, kilt, lm, mlm, multiple_choice, qa, section, causal_chat

AVAILABLE_TASKS = {m.__name__.split(".")[-1]: m for m in [base, mlm, lm, multiple_choice, kilt, section, fever, qa, causal_chat]}


def get_task(opt, tokenizer):
    if opt.task not in AVAILABLE_TASKS:
        raise ValueError(f"{opt.task} not recognised")
    task_module = AVAILABLE_TASKS[opt.task]
    return task_module.Task(opt, tokenizer)
