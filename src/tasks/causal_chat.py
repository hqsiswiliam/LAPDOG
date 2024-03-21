

import random

from src.evaluation import exact_match_score, f1_score, normalize_answer
from src.options import Options
from src.tasks.base import BaseTask


class Task(BaseTask):
    metrics = ["exact_match", "f1", "eval_loss"]

    def __init__(self, opt: Options, *args, **kwargs):
        super().__init__()
        self.opt = opt
        self.qa_prompt_format_str = " {question}"
        self.tokenizer = args[0]
        if 't5' in self.tokenizer.name_or_path and self.tokenizer.bos_token is None:
            self.tokenizer.add_special_tokens({'bos_token': 'R:'})
        assert self.tokenizer.bos_token is not None

    def get_qa_prompt(self, question: str) -> str:
        return self.qa_prompt_format_str.format(question=question)

    def process(self, example, *args, **kwargs):

        if "target" in example:
            target = example["target"]
        elif "answers" in example:
            target = random.choice(example["answers"])
        else:
            target = None

        if not "passages" in example:
            example["passages"] = [{"title": "", "text": ""}]

        example["metadata"] = example.get("metadata", {})
        example["query"] = self.get_qa_prompt(example["question"])
        if target is not None:
            if self.opt.dont_add_bos:
                example["target"] = f"{target}"
            else:
                example["target"] = self.tokenizer.bos_token+f" {target}"

        return example

    def evaluation(self, prediction, ground_truths):
        sample_metrics = {
            "exact_match": exact_match_score(prediction, ground_truths, normalize_answer),
            "f1": f1_score(prediction, ground_truths, normalize_answer),
        }
        return sample_metrics
