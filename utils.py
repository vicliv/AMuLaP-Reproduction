task_input_key = {
    "debat": ["sentence1", "sentence2"],
    "trump": ["sentence"],
    "sst2": ["sentence"],
    "cola": ["sentence"],
    "mnli": ["sentence1", "sentence2"],
    "mrpc": ["#1 String", "#2 String"],
    "qnli": ["question", "sentence"],
    "qqp": ["question1", "question2"],
    "rte": ["sentence1", "sentence2"],
}

task_label_key = {
    "debat": "label",
    "trump": "label",
    "sst2": "label",
    "cola": "label",
    "mnli": "gold_label",
    "mrpc": "Quality",
    "qnli": "label",
    "qqp": "is_duplicate",
    "rte": "label",
}

task_metric = {
    "debat": "accuracy",
    "trump": "accuracy",
    "sst2": "accuracy",
    "cola": "matthews_correlation",
    "mnli": "accuracy",
    "qnli": "accuracy",
    "rte": "accuracy",
    "mrpc": "f1",
    "qqp": "f1",
}