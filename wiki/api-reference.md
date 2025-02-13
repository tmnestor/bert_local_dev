# API Reference

## Training API
```python
from bert_classifier import Trainer

trainer = Trainer(
    model_name="bert-base-uncased",
    num_labels=2
)
```

## Inference API
```python
from bert_classifier import BertClassifier

classifier = BertClassifier.from_pretrained("path/to/model")
predictions = classifier.predict(texts=["example text"])
```

## Optimization API
```python
from bert_classifier import Optimizer

optimizer = Optimizer(
    search_space=search_space,
    n_trials=50
)
```

## Evaluation API
```python
from bert_classifier import Evaluator

evaluator = Evaluator(model_path="path/to/model")
metrics = evaluator.evaluate(test_data="test.csv")
```
