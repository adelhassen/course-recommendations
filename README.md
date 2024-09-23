# Course Recommendation System

### Introduction 

Finding a course can be an exhausting process leading to decision fatigue. This project aims to develop a personalized recommendation engine that suggests the most relevant data science and AI courses to an individual based on their preferences, interests, and goals. 

### Approach

Our content-based approach leverages the power of embeddings to find the best courses. Universal Sentence Encoder (USE) serves as the model at the core of the recommendation system. Universal Sentence Encoder (USE) is a pre-trained model created by Google for encoding sentences into embedding vectors that capture the semantic meaning of text. USE leverages transformer and Deep Averaging Network (DAN) architectures. USE performs well on zero-shot tasks where labeled data isn’t available given its extensive training on large-scale datasets.

### Example

**Prompt:** "I'm interested in learning NLP techniques using Python. I'd like a course that covers text preprocessing, sentiment analysis, and topic modeling."

**Output**: 


### Reproducibility Steps

```
pip install pipenv
```

```
pipenv install
```

```
pipenv shell
```

```
python -m spacy download en_core_web_lg
```

```
python data_processing.py
```

```
python model.py
```

### Notice

**This project is proprietary and closed-source. No part of this repository may be copied, modified, distributed, or used without the author's explicit permission.**


