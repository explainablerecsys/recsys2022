# [CIKM 2020] Cafe: Coarse-to-Fine Neural Symbolic Reasoning for Explainable Recommendation

This repository contains the source code of the CIKM 2020 paper "Cafe: Coarse-to-Fine Neural Symbolic Reasoning for Explainable Recommendation" [2].

## Data
The Amazon Beauty dataset is available in the "data/" directory and the split is consistent with [1]. 
Data files include:
- *kg_entities.txt.gz*: KG entities where each row is in the format of *<entity_global_id>\t<entity_type>\t<entity_local_id>\t<entity_value>*. <entity_local_id> numbers the entities of the same type while <entity_global_id> assigns a unique id to each entity in KG.
- *kg_relations.txt.gz*: KG relations including reverse relation, where each row is in the format of *<relation_global_id>\t<relation_name>*.
- *kg_triples.txt.gz*: all KG triples including reverse edges, where each row is in the format of *<head_entity_global_id>\t<relation_global_id>\t<tail_entity_global_id>*.
- *kg_rules.txt.gz*: KG rules of length at most 3. Each row is a sequence of <relation_global_id>'s.
- *kg_embedding.ckpt*: The pretrained KG embeddings based on the work [1].
- *train_labels.txt.gz*: user-item training pairs, where in each row, the first number is a <user_id> and the rest numbers are <item_id>'s. These pairs are already contained in the KG triples, e.g., if a user-item pair (0, 100) is in the training labels, there exists a corresponding triple (user_0, purchase, item_100) in KG.
- *test_labels.txt.gz*: user-item test pairs. They are the groundtruth to be predicted by models, and are NOT part of the KG.

## How to Run
1. Data preprocessing.
```python
python preprocess.py --dataset <dataset_name>
```

2. Train neural-symbolic model.
```python
python train_neural_symbol.py --dataset <dataset_name> --name <model_name>
```
The model checkpoint can be located in the directory "tmp/<dataset_name>/<model_name>/symbolic_model_epoch*.ckpt".

3. Do path inference by the trained neural-symbolic model.
```python
python execute_neural_symbol.py --dataset <dataset_name> --name <model_name> --do_infer true
```
4. Execute neural program (tree layout given by user profile) for profile-guided path reasoning.
```python
python execute_neural_symbol.py --dataset <dataset_name> --name <model_name> --do_execute true
```

## References
[1] Qingyao Ai, et al. "Learning Heterogeneous Knowledge Base Embeddings for Explainable Recommendation." In *Algorithms*. 2018.  
[2] Yikun Xian, et al. "Cafe: Coarse-to-Fine Neural Symbolic Reasoning for Explainable Recommendation." In *CIKM*. 2020.  
