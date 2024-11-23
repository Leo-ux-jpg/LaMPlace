### We provide the brief flow for LaMPlace.
***
1. Run following command to generate dataflow graph:
```
python make_graph.py
```
2. Train the preditor:
```
python graph_train.py
```
3. Generate the learnable flow:
```
python gen_l_flow.py
```
4. Run l_mask guided placement:
```
python lamplace.py
```
