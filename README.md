# THROW TASK
```mpiexec -np 6 python -u train_HER.py --env-name=Throwing --entropy-regularization=0.001 --gamma=.95 --n-epochs=150 --ratio-offset=.01 --replay-k=8 --ratio-clip=.3 --action-noise=0.02 --seed=94 --two-goal --apply-ratio```
