

## Run task worker and app

```bash
$ redis-server
$ rq worker 
$ python app.py
```

## Run tensorboard to explore model output data
```bash
$ tensorboard --logdir server/model/output/tb_runs
```
