# PRT-DRL-Experiments
## Create python environment through anaconda

```bash
conda create -n prt python=3.12
conda activate prt
pip install -e ./Gymnasium/
pip install -r requirements.txt
```

Note that for Gymnasium we modify four files: `Gymnasium/gymnasium/envs/classic_control/cartpole.py`, `Gymnasium/gymnasium/envs/classic_control/continuous_mountain_car.py`, `Gymnasium/gymnasium/envs/classic_control/mountain_car.py` and `Gymnasium/gymnasium/envs/box2d/lunar_lander.py`. 

## Run results

```bash
python rq1.py
python rq2.py
python rq3.py
```

## Development

### Add testing subjects

1. Create your testing subjects by inheriting `environments.SeedBase` and `environments.ExecuteBase` class.
2. Register them in `environments/__init__.py`. 

### Add testing methods

1. Create your testing methods by inheriting `frameworks.Framework` class and overwriting the `test` method.
2. Register them in `frameworks/__init__.py`. 

### Custom testing

We make an example in `example.py`.

1. Obtain the input domain class `SeedSpace` and the executing class `Execute` from `environments.ENVS` by your testing subject's id. 
2. Inherit `frameworks.Framework` class and overwrite the `terminate` method.
3. Import your testing method class.
4. Set testing parameters by creating a `frameworks.Args` class object.
5. Create a new class by multiple inheritance of the former two classes in step 2 and step 3.
6. Create the class instance in step 5.
7. (Optional) Assign a `pbar` attribute for the class instance, which is `tqdm.tqdm` instance.
8. Use the `test` method of the class instance in step 5.
9. Use the `save` method of the class instance in step 5. You can also overwrite the `save` method in step 2.