## Procedure to Execute Experiments

> [!Note]
> Only the procedures to train, batch-wise sample, and test the embedding network are included below.
> The remainder of the execution procedure is yet to be documented, but follows a similar workflow and can be retraced through the modularly structured pluggable portions and the commented lines in [run.py](src/run.py).

### Clone the repository and set up the Python environment
  - This implementation used **anaconda** for managing libraries, and the environment can be reproduced using the dependency file: [dependencies.yml](./dependencies.yml).
  - For instance, to clone the dependencies into a new environment, use,
    ```
    conda env create -f dependencies.yml
    ```
  - **Alternatively**, use `pip` manager to install the dependencies from the dependencies.txt](./dependencies.txt) file.
      ```
      pip install -r dependencies.txt
      ```
      

  #### Further References
  - A detailed guide for managing anaconda environbments can be found at [anaconda's official page](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
  - A helpful reference for managing pip packages can be [found here](https://note.nkmk.me/en/python-pip-install-requirements/).

### Train the Embedding Network

Keeping all parameters fixed, and using the default architecture for the underlying network, the following steps will train the embedding network.

> [!Note]
> - By default, the model states after each epoch will be saved into [`Experiments/logs/prototypical`](./Experiments/logs/prototypical).
> - To train on a different dataset, the dataset interfacing classes need to be defined according to the data files structure. See [the class for ISIC 2018 data](/Experiments/src/data/ISIC18_T3_Dataset.py), for instance.

- Activate the conda environment set up in the previous stage.
  ```
  conda activate skin_fsl
  ```
- Navigate to `Skin-FSL/prototypical/Experiments/src`.
- Edit [run.py](src/run.py) and uncomment the line to call the trainer. The line is a function call like so: `trainer.train()`.
  - To edit the file on a CLI interface, *nano* editor is a convenient option.
    ```
    nano run.py    (to open the file in nano editor)
    ctrl+o         (to save changes)
    ctrl+x         (to close the editor)
    ```
- Execute [run.py](src/run.py) (not context-sensitive).
  ```
  python run.py
  ```
  
- The following is a sample output when running the training experiment.
  <img src="../Docs/assets/embedding-train-op.png" width="950" />

### Test the Embedding Network

> [!Note]
> - The model state for evaluation is set by default. To change this, change the value assigned to variable `model_path` in [src/prototypical/tester_exhaustive.py](src/prototypical/tester_exhaustive.py).
> - To test on a different dataset, the dataset interfacing classes need to be defined according to the data files structure. See [the class for ISIC 2018 data](/Experiments/src/data/ISIC18_T3_Dataset.py), for instance.

- Activate the conda environment set up in the previous stage.
  ```
  conda activate skin_fsl
  ```
- Navigate to `Skin-FSL/prototypical/Experiments/src`.
- Edit [run.py](src/run.py) and uncomment the line to call the tester. The line is a function call like so: `tester_exhaustive.test()`.
  - To edit the file on a CLI interface, *nano* editor is a convenient option.
    ```
    nano run.py    (to open the file in nano editor)
    ctrl+o         (to save changes)
    ctrl+x         (to close the editor)
    ```
- Execute [run.py](src/run.py) (not context-sensitive).
  ```
  python run.py
  ```

- The following is a sample output when running a test on the trained network.
  <img src="../Docs/assets/embedding-test-op.png" width="950" />
