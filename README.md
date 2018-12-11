# ibl_algorithm

The folder called ibl_algorithms comes with a:  
- requirements.txt for downloading possible dependencies (pip install -r requirements.txt)  
- ibl.cfg configuration file in which you can define the specs of the algorithm you want to run  

When you define what you want to run in the configuration file you just run the MainLauncher.py file.  

NOTE: Don't worry about some Warnings that you may get in runtime.  

Concerning the configuration file:  
- dataset: the name of the dataset without the \textit{.arff} extension, which is in the same directory as this file  
- ib_algo: the ib algorithm you want to run  
    - ib1  
    - ib2  
    - ib3  
- run: the type of run you want  
    - all: run all the combinations/models  
    - weights: runs the combination defined in run_type using feature selection  
    - combo: runs the combination defined in run_type  
- run_type: the combinations you want to run in case you haven't chosen 'all'. The format is <n_neighbors>-<distance>-<voting>  
