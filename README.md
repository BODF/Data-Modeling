# Data-Modeling
Scripts for data modeling

Grasshopper Algorithm:
* Partly experimental, introduces fairness and balance metrics not in the original authors' work (Saremi et al 2017)
* Essentially brute forces finding an optimal parameter set for feeding into another ML algorithm
* Uses KNN, so it will get bogged down by massive datasets and is not parellelizable in Spark
* KNN was chosen by the original authors because it is particularly sensitive to poor parameter selections
