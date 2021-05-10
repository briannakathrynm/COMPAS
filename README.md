# Sample Impact of Fair Classifiers
Experimenting with different IBM AIF360 classifiers/algorithims on the COMPAS dataset (included in aif360.datasets). 

More information on the COMPAS dataset and examples are found in the PDF, located [here](https://github.com/ml4sts/sample_impact_of_fair_classifiers/blob/main/pre-research/CSC%20499%20J-Term%20Research.pdf).

The [functions](https://github.com/ml4sts/sample_impact_of_fair_classifiers/blob/main/functions_file.py) file contains all necessary functions as well as docstrings with how they work. The [example](https://github.com/ml4sts/sample_impact_of_fair_classifiers/blob/main/impact_of_eta_on_PR.ipynb) file shows usage of these functions, along with sources and other information about this research.

A summary of objectives and learned/completed outcomes is available [here](https://github.com/ml4sts/sample_impact_of_fair_classifiers/blob/main/CSC499_Summary_MacDonald.pdf).

To-do list with objectives that could be completed in other semesters:
- Add a way for eta to be a bounded value from **0 to eta_bound+1** and plot all values of eta in this range to the upper-bound. This couldn't be completed due to computational cost (i.e: my computer had boot-failed several times after attempting to plot and compute the fairness metrics for eta within a range, thus the function will only take one value of eta).
- Look at different datasets and how their metrics are graded. This could be done by using the functions and tweaking the "fetch_input" function.

Here are some key values of ETA to test:
- 0 
- 250
- 500
- 1000

**The difference in fairness metrics for Prejudice Remover is seen at higher values of eta. This is most noticeable if you compare the fairness metrics from ETA=0 and ETA=250.**
