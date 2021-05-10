# Sample Impact of Fair Classifiers
Experimenting with different IBM AIF360 classifiers/algorithims on the COMPAS dataset (included in aif360.datasets). 

More information on the COMPAS dataset and examples are found in the PDF (located in pre-research).

To-do list with objectives that could be completed in other semesters:
- Add a way for eta to be a bounded value from **0 to eta_bound+1** and plot all values of eta in this range to the upper-bound. This couldn't be completed due to computational cost (i.e: my computer had boot-failed several times after attempting to plot and compute the fairness metrics for eta within a range, thus the function will only take one value of eta).
- Lok at different datasets and how their metrics are graded. This could be done by using the functions and tweaking the "fetch_input" function.
