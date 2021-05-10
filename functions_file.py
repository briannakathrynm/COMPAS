# Imports
# AIF 360 Imports
from aif360.algorithms.inprocessing import PrejudiceRemover
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_compas
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
# SK-Learn Imports
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, f1_score
# Data Manipulation/Visualization Imports
import matplotlib.patches as patches
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


# Start function definitions
def model_performance(X_test, y_true, y_pred, probs):
    """
    Gets model performance by computing accuracy, confusion matrix results, F1 score and
    visualizing the results with a ROC-AUC curve.
    :param X_test: X test data.
    :param y_true: Y true data.
    :param y_pred: Y prediction data.
    :param probs: Probabilities of predictions.
    :return: Graph of all the computed model performance metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    preds = probs[:, 1]
    fpr, tpr, threshold = roc_curve(y_true, preds)
    roc_auc = auc(fpr, tpr)

    return accuracy, matrix, f1, fpr, tpr, roc_auc


def plot_model_performance(model, X_test, y_true):
    """
    Function for plotting model performance. Utilizes "model_performance" function.
    :param model: Model created using dataset. EX: logistic regression model, random forest model.
    :param X_test: X test data.
    :param y_true: Y true data, ground truths.
    :return: Returns a plot with the model performance metrics from previous function.
    """
    y_pred = model.predict(X_test)
    probs = model.predict_proba(X_test)
    accuracy, matrix, f1, fpr, tpr, roc_auc = model_performance(X_test, y_true, y_pred, probs)

    print('Accuracy of the model :')
    print(accuracy)
    print('F1 score of the model :')
    print(f1)

    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(1, 2, 1)
    sns.heatmap(matrix, annot=True, cmap='Blues', fmt='g')
    plt.title('Confusion Matrix')

    ax = fig.add_subplot(1, 2, 2)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic curve')
    plt.legend(loc="lower right")


def fairness_metrics(dataset, pred):
    """
    Produce fair metrics for input dataset, if the dataset is not of type "aif360 dataset" the dataset
    will automatically make groups.
    :param dataset: Dataset that is type "aif306 dataset" or uses the same structure.
    :param pred: Prediction dataset.
    :param pred_is_dataset: Optional value set to false. Only if the prediction data matches the input data.
    :return: Gives the output of the fair metrics, also returns the value stored in the "fair_metrics" variable to
    be used in subsequent functions.
    """
    dataset_pred = dataset.copy()
    dataset_pred.labels = pred

    cols = ['statistical_parity_difference', 'equal_opportunity_difference', 'average_abs_odds_difference', 'disparate_impact']
    obj_fairness = [[0, 0, 0, 1]]
    fair_metrics = pd.DataFrame(data=obj_fairness, index=['objective'], columns=cols)
    
    # Goes through each sensitive/non-sensitive attribute of the dataset
    for attr in dataset_pred.protected_attribute_names:
        idx = dataset_pred.protected_attribute_names.index(attr)
        privileged_groups = [{attr: dataset_pred.privileged_protected_attributes[idx][0]}]
        unprivileged_groups = [{attr: dataset_pred.unprivileged_protected_attributes[idx][0]}]

        classified_metric = ClassificationMetric(dataset,
                                                 dataset_pred,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)

        metric_pred = BinaryLabelDatasetMetric(dataset_pred,
                                               unprivileged_groups=unprivileged_groups,
                                               privileged_groups=privileged_groups)

        acc = classified_metric.accuracy()

        row = pd.DataFrame([[metric_pred.mean_difference(),
                             classified_metric.equal_opportunity_difference(),
                             classified_metric.average_abs_odds_difference(),
                             metric_pred.disparate_impact()]],
                           columns=cols,
                           index=[attr]
                           )
        fair_metrics = fair_metrics.append(row)

    fair_metrics = fair_metrics.replace([-np.inf, np.inf], 2)
    return fair_metrics


def plot_fairness_metrics(fair_metrics):
    """
    Utilizes "fair_metrics" function to plot the results.
    :param fair_metrics: Result return from "fair_metrics" function.
    :return: Returns a plot containing the results of the "fair_metrics" function.
    """
    fig, ax = plt.subplots(figsize=(20, 4), ncols=5, nrows=1)

    plt.subplots_adjust(
        left=0.125,
        bottom=0.1,
        right=0.9,
        top=0.9,
        wspace=.5,
        hspace=1.1
    )

    y_title_margin = 1.2

    plt.suptitle("Fairness Metrics", y=1.09, fontsize=20)
    sns.set(style="dark")

    cols = fair_metrics.columns.values
    obj = fair_metrics.loc['objective']
    size_rect = [0.2, 0.2, 0.2, 0.4, 0.25]
    rect = [-0.1, -0.1, -0.1, 0.8, 0]
    bottom = [-1, -1, -1, 0, 0]
    top = [1, 1, 1, 2, 1]
    bound = [[-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1], [0.8, 1.2], [0, 0.25]]

    for i in range(0, 4):
        plt.subplot(1, 4, i + 1)
        ax = sns.barplot(x=fair_metrics.index[1:len(fair_metrics)],
                         y=fair_metrics.iloc[1:len(fair_metrics)][cols[i]])

        for j in range(0, len(fair_metrics) - 1):
            a, val = ax.patches[j], fair_metrics.iloc[j + 1][cols[i]]
            marg = -0.2 if val < 0 else 0.1
            ax.text(a.get_x() + a.get_width() / 5, a.get_y() + a.get_height() + marg, round(val, 3), fontsize=15,
                    color='black')

        plt.ylim(bottom[i], top[i])
        plt.setp(ax.patches, linewidth=0)
        ax.add_patch(patches.Rectangle((-5, rect[i]), 10, size_rect[i], alpha=0.3, facecolor="green", linewidth=1,
                                       linestyle='solid'))
        plt.axhline(obj[i], color='black', alpha=0.3)
        plt.title(cols[i])
        ax.set_ylabel('')
        ax.set_xlabel('')


def fetch_and_plot(data, model, plot):
    """
    Combines the usage of "fair_metrics" function" and "plot_fair_metrics" function.
    :param data: Dataset to be used.
    :param model: Model to be used.
    :param plot: True or False value
    :param eta: Range of eta to calculate fair metrics on.
    :return: Returns the resulting fair_metrics and a plot.
    """
    pred = model.predict(data).labels
    fair = fairness_metrics(data, pred)

    if plot:
        plot_fairness_metrics(fair)
        print(fair)

    return fair


def add_to_df_metrics(df_metrics, model, fair_metrics, preds, probs, name):
    """
        Compiled data-frame that includes all metrics in an easy-to-read format.
        :param df_metrics: Metrics that were compiled in the "algo_metrics" function.
        :param model: Model that was used to fit onto the dataset.
        :param fair_metrics: Fair metrics that is returned in "fair_metrics" function.
        :param preds: Prediction data.
        :param probs: Probability of predictions data.
        :param name: Name of the model.
        :return: Returns the compiled data-frame.
        """
    return df_metrics.append(pd.DataFrame(data=[[model, fair_metrics, preds, probs]],
                                          columns=['model', 'fair_metrics', 'prediction', 'probs'], index=[name]))


def fetch_input(eta_value):
    """
     Driver function that takes a user-specified value of ETA to use for the rest 
     of the functions and plots. Also computes model for COMPAS data.
     :param eta_value: Value of eta to use.
     :return: Returns range of eta for plotting purposes.
    """
    # To-do: Allow for program to sense what sensitive/non-sensitive attribute is.
    # To-do: Allow for user to enter an upper-bound for eta (the program will run
    # for 0 to upper-bound+1 for all of the plots ans graphs - tried to do this but
    # it would cause my computer to crash.

    # Splitting and processing data
    data = load_preproc_data_compas()
    privileged_groups = [{"sex": 1}]
    unprivileged_groups = [{"sex": 0}]
    data_train, data_test = data.split([0.7], shuffle=True)
    # Logistic regression classifier and predictions for training data
    # To-do: Allow user to input what kind of model to use.
    scale_orig = StandardScaler()
    X_train = scale_orig.fit_transform(data_train.features)
    y_train = data_train.labels.ravel()
    y_test = data_train.labels.ravel()
    lmod = LogisticRegression()
    lmod.fit(X_train, y_train)
    # Plotting model performance
    plot_model_performance(lmod, X_train, y_test)
    # Initialzing dataframe for metrics
    df_metrics = pd.DataFrame(columns=['model', 'fair_metrics', 'prediction', 'probs'])
    # Intializing debiased PR model
    debiased_model = PrejudiceRemover(eta=eta_value, sensitive_attr='sex')
    debiased_model.fit(data_train)
    # Getting fair metrics and plotting
    fair = fetch_and_plot(data_test, debiased_model, plot=True)
    data_pred = debiased_model.predict(data_test)
    # Adding to metrics dataframe
    df_metrics = add_to_df_metrics(df_metrics, debiased_model, fair, data_pred.labels, data_pred.scores, 'PrejudiceRemover')
    