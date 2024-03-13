import logging
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, make_scorer
from sklearn.model_selection import cross_validate
# from pyspark import SparkConf, SparkContext
logging.basicConfig(format='%(message)s', level=logging.INFO)

logger = logging.getLogger(__name__)


class GrasshopperSwarm():
    def __init__(self, data, dependent_var, iterations=100, subgroups=None):
        # A Note on subgroups: we don't need to know the names here, just
        # the worst performing one to punish the model. Factorizing into
        # numbers makes selection a bit simpler later on
        logger.info("Init Grasshopper Feature Selection")
        self.y = data.loc[:, dependent_var]
        self.data = data.drop(dependent_var, axis=1)
        if subgroups:
            self.subgroups = pd.factorize(data[subgroups])[0]  # expecting a single col string
            self.data = data.drop(subgroups, axis=1)
        else:
            self.subgroups = None
        self.iterations = iterations
        # initialize the first grasshopper swarm
        self.rand_gen = np.random.default_rng()
        self.grasshoppers = self.rand_gen.integers(
            2,  # binary, 0's and 1's
            size=(10, self.data.shape[1])  # 10 grasshoppers x column width
            )

    # Define helper functions for use in the swarm

    # Build a matrix of pairwise distances between grasshoppers
    # Adapted from
    # https://stackoverflow.com/questions/64952027/compute-l2-distance-with-numpy-using-matrix-multiplication
    # These distances help to give sign to the grasshopper velocity vector
    def dist(self):  # dist = sqrt (distance vector self-dot product)
        sx = np.sum(self.grasshoppers**2, axis=1, keepdims=True)
        # returns a distance matrix
        self.distance = np.sqrt(-2 * self.grasshoppers.dot(self.grasshoppers.T) + sx + sx.T)

    # Now calculate a normalized distance matrix that will be used to rescale the raw distances.
    # For reasons that aren't clear, all grasshopper alg. authors normalize to [1,4]
    def norm_dist(self):  # input the raw distance matrix
        try:
            min_dist = min(self.distance[self.distance != 0])  # have to filter out zeros
        except:
            min_dist = 0
        max_dist = np.max(self.distance)
        if max_dist != 0:
            # Note: the trace will be corrupted by the next line, don't use the trace for anything
            scaled = (self.distance - min_dist) / (max_dist - min_dist)
        else:
            scaled = self.distance
        self.norm_distance = 4**scaled  # B/c the scale is [0,1], this returns a matrix that is [1,4]

    # Define the social attraction/repulsion function
    def social(
            self,
            attraction=0.5,  # defaults based on literature
            length_scale=1.5
            ):
        # make a tensor where each 2D matrix is one grasshopper's distances to all others
        for i in range(self.grasshoppers.shape[0]):  # for each grasshopper (row)
            if i == 0:  # initial matrix
                dist_matrix = self.grasshoppers - self.grasshoppers[i]  # get the distances
                # drop the self row and transpose so each row is grasshopper i's
                # variable d compared to each other grasshopper
                dist_matrix = np.delete(dist_matrix, i, axis=0).T
            else:  # stack the rest
                # the stacking is such that each grasshopper's comparisons will be
                # selectable from the first dimension, i.e. dist_matrix[0] is grasshopper 1's
                # distance calculations, with row 0 being grasshopper 1 vs 2
                buffer_matrix = self.grasshoppers - self.grasshoppers[i]
                buffer_matrix = np.delete(buffer_matrix, i, axis=0).T
                dist_matrix = np.append(dist_matrix, buffer_matrix)  # add in data
        # reshape dist_matrix to be a tensor
        dist_matrix = np.reshape(
            dist_matrix,
            (
                10,  # 10 grasshoppers
                self.grasshoppers.shape[1],  # num parameters
                self.grasshoppers.shape[0] - 1  # num comparisons (hoppers - 1)
            ))
        return attraction * np.exp(-dist_matrix / length_scale) - np.exp(-dist_matrix)

    # baseline fitness calculation
    def fitness(self, fit_inputs, alpha=0.99):
        # fit_inputs[0,1,2]: error rate, selected parameters, total parameters
        F = alpha * fit_inputs[0] + (1 - alpha) * (fit_inputs[1] / fit_inputs[2])
        return F

    # For the below, to calculate binary class specific error rates,
    # we assume (1 - sklearn's recall score) and flip the 'positive' class
    # to get recall for both 0 and 1 predictions. There are other metrics
    # one could use, but this matches the way sklearn's balanced accuracy metric works.
    def balanced_fitness(  # this is experimental, for imbalanced class issues
            self,
            # majority class error, minority error, selected, and total parameters
            fit_inputs,
            majority_weight=1,  # The fraction of rows in the majority class e.g. 0.98
            minority_weight=1,  # The fraction of the minority class, defaults to no usage
            alpha=0.99
            ):  # below uses the average or weighted avg error
        F = alpha * (fit_inputs[0] / majority_weight + fit_inputs[1] / minority_weight)/2 + \
            (1 - alpha) * (fit_inputs[2] / fit_inputs[3])
        return F

    def velocity(
            self,
            i,  # grasshopper i
            r,  # mutation rate reducer
            c,  # exploration rate reducer
            social_tensor,  # calc'd for each parameter and grasshopper combo
            upper_limit=2.079  # velocity limit
            # ,lower_limit=0  # if needed, subtract this from the upper limit
            ):
        # for both distance metrics, extract the i grasshopper's row and delete it's
        # self comparison. Then create the ratio
        dist_buffer = self.distance[i]
        dist_buffer = np.delete(dist_buffer, i)
        n_dist_buffer = self.norm_distance[i]
        n_dist_buffer = np.delete(n_dist_buffer, i)
        # Below, dist_ratio is a scaling vector of form:
        # [ratio_i_vs_j, ratio_i_vs_j+1 etc]
        dist_ratio = dist_buffer / n_dist_buffer
        # Below, the extracted social_tensor is of form:
        # [[i_vs_j, i_vs_j+1, ...] parameter 1
        #  [i_vs_j, i_vs_j+1, ...] parameter 2
        #  ...]
        # and multiplying by the dist_ratio will map the i vs j's by column appropriately
        return np.round(r * np.sum(
            c * (upper_limit/2) * social_tensor[i] * dist_ratio,
            axis=1  # sums across each row (parameter) of each grasshopper
        ), 4)

    def fitting_knn(
            self,
            X,
            y,
            grasshopper,
            fitness='basic',  # 'basic' or 'balanced'
            majority_label=1,
            minority_label=0,
            cv=5,
            knn_neighbors=5,
            knn_weight='uniform',
            subgroup_training=None,
            averaging='macro'  # use 'weighted' if you want to consider class size imbalance
            ):
        knn = KNeighborsClassifier(n_neighbors=knn_neighbors, weights=knn_weight)
        # use the grasshopper as a boolean selector of the data columns
        param_select = pd.array(grasshopper, dtype='boolean')
        X = X.loc[:, param_select]
        #
        # branches here depending upon the chosen fitness metric
        if cv == 1 and fitness == 'basic':
            knn.fit(X, y)  # bare knn fit
            accuracy = knn.score(X, y)
            fit_inputs = [
                1 - accuracy,  # error rate
                sum(param_select),  # num selected parameters
                len(param_select)  # total parameters
                ]
        elif cv == 1:
            knn.fit(X, y)
            y_pred = knn.predict(X)
            majority_error = 1 - recall_score(
                y,
                y_pred,
                pos_label=majority_label  # calculate majority label recall
            )
            minority_error = 1 - recall_score(
                y,
                y_pred,
                pos_label=minority_label
            )
            fit_inputs = [majority_error, minority_error, sum(param_select), len(param_select)]
        elif fitness == 'basic':
            # accuracy = knn.score(test_X, test_y)
            accuracy = cross_validate(knn, X, y, scoring='accuracy', cv=cv)
            accuracy = sum(accuracy['test_score']) / len(accuracy['test_score'])  # avg
            fit_inputs = [
                1 - accuracy,  # error rate
                sum(param_select),  # num selected parameters
                len(param_select)  # total parameters
                ]
        elif fitness == 'balanced' and subgroup_training:
            # get predictions for the test set then get (1 - recall)
            average_recall = make_scorer(  # averages between 1 and 0 outcome predictions
                recall_score,
                labels=subgroup_training,
                average=averaging  # default is 'macro' that is an unweighted 50:50 avg
            )
            accuracy = cross_validate(knn, X, y,
                                      scoring={'average_recall': average_recall},
                                      cv=cv
                                      )
            error = 1 - min(  # retrieve the weakest accuracy for retraining
                accuracy['test_average_recall']
                )
            fit_inputs = [error, sum(param_select), len(param_select)]
        elif fitness == 'balanced':
            # get predictions for the test set then get (1 - recall)
            major_recall = make_scorer(
                recall_score,
                pos_label=majority_label
            )
            minor_recall = make_scorer(
                recall_score,
                pos_label=minority_label
            )
            accuracy = cross_validate(knn, X, y,
                                      scoring={'major_recall': major_recall,
                                               'minor_recall': minor_recall},
                                      cv=cv
                                      )
            majority_error = 1 - sum(
                accuracy['test_major_recall']
                ) / len(accuracy['test_major_recall'])
            minority_error = 1 - sum(
                accuracy['test_minor_recall']
                ) / len(accuracy['test_minor_recall'])
            fit_inputs = [majority_error, minority_error, sum(param_select), len(param_select)]
        return fit_inputs

    def swarm(
            self,
            knn_weight='uniform',
            knn_neighbors=5,
            fitness='basic',  # fitness as 'basic' or 'balanced'
            majority_weight=1,  # <-- fractional size of class, e.g. 0.98
            minority_weight=1,  # weights only apply in 'balanced' case
            alpha=0.99,  # 99% weight to error over parsimony
            majority_label=1,  # for KNN fitting with imbalanced classes
            minority_label=0,
            cv=5,
            random_state=4855,
            test_split=0.8,
            # for subgrp knn, use 'weighted' if you want to consider class size imbalance
            averaging='macro'
            ):
        # initial fitness scores for random initial grasshoppers
        # create the same training and test sets for all hoppers
        # then run KNN and solve for the fitness scores
        if self.subgroups.any():
            X_train, X_test, y_train, y_test, subgrp_tr, subgrp_tst = train_test_split(
                self.data,
                self.y,
                self.subgroups,
                test_size=test_split,
                random_state=random_state,
                stratify=self.subgroups
                )
        else:
            subgrp_tr = None  # for compatibility lower down
            X_train, X_test, y_train, y_test = train_test_split(
                self.data,
                self.y,
                test_size=test_split,
                random_state=random_state
                )
        fit_scores = []
        for g in range(self.grasshoppers.shape[0]):
            # For a balanced problem
            if fitness == 'basic':
                fit_inputs = self.fitting_knn(
                    X_train, y_train, self.grasshoppers[g], cv=cv,
                    knn_weight=knn_weight, knn_neighbors=knn_neighbors,
                    subgroup_training=subgrp_tr
                    )
                fit_scores.append(
                    self.fitness(fit_inputs, alpha)
                )
            elif fitness == 'balanced' and subgrp_tr:
                fit_inputs = self.fitting_knn(
                    X_train, y_train, self.grasshoppers[g], fitness='balanced', cv=cv,
                    knn_weight=knn_weight, knn_neighbors=knn_neighbors,
                    subgroup_training=subgrp_tr,
                    averaging=averaging
                    )
                fit_scores.append(
                    self.fitness(fit_inputs, alpha)
                )
            elif fitness == 'balanced':
                fit_inputs = self.fitting_knn(
                    X_train, y_train, self.grasshoppers[g], fitness='balanced', cv=cv,
                    knn_weight=knn_weight, knn_neighbors=knn_neighbors,
                    majority_label=majority_label, minority_label=minority_label,
                    subgroup_training=subgrp_tr
                    )
                fit_scores.append(
                    self.balanced_fitness(fit_inputs, majority_weight, minority_weight, alpha)
                )
            else:
                raise ValueError("Choose a fitness metric: fitness = 'basic' or 'balanced'")
        # Now find the grasshopper with the smallest fitness value
        # Note: ties will default to the last min value in the fitness list
        fittest_index = 0  # default value to avoid errors
        for index in range(len(fit_scores)):
            if fit_scores[index] == min(fit_scores):
                fittest_index = index
        fittest = self.grasshoppers[fittest_index]
        fittest_score = fit_scores[fittest_index]  # retain for comparison
        ###################
        # main fitting loop
        ###################
        # error_buffer = []
        for i in range(self.iterations):
            #
            # Update c and r that truncate swarming behavior as iterations pass
            # Note: I have placed r in the most logical point in the eqn.
            # However, the authors of Mafarja 2019 do not explicitly say how
            #   they applied r
            # Also, the below i's in c and r had to be shifted b/c of Python indexing
            c = 1 - ((i + 1) * ((1 - 1e-5) / self.iterations))
            r = 0.9 - ((0.9 * i) / (self.iterations - 1))
            #
            # Calculate the grasshopper distances and forces
            self.dist()
            self.norm_dist()
            social_tensor = self.social()
            #
            # Calculate the current velocity vector
            deltaX = []  # just append to the list and np.reshape later
            for g in range(self.grasshoppers.shape[0]):  # for each grasshopper
                dX_buffer = self.velocity(
                    g,
                    r,
                    c,
                    social_tensor
                )
                deltaX.append(dX_buffer)
            # reshape such that each row is a grasshopper X num parameters
            deltaX = np.reshape(
                deltaX,
                (self.grasshoppers.shape[0], self.grasshoppers.shape[1])
                )
            # Update the grasshopper positions (parameter selections)
            for d in range(self.grasshoppers.shape[1]):  # parameters
                for g in range(self.grasshoppers.shape[0]):  # grasshoppers
                    if deltaX[g, d] >= 0:
                        self.grasshoppers[g, d] = fittest[d]  # set to fittest value
                    else:  # else randomly set to selected/unselected, 1/0
                        if self.rand_gen.random() >= 0.5:
                            self.grasshoppers[g, d] = 1
                        else:
                            self.grasshoppers[g, d] = 0
            #
            # Run the kNN again for the next iteration
            fit_scores = []
            # fit_input_buffer = []
            for g in range(self.grasshoppers.shape[0]):
                if fitness == 'basic':
                    fit_inputs = self.fitting_knn(
                        X_train, y_train, self.grasshoppers[g], cv=cv,
                        knn_weight=knn_weight, knn_neighbors=knn_neighbors,
                        subgroup_training=subgrp_tr
                        )
                    fit_scores.append(
                        self.fitness(fit_inputs, alpha)
                    )
                    # fit_input_buffer.append(fit_inputs)
                elif fitness == 'balanced' and subgrp_tr:
                    fit_inputs = self.fitting_knn(
                        X_train, y_train, self.grasshoppers[g], fitness='balanced', cv=cv,
                        knn_weight=knn_weight, knn_neighbors=knn_neighbors,
                        subgroup_training=subgrp_tr,
                        averaging=averaging
                        )
                    fit_scores.append(
                        self.fitness(fit_inputs, alpha)
                    )
                    # fit_input_buffer.append(fit_inputs)
                elif fitness == 'balanced':
                    fit_inputs = self.fitting_knn(
                        X_train, y_train, self.grasshoppers[g], fitness='balanced', cv=cv,
                        knn_weight=knn_weight, knn_neighbors=knn_neighbors,
                        majority_label=majority_label, minority_label=minority_label,
                        subgroup_training=subgrp_tr
                        )
                    fit_scores.append(
                        self.balanced_fitness(fit_inputs, majority_weight, minority_weight, alpha)
                    )
                    # fit_input_buffer.append(fit_inputs)
                else:
                    raise ValueError("Choose a fitness metric: fitness = 'basic' or 'balanced'")
            # Now find the grasshopper with the smallest fitness value
            # and replace the fittest grasshopper, only if this new one is better
            # Note: ties will default to the last min value in the fitness list
            for index in range(len(fit_scores)):
                if fit_scores[index] == min(fit_scores):
                    fittest_index = index
            if fit_scores[fittest_index] < fittest_score:
                fittest = self.grasshoppers[fittest_index]
                fittest_score = fit_scores[fittest_index]
        ''' Testing code below
        if fitness == 'basic':
            fit_inputs = self.fitting_knn(
                X_test, y_test, fittest, cv=1,  # force to just one fold, i.e. regular fit
                knn_weight=knn_weight, knn_neighbors=knn_neighbors
                )
            fit_input_buffer = fit_inputs
            logging.info(
                "A swarm finished: %s%% error rate (test data), %s selected vars, %s total vars, %s alpha, %s knn weight, %s knn neighbors, var selection: %s",
                round(fit_input_buffer[0], 4) * 100,
                fit_input_buffer[1],
                fit_input_buffer[2],
                alpha,
                knn_weight,
                knn_neighbors,
                fittest
                )
        elif fitness == 'balanced':
            fit_inputs = self.fitting_knn(
                X_test, y_test, fittest, fitness='balanced', cv=1,
                knn_weight=knn_weight, knn_neighbors=knn_neighbors,
                majority_label=majority_label, minority_label=minority_label
                )
            fit_input_buffer = fit_inputs
            logging.info(
                "A swarm finished: %s%% majority error (test data), %s%% minority error, %s selected vars, %s total vars, var selection: %s",
                round(fit_input_buffer[0], 4) * 100,
                round(fit_input_buffer[1], 4) * 100,
                fit_input_buffer[2],
                fit_input_buffer[3],
                fittest
                )
        fit_input_buffer.append(fittest_score)
        '''
        return fittest  # , fit_input_buffer  # , error_buffer
    
    # try to find an optimal hyperparameter set for the KNN and the fitness score
    # If using, make sure the testing code at the end of 'swarm' is uncommented
    def hyper_opt(
            self,
            alpha_set=[0.5, 0.75, 0.99],
            weight_set=['uniform', 'distance'],
            neighbor_set=[3, 5, 7],
            fitness='basic',
            majority_weight=1,  # <-- fractional size of class, e.g. 0.98
            minority_weight=1,  # weights only apply in 'balanced' case
            majority_label=1,  # for KNN fitting with imbalanced classes
            minority_label=0,
            cv=5,
            random_state=4855,
            test_split=0.8
            ):
        best_score = False  # initial value to force replacement on first loop
        best_grasshopper = []
        best_hyperparam = []
        for i in alpha_set:
            for j in weight_set:
                for k in neighbor_set:
                    # each swarm outputs its best grasshopper and
                    # associated metrics
                    new_contender, new_metrics = self.swarm(
                        alpha=i,
                        knn_weight=j,
                        knn_neighbors=k,
                        fitness=fitness,
                        majority_weight=majority_weight,
                        minority_weight=minority_weight,
                        majority_label=majority_label,
                        minority_label=minority_label,
                        cv=cv,
                        random_state=random_state,
                        test_split=test_split
                    )
                    if best_score:  # not first loop
                        # Minimization goal--> replace if true.
                        # * index -1 should be the fitness metric of the training data,
                        #   everything else is metadata
                        if best_score[-1] > new_metrics[-1]:
                            best_grasshopper = new_contender
                            best_score = new_metrics
                            best_hyperparam = [i, j, k]
                        # Below catches the case of the same grasshopper but a better KNN fit
                        # if best_grasshopper == new_contender and best_score[0] > new_metrics[0]:
                        #    best_score = new_metrics
                        #    best_hyperparam = [i,j,k]
                    else:  # first loop
                        best_grasshopper = new_contender
                        best_score = new_metrics
                        best_hyperparam = [i, j, k]
        if fitness == 'basic':
            logging.info(
                "The final parameter set's fit on the test data: %s%% error rate, %s selected vars, %s total vars, %s fitness\n%s alpha, %s knn weight, %s knn neighbors",
                round(best_score[0], 4) * 100,
                best_score[1],
                best_score[2],
                best_score[3],
                best_hyperparam[0],
                best_hyperparam[1],
                best_hyperparam[2]
                )
        elif fitness == 'balanced':
            logging.info(
                "The final parameter set's fit on the test data: %s%% majority error, %s%% minority error, %s selected vars, %s total vars, %s training data fitness\n%s alpha, %s knn weight, %s knn neighbors",
                round(best_score[0], 4) * 100,
                round(best_score[1], 4) * 100,
                best_score[2],
                best_score[3],
                best_score[4],
                best_hyperparam[0],
                best_hyperparam[1],
                best_hyperparam[2]
                )
        self.best_grasshopper = pd.array(best_grasshopper, dtype='boolean')
        return best_grasshopper

    def random_seeder(
            self,
            random_seeds=list(range(10)),
            fitness='basic',
            majority_weight=1,  # <-- fractional size of class, e.g. 0.98
            minority_weight=1,  # weights only apply in 'balanced' case
            majority_label=1,  # for KNN fitting with imbalanced classes
            minority_label=0,
            cv=5,
            test_split=0.8,
            threshold=0.2,
            # for subgrp knn, use 'weighted' if you want to consider class size imbalance
            # otherwise, this 'macro' default yields a 50:50 balancing
            averaging='macro'
            ):
        locusts = np.array(False)  # buffer for holding the list of grasshoppers
        for i in random_seeds:
            # each swarm outputs its best grasshopper and
            # associated metrics
            new_contender = self.swarm(
                alpha=0.99,
                knn_weight='uniform',
                knn_neighbors=3,
                fitness=fitness,
                majority_weight=majority_weight,
                minority_weight=minority_weight,
                majority_label=majority_label,
                minority_label=minority_label,
                cv=cv,
                random_state=i,
                test_split=test_split,
                # for subgrp knn, use 'weighted' if you want to consider class size imbalance
                averaging=averaging
            )
            if locusts.any():  # not first loop
                # Minimization goal--> replace if true.
                # * index -1 should be the fitness metric of the training data,
                #   everything else is metadata
                locusts = np.append(locusts, [new_contender], axis=0)
            else:  # first loop
                locusts = np.array([new_contender])
        # Now aggregate together and threshold
        threshold = round(  # convert the threshold to an int
            np.shape(locusts)[1] * threshold  # numb of features * %
        )
        locusts = np.sum(locusts, axis=0)
        locusts = np.where(
            locusts <= threshold,
            0,
            1
            )
        self.best_grasshopper = pd.array(locusts, dtype='boolean')
        # output names of columns to be dropped
        # first is for Pandas, second works in PySpark env
        deselect = self.data.columns.iloc[~self.best_grasshopper]
        # deselect = [column for column in self.data.columns if column not in self.best_grasshopper]
        return deselect

    def transform(self):  # returns the finished dataframe
        # subset the original data columns and paste the outcomes back in
        self.data = self.data.iloc[:, self.best_grasshopper]
        self.data.loc[:, 'outcome'] = self.y
        return self.data
        # transform features using grasshopper model
        # return self.model.transform(data)
