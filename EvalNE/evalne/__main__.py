#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:
# Contact:
# Date: 18/12/2018

import logging
import os
import random
import time
import numpy as np
import pickle

from datetime import datetime
from datetime import timedelta
from sys import argv
from tqdm import tqdm

from evalne.evaluation.evaluator import *
from evalne.evaluation.split import *
from evalne.evaluation.pipeline import EvalSetup
from evalne.evaluation.score import Scoresheet, EvalScores
from evalne.evaluation.robustness import Attack
from evalne.utils import preprocess as pp
from evalne.utils import util
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV


def main():
    # Start timer
    start = time.time()

    # Simple check of input args
    if len(argv) != 2:
        print("Error: Wrong number of parameters!")
        print("Usage: python -m evalne <path_to_ini_file>")
        exit(-1)

    # Create evaluation setup and pass ini file
    setup = EvalSetup(argv[1])

    # Evaluate
    evaluate(setup)

    # Get execution time
    end = time.time() - start
    print("Evaluation finished in: {} ({:.2f} sec.)".format(str(timedelta(seconds=round(end))), end))


def evaluate(setup):
    # Set the random seed
    random.seed(setup.seed)
    np.random.seed(setup.seed)

    # Get input and output paths
    inpaths = setup.inpaths
    filename = '{}_{}_eval_{}'.format(setup.objective[0].upper(), setup.task, datetime.now().strftime("%m%d_%H%M"))
    outpath = os.path.join(os.getcwd(), filename)
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # Logging configuration (file opened in append mode)
    logging.basicConfig(filename=os.path.join(outpath, 'eval.log'), format='%(asctime)s - %(levelname)s: %(message)s',
                        datefmt='%d-%m-%y %H:%M:%S', level=logging.INFO)
    logging.info('Evaluation start')
    logging.info('Running evaluation using classifier: {}'.format(setup.classifier))

    # Create an EvalScores object to store the results
    cols = ['dataset', 'exp_repeat']
    if setup.objective == 'robustness':
        cols.extend(['atk_strategy', 'budget'])
    scoresheet = EvalScores(cols=cols, precatk_vals=setup.precatk_vals)

    # Initialize some variables
    lp_coef = dict()
    edge_split_times = list()
    repeats = setup.lp_num_edge_splits if setup.task in ['lp', 'sp'] else 1
    t = tqdm(total=len(inpaths) * repeats)
    t.set_description(desc='Progress on {} task'.format(setup.task))

    # Loop over all input networks
    for i in range(len(inpaths)):
        logging.info('====== Evaluating {} network ======'.format(setup.names[i]))
        print('\nEvaluating {} network...'.format(setup.names[i]))
        print('=====================================')

        # Create path to store info per network if needed
        nw_outpath = os.path.join(outpath, setup.names[i])
        if not os.path.exists(nw_outpath):
            os.makedirs(nw_outpath)

        # Load and preprocess the graph
        G, ids = preprocess(setup, nw_outpath, i)
        # print('Assortativity of the original network: {}'.format(nx.degree_assortativity_coefficient(G)))
        labels = None
        if setup.task == 'nc':
            try:
                labels = pp.read_labels(setup.labelpaths[i], idx_mapping=ids)
            except (ValueError, IOError) as e:
                logging.exception('Exception occurred while reading labels of `{}` network. Skipping network eval...'
                                  .format(setup.names[i]))
                break

        if setup.objective == 'robustness':
            logging.info('Running robustness evaluation using strategy: {} and budgets: {}'
                         .format(setup.attack_strategy, setup.budgets))
            approx = int(setup.attack_approx * len(G.nodes)) if setup.attack_approx else np.inf
            nw_atk = Attack(G, setup.attack_strategy, max(setup.budgets), approx=approx, node_labels=labels)
            for budget in setup.budgets:
                logging.info('------ Attack with budget {} ------'.format(budget))
                print('\nAttack with budget {}...'.format(budget))
                print('-------------------------------------')
                G_ = nw_atk.attack(budget)
                print('Assortativity of the network: {}'.format(nx.degree_assortativity_coefficient(G_)))
                print('Main CC size: {}'.format(len(max(nx.connected_components(G_), key=len))))
                print('Num edges after attack: {}'.format(len(G_.edges)))
                edge_split_time, lp_coef = evaluate_one_nw(setup, G_, G, labels, i, repeats, t, budget, scoresheet,
                                                           outpath, nw_outpath)
                edge_split_times.append(edge_split_time)

        else:
            budget = None
            edge_split_time, lp_coef = evaluate_one_nw(setup, G, None, labels, i, repeats, t, budget, scoresheet,
                                                       outpath, nw_outpath)
            edge_split_times.append(edge_split_time)

    # Store the results
    if setup.scores is not None:
        if setup.scores == 'all':
            scoresheet.write_tabular(filename=os.path.join(outpath, 'eval_output_tr.txt'), tr_te='train',
                                     cols='all')
            scoresheet.write_tabular(filename=os.path.join(outpath, 'eval_output_te.txt'), tr_te='test',
                                     cols='all')
        else:
            scoresheet.write_tabular(filename=os.path.join(outpath, 'eval_output_tr.txt'), tr_te='train',
                                     cols=cols + ['method', 'eval_time', 'edge_embed_method', setup.scores])
            scoresheet.write_tabular(filename=os.path.join(outpath, 'eval_output_te.txt'), tr_te='test',
                                     cols=cols + ['method', 'eval_time', 'edge_embed_method', setup.scores])

    scoresheet.write_pickle(os.path.join(outpath, 'eval.pkl'))

    # Close progress bar
    t.close()
    print('Average edge split times per dataset:')
    print(setup.names)
    print(np.array(edge_split_times).reshape(-1, repeats).mean(axis=1))
    if setup.task != 'nc' and (isinstance(setup.classifier, LogisticRegression) or
                               isinstance(setup.classifier, LogisticRegressionCV)):
        pickle.dump(lp_coef, open(os.path.join(outpath, 'classifier_coefficients.pkl'), "wb"))
    logging.info('Evaluation end\n\n')


def preprocess(setup, nw_outpath, i):
    """
    Graph preprocessing routine.
    """
    print('Preprocessing graph...')

    # Load a graph
    if setup.task == 'sp':
        G = pp.load_graph(setup.inpaths[i], delimiter=setup.separators[i], comments=setup.comments[i],
                          directed=setup.directed, datatype=int)
    else:
        G = pp.load_graph(setup.inpaths[i], delimiter=setup.separators[i], comments=setup.comments[i],
                          directed=setup.directed, datatype=float)

    # Preprocess the graph
    if setup.task == 'lp' and setup.split_alg == 'random':
        G, ids = pp.prep_graph(G, relabel=setup.relabel, del_self_loops=setup.del_selfloops, maincc=False)
    else:
        G, ids = pp.prep_graph(G, relabel=setup.relabel, del_self_loops=setup.del_selfloops)

    # Save preprocessed graph to a file
    if setup.save_prep_nw:
        pp.save_graph(G, output_path=os.path.join(nw_outpath, 'prep_nw.edgelist'), delimiter=setup.delimiter,
                      write_stats=setup.write_stats, write_weights=False, write_dir=True)

    # Return the preprocessed graph
    return G, ids


def eval_baselines(setup, nee, i, budget, scoresheet, repeat, nw_outpath):
    """
    Experiment to test the baselines.
    """
    print('Evaluating baselines...')

    for method in setup.lp_baselines:
        try:
            # Evaluate baseline methods
            if setup.directed:
                print('Input {} network is directed. Running baseline for all neighbourhoods specified...'
                      .format(setup.names[i]))
                for neigh in setup.neighbourhood:
                    result = nee.evaluate_baseline(method=method, neighbourhood=neigh, timeout=setup.timeout)
                    if setup.objective == 'robustness':
                        scoresheet.log_results(result, cols_vals=[setup.names[i], repeat, setup.attack_strategy, budget])
                    else:
                        scoresheet.log_results(result, cols_vals=[setup.names[i], repeat])

                    # Plot the curves if needed
                    if setup.curves is not None:
                        result.plot(filename=os.path.join(nw_outpath, '{}_rep_{}'.format(result.method, repeat)),
                                    curve=setup.curves)

            else:
                print('Input {} network is undirected. Running standard baselines...'.format(setup.names[i]))
                result = nee.evaluate_baseline(method=method, timeout=setup.timeout)
                if setup.objective == 'robustness':
                    scoresheet.log_results(result, cols_vals=[setup.names[i], repeat, setup.attack_strategy, budget])
                else:
                    scoresheet.log_results(result, cols_vals=[setup.names[i], repeat])
                # Plot the curves if needed
                if setup.curves is not None:
                    result.plot(filename=os.path.join(nw_outpath, '{}_rep_{}'.format(result.method, repeat)),
                                curve=setup.curves)
        except (MemoryError, AttributeError, TypeError, util.TimeoutExpired) as e:
            logging.exception('Exception occurred while evaluating method `{}` on `{}` network.'
                              .format(method, setup.names[i]))


def eval_other(setup, nee, i, budget, scoresheet, repeat, nw_outpath):
    """
    Function to test other embedding methods not integrated in the library.
    """
    print('Evaluating Embedding methods...')

    lp_coef = dict()
    if setup.methods_other is not None:
        # Evaluate non OpenNE method
        # -------------------------------
        for j in range(len(setup.methods_other)):
            try:
                if setup.task == 'nc':
                    # Evaluate the method
                    results = nee.evaluate_cmd(method_name=setup.names_other[j], command=setup.methods_other[j],
                                               input_delim=setup.input_delim_other[j],
                                               output_delim=setup.output_delim_other[j],
                                               tune_params=setup.tune_params_other[j],
                                               maximize=setup.maximize, write_weights=setup.write_weights_other[j],
                                               write_dir=setup.write_dir_other[j], timeout=setup.timeout,
                                               verbose=setup.verbose)
                    for r in results:
                        r.save_predictions(filename=os.path.join(nw_outpath,
                                                                 '{}_rep_{}_bgt_{}_sh_{}_preds'
                                                                 .format(r.method, repeat, budget,
                                                                         r.params['shuffle'])))
                        r.save_margins(filename=os.path.join(nw_outpath,
                                                             '{}_rep_{}_bgt_{}_sh_{}_mrgn'
                                                             .format(r.method, repeat, budget, r.params['shuffle'])))
                else:
                    # Evaluate the method
                    results = nee.evaluate_cmd(method_name=setup.names_other[j], method_type=setup.embtype_other[j],
                                               command=setup.methods_other[j],
                                               edge_embedding_methods=setup.edge_embedding_methods,
                                               input_delim=setup.input_delim_other[j],
                                               output_delim=setup.output_delim_other[j],
                                               tune_params=setup.tune_params_other[j], maximize=setup.maximize,
                                               write_weights=setup.write_weights_other[j],
                                               write_dir=setup.write_dir_other[j], timeout=setup.timeout,
                                               verbose=setup.verbose)
                    # Store LP model coefficients
                    if setup.embtype_other[j] != 'e2e' and (isinstance(setup.classifier, LogisticRegression) or
                                                            isinstance(setup.classifier, LogisticRegressionCV)):
                        lp_coef.update({setup.names_other[j]: nee.lp_model.coef_})

                    # Generate plots if necessary
                    if setup.curves is not None:
                        results.plot(filename=os.path.join(nw_outpath, '{}_rep_{}'.format(results.method, repeat)),
                                     curve=setup.curves)

                # Log the results
                if setup.objective == 'robustness':
                    scoresheet.log_results(results, cols_vals=[setup.names[i], repeat, setup.attack_strategy, budget])
                else:
                    scoresheet.log_results(results, cols_vals=[setup.names[i], repeat])

            except (MemoryError, ValueError, IOError, util.TimeoutExpired) as e:
                logging.exception('Exception occurred while evaluating method `{}` on `{}` network.'
                                  .format(setup.names_other[j], setup.names[i]))

    if setup.methods_opne is not None:
        # Evaluate methods from OpenNE
        # ----------------------------
        for j in range(len(setup.methods_opne)):
            try:
                # Evaluate the method
                if setup.directed:
                    command = setup.methods_opne[j] + \
                              " --graph-format edgelist --directed --input {} --output {} --representation-size {}"
                else:
                    command = setup.methods_opne[j] + \
                              " --graph-format edgelist --input {} --output {} --representation-size {}"

                if setup.task == 'nc':
                    # Evaluate the method
                    results = nee.evaluate_cmd(method_name=setup.names_opne[j], command=command, input_delim=' ',
                                               output_delim=' ', tune_params=setup.tune_params_opne[j],
                                               maximize=setup.maximize, write_weights=False, write_dir=True,
                                               timeout=setup.timeout, verbose=setup.verbose)
                    for r in results:
                        r.save_predictions(filename=os.path.join(nw_outpath,
                                                                 '{}_rep_{}_bgt_{}_sh_{}_preds'
                                                                 .format(r.method, repeat, budget,
                                                                         r.params['shuffle'])))
                        r.save_margins(filename=os.path.join(nw_outpath,
                                                             '{}_rep_{}_bgt_{}_sh_{}_mrgn'
                                                             .format(r.method, repeat, budget, r.params['shuffle'])))
                else:
                    # Evaluate the method
                    results = nee.evaluate_cmd(method_name=setup.names_opne[j], method_type='ne', command=command,
                                               input_delim=' ', edge_embedding_methods=setup.edge_embedding_methods,
                                               output_delim=' ', tune_params=setup.tune_params_opne[j],
                                               maximize=setup.maximize, write_weights=False, write_dir=True,
                                               timeout=setup.timeout, verbose=setup.verbose)
                    # Store LP model coefficients
                    if (isinstance(setup.classifier, LogisticRegression) or
                       isinstance(setup.classifier, LogisticRegressionCV)):
                        lp_coef.update({setup.names_opne[j]: nee.lp_model.coef_})

                    # Generate plots if necessary
                    if setup.curves is not None:
                        results.plot(filename=os.path.join(nw_outpath, '{}_rep_{}'.format(results.method, repeat)),
                                     curve=setup.curves)

                # Log the results
                if setup.objective == 'robustness':
                    scoresheet.log_results(results, cols_vals=[setup.names[i], repeat, setup.attack_strategy, budget])
                else:
                    scoresheet.log_results(results, cols_vals=[setup.names[i], repeat])

            except (MemoryError, ValueError, IOError, TypeError, util.TimeoutExpired) as e:
                logging.exception('Exception occurred while evaluating method `{}` on `{}` network.'
                                  .format(setup.names_opne[j], setup.names[i]))

    # Return the coefficients of the LP model
    return lp_coef


def evaluate_one_nw(setup, G, G_original, labels, i, repeats, t, budget, scoresheet, outpath, nw_outpath):

    lp_coef = dict()
    edge_split_time = 0
    # For each repeat of the experiment generate new edge splits
    for repeat in range(repeats):
        logging.info('------ Repetition {} of experiment ------'.format(repeat))
        print('\nRepetition {} of experiment...'.format(repeat))
        print('-------------------------------------')

        split_time = time.time()
        if setup.task == 'lp':
            # Create train and validation edge splits
            traintest_split = LPEvalSplit()
            trainvalid_split = LPEvalSplit()

            if G_original is not None:
                # In this case we are testing robustness
                # TODO: IN THIS EVALUATION WE CANNOT USE THE GENERAL LP TRAIN/TEST SPLIT. WE NEED TO USE THE NR SPLIT
                train_E, train_E_false = stt.random_edge_sample(nx.adj_matrix(G), setup.nr_edge_samp_frac,
                                                                nx.is_directed(G))
                test_E, test_E_false = stt.random_edge_sample(nx.adj_matrix(G_original), setup.nr_edge_samp_frac,
                                                              nx.is_directed(G_original))
                traintest_split.set_splits(train_E, train_E_false, test_E, test_E_false, directed=G.is_directed(),
                                           nw_name=setup.names[i], TG=G, split_id=repeat, split_alg='custom', # to bypass conectedness checks
                                           owa=setup.owa, verbose=setup.verbose)
                # TODO: This is just a workaround to make the evaluation work. Can't tune hyperparameters atm.
                trainvalid_split.set_splits(train_E, train_E_false, test_E, test_E_false, directed=G.is_directed(),
                                            nw_name=setup.names[i], TG=G, split_id=repeat, split_alg='custom',
                                            owa=setup.owa, verbose=setup.verbose)
            else:
                # In this case we are testing performance
                # For LP compute train/test and train/valid splits
                traintest_split.compute_splits(G, nw_name=setup.names[i], train_frac=setup.traintest_frac,
                                               split_alg=setup.split_alg, owa=setup.owa,
                                               fe_ratio=setup.fe_ratio, split_id=repeat, verbose=setup.verbose)
                trainvalid_split.compute_splits(traintest_split.TG, nw_name=setup.names[i],
                                                train_frac=setup.trainvalid_frac, split_alg=setup.split_alg,
                                                owa=setup.owa, fe_ratio=setup.fe_ratio, split_id=repeat,
                                                verbose=setup.verbose)

            # print('Assortativity of the validation network: {}'.format(
            #     nx.degree_assortativity_coefficient(traintest_split.TG)))
            # print('Assortativity of the train network: {}'.format(
            #     nx.degree_assortativity_coefficient(trainvalid_split.TG)))

            # Create an LP evaluator
            nee = LPEvaluator(traintest_split, trainvalid_split, setup.embed_dim, setup.classifier)

        elif setup.task == 'nr':
            # Create train edge split
            traintest_split = NREvalSplit()

            if G_original is not None:
                # We are testing robustness
                # First sample the original graph for some edges and non-edges
                train_E, train_E_false = stt.random_edge_sample(nx.adj_matrix(G_original), setup.nr_edge_samp_frac,
                                                                nx.is_directed(G_original))
                # Use the above edges and non-edges for evaluation, but ensure the TG is the attacked graph
                # so embeddings are leaned form the attacked graph but reconstruction of the original is evaluated
                traintest_split.set_splits(TG=G, train_E=train_E, train_E_false=train_E_false,
                                           samp_frac=setup.nr_edge_samp_frac, directed=nx.is_directed(G),
                                           nw_name=setup.names[i], split_id=repeat, verbose=setup.verbose)

            else:
                # We are testing performance
                # For NR compute train/test split only
                traintest_split.compute_splits(G, nw_name=setup.names[i], samp_frac=setup.nr_edge_samp_frac,
                                               split_id=repeat, verbose=setup.verbose)

            # Create an NR evaluator
            nee = NREvaluator(traintest_split, setup.embed_dim, setup.classifier)

        elif setup.task == 'sp':
            # Create train and validation edge splits
            traintest_split = SPEvalSplit()
            trainvalid_split = SPEvalSplit()

            # For SP compute train/test and train/valid splits
            traintest_split.compute_splits(G, nw_name=setup.names[i], train_frac=setup.traintest_frac,
                                           split_alg=setup.split_alg, split_id=repeat, verbose=setup.verbose)
            trainvalid_split.compute_splits(traintest_split.TG, nw_name=setup.names[i],
                                            train_frac=setup.trainvalid_frac, split_alg=setup.split_alg,
                                            split_id=repeat, verbose=setup.verbose)

            # Create an SP evaluator
            nee = SPEvaluator(traintest_split, trainvalid_split, setup.embed_dim, setup.classifier)

        else:
            # Create an NC evaluator (train/valid fraction hardcoded to 20%)
            nee = NCEvaluator(G, labels, setup.names[i], setup.nc_num_node_splits, setup.nc_node_fracs, 0.2,
                              setup.embed_dim, setup.classifier)

        edge_split_time = time.time() - split_time

        # Evaluate baselines
        if setup.lp_baselines is not None and setup.task != 'nc':
            eval_baselines(setup, nee, i, budget, scoresheet, repeat, nw_outpath)

        # Evaluate other NE methods
        if setup.methods_opne is not None or setup.methods_other is not None:
            lp_coef = eval_other(setup, nee, i, budget, scoresheet, repeat, nw_outpath)

        # Update progress bar
        t.update(1)

    # Store in a pickle file the results up to this point in evaluation
    scoresheet.write_pickle(os.path.join(outpath, 'eval.pkl'))

    return edge_split_time, lp_coef


if __name__ == "__main__":
    main()
