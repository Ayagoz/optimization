import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, hinge_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


from RegOptim.ml import diff_hinge_loss_lr, hinge_loss_coef


def count_grads(K_train, y_train, da_train, db_train, params, dJ=None, scaled=False, kernel=False,
                with_template=False, n_splits=10, ndim=3, random_state=0):
    if kernel:
        exp_K_train = np.exp(-params['kernel__gamma'] * K_train)
    else:
        exp_K_train = K_train.copy()

    roc_aucs = []
    grads_a, grads_b = [], []
    grads_J = []
    hinge_losses = []

    for idx_train, idx_test in StratifiedShuffleSplit(n_splits=n_splits, test_size=0.3,
                                                      random_state=random_state).split(K_train, y_train):

        exp_K_train_loc, exp_K_test_loc = exp_K_train[np.ix_(idx_train, idx_train)], exp_K_train[
            np.ix_(idx_test, idx_train)]
        y_train_loc, y_test_loc = y_train[idx_train], y_train[idx_test]

        lr = LogisticRegression(n_jobs=1, max_iter=10 ** 5)
        lr.set_params(**{'C': params['ml__C']})

        lr.fit(exp_K_train_loc, y_train_loc)

        roc_aucs.append(roc_auc_score(y_test_loc, lr.predict_proba(exp_K_test_loc).T[1]))

        y_pred = lr.predict(exp_K_test_loc)
        hinge_losses.append(np.mean(hinge_loss(y_pred, y_test_loc)))



        if with_template:
            if kernel:
                grad_a, grad_b, grad_J = diff_hinge_loss_lr(beta=lr.coef_,
                                                            proba_test=lr.predict_proba(exp_K_test_loc).T[1],
                                                            proba_train=lr.predict_proba(exp_K_train_loc).T[1],
                                                            y_train=y_train_loc, y_test=y_test_loc,
                                                            exp_K_test=exp_K_test_loc,
                                                            exp_K_train=exp_K_train_loc,
                                                            da_train=da_train[np.ix_(idx_train, idx_train)],
                                                            db_train=db_train[np.ix_(idx_train, idx_train)],
                                                            da_test=da_train[np.ix_(idx_test, idx_train)],
                                                            db_test=db_train[np.ix_(idx_test, idx_train)],
                                                            alpha=params['kernel__gamma'],
                                                            dJ_train=dJ[np.ix_(idx_train, idx_train)],
                                                            dJ_test=dJ[np.ix_(idx_test, idx_train)],
                                                            scaled=scaled, kernel=kernel,
                                                            with_template=with_template, ndim=ndim)
            else:
                grad_a, grad_b, grad_J = diff_hinge_loss_lr(beta=lr.coef_,
                                                            proba_test=lr.predict_proba(exp_K_test_loc).T[1],
                                                            proba_train=lr.predict_proba(exp_K_train_loc).T[1],
                                                            y_train=y_train_loc, y_test=y_test_loc,
                                                            exp_K_test=exp_K_test_loc,
                                                            exp_K_train=exp_K_train_loc,
                                                            da_train=da_train[np.ix_(idx_train, idx_train)],
                                                            db_train=db_train[np.ix_(idx_train, idx_train)],
                                                            da_test=da_train[np.ix_(idx_test, idx_train)],
                                                            db_test=db_train[np.ix_(idx_test, idx_train)],
                                                            dJ_train=dJ[np.ix_(idx_train, idx_train)],
                                                            dJ_test=dJ[np.ix_(idx_test, idx_train)],
                                                            scaled=scaled, kernel=kernel,
                                                            with_template=with_template, ndim=ndim)

            grads_J.append(grad_J)
        else:
            if kernel:
                grad_a, grad_b = diff_hinge_loss_lr(beta=lr.coef_,
                                                proba_train=lr.predict_proba(exp_K_train_loc).T[1],
                                                proba_test=lr.predict_proba(exp_K_test_loc).T[1],
                                                y_train=y_train_loc, y_test=y_test_loc,
                                                exp_K_test=exp_K_test_loc,
                                                exp_K_train=exp_K_train_loc,
                                                da_train=da_train[np.ix_(idx_train, idx_train)],
                                                db_train=db_train[np.ix_(idx_train, idx_train)],
                                                da_test=da_train[np.ix_(idx_test, idx_train)],
                                                db_test=db_train[np.ix_(idx_test, idx_train)],
                                                alpha=params['kernel__gamma'],
                                                dJ_train=None, dJ_test=None,
                                                scaled=scaled, kernel=kernel,
                                                with_template=with_template, ndim=ndim)
            else:
                grad_a, grad_b = diff_hinge_loss_lr(beta=lr.coef_,
                                                    proba_train=lr.predict_proba(exp_K_train_loc).T[1],
                                                    proba_test=lr.predict_proba(exp_K_test_loc).T[1],
                                                    y_train=y_train_loc, y_test=y_test_loc,
                                                    exp_K_test=exp_K_test_loc,
                                                    exp_K_train=exp_K_train_loc,
                                                    da_train=da_train[np.ix_(idx_train, idx_train)],
                                                    db_train=db_train[np.ix_(idx_train, idx_train)],
                                                    da_test=da_train[np.ix_(idx_test, idx_train)],
                                                    db_test=db_train[np.ix_(idx_test, idx_train)],
                                                    dJ_train=None, dJ_test=None,
                                                    scaled=scaled, kernel=kernel,
                                                    with_template=with_template, ndim=ndim)


        grads_a.append(grad_a)
        grads_b.append(grad_b)

    print 'Hinge Loss Mean: ', np.mean(hinge_losses)
    print 'ROC AUC MEAN: ', np.mean(roc_aucs)

    if with_template:
        return np.mean(grads_a), np.mean(grads_b), np.mean(grads_J, axis=0), np.mean(roc_aucs), np.mean(hinge_losses)

    return np.mean(grads_a), np.mean(grads_b), np.mean(roc_aucs), np.mean(hinge_losses)


def test_score_prediction(K, y, idx_train, idx_test, params, scaled=False, kernel=False):
    if kernel:
        exp_K = 1 - np.exp(-params['kernel__gamma'] * K)
    else:
        exp_K = K.copy()

    K_train, y_train = exp_K[np.ix_(idx_train, idx_train)], y[np.ix_(idx_train)]
    K_test, y_test = exp_K[np.ix_(idx_test, idx_train)], y[np.ix_(idx_test)]

    if scaled:
        sc = StandardScaler(with_std=False)
        K_train = sc.fit_transform(K_train)
        K_test = sc.transform(K_test)

    # find best params on train

    # find best score on test
    clf = LogisticRegression(C=params['ml__C'], max_iter=10 ** 5)
    clf.fit(K_train, y_train)

    lr_best_score = roc_auc_score(y_test, clf.predict_proba(K_test).T[1])

    print "Test scores: ", lr_best_score

    return lr_best_score, hinge_loss(y_test, clf.predict_proba(K_test).T[1])
