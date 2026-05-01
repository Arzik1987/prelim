import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y
from .generators import build_generator


def _require_predict_proba(bb_model, gen_name, proba):
    if hasattr(bb_model, "predict_proba"):
        return

    if gen_name == "vva":
        raise ValueError("Generator 'vva' requires bb_model.predict_proba()")
    if proba:
        raise ValueError("proba=True requires bb_model.predict_proba()")


def _class_one_index(bb_model):
    if not hasattr(bb_model, "classes_"):
        raise ValueError("bb_model must expose classes_ after fitting")

    matches = np.where(bb_model.classes_ == 1)[0]
    if len(matches) == 0:
        raise ValueError("bb_model.classes_ must contain class label 1 for probability-based flows")
    return int(matches[0])


def _vva_split(X, y, seed):
    _, counts = np.unique(y, return_counts=True)
    stratify = y if len(counts) > 1 and np.min(counts) > 1 else None
    return train_test_split(
        X,
        y,
        train_size=2 / 3,
        random_state=seed,
        stratify=stratify,
    )


def prelim(X, y, bb_model, wb_model, gen_name, new_size, proba=False, verbose=True, seed=2020):
    # X - np aray of feature values
    # y - binary class label from {0,1}
    # bb_model trained or not trained black-box model
    # wb_model white-box model to be fitted
    # gen_name - name of generator to use
    # new_size - size of the the dataset to be used for wb_model. Ignored for 'rerx', 'vva', 'dummy' generators
    # proba - if the black-box model should output class probabilities
    # verbose - if the function should print messages
    # seed - random seed used by stochastic generators

    X, y = check_X_y(X, y)

    if gen_name != "vva" and gen_name not in {"dummy", "rerx"} and new_size < len(y):
        raise ValueError("new_size must be at least len(y) for generators that add synthetic data")

    if not hasattr(bb_model, "classes_"):
        if verbose:
            print("fitting bb_model to data")
        bb_model.fit(X, y)

    _require_predict_proba(bb_model, gen_name, proba)
    gen = build_generator(gen_name, seed)

    if gen_name == "vva":
        class_one_ind = _class_one_index(bb_model)
        Xtrain, Xval, ytrain, yval = _vva_split(X, y, seed)
        gen.fit(Xtrain, metamodel=bb_model)

        if proba:
            wb_model.fit(Xtrain, bb_model.predict_proba(Xtrain)[:, class_one_ind])
        else:
            wb_model.fit(Xtrain, ytrain)
        sctest0 = wb_model.score(Xval, yval)
        ropt = 0

        if gen.will_generate():
            for r in np.linspace(0.5, 2.5, num=5):
                Xnew = gen.sample(r)
                if proba:
                    Xnew = np.concatenate([Xnew, Xtrain])
                    ynew = bb_model.predict_proba(Xnew)[:, class_one_ind]
                else:
                    ynew = bb_model.predict(Xnew)
                    Xnew = np.concatenate([Xnew, Xtrain])
                    ynew = np.concatenate([ynew, ytrain])

                wb_model.fit(Xnew, ynew)
                sctest = wb_model.score(Xval, yval)
                if sctest > sctest0:
                    sctest0 = sctest
                    ropt = r

        if ropt > 0:
            Xnew = gen.fit(X, metamodel=bb_model).sample(ropt)
            if proba:
                Xnew = np.concatenate([Xnew, X])
                ynew = bb_model.predict_proba(Xnew)[:, class_one_ind]
            else:
                ynew = bb_model.predict(Xnew)
                Xnew = np.concatenate([Xnew, X])
                ynew = np.concatenate([ynew, y])

            wb_model.fit(Xnew, ynew)
        else:
            wb_model.fit(X, y)
    else:
        gen.fit(X, y, metamodel=bb_model)
        Xnew = gen.sample(new_size - len(y))
        if proba:
            class_one_ind = _class_one_index(bb_model)
            if gen_name != "rerx":
                Xnew = np.concatenate([Xnew, X])
            ynew = bb_model.predict_proba(Xnew)[:, class_one_ind]
        else:
            ynew = bb_model.predict(Xnew)
            if gen_name != "rerx":
                Xnew = np.concatenate([Xnew, X])
                ynew = np.concatenate([ynew, y])

        wb_model.fit(Xnew, ynew)

    return wb_model
