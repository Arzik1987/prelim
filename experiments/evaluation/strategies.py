import time

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from .scoring import (
    GENERATED_TREE_ALIASES,
    SD_MODEL_NAMES,
    fidelity_score,
    fit_score_classifier,
    fit_score_sd_model,
    get_standard_sd_models,
    get_supervised_models,
    get_vva_models,
    model_size,
)
from prelim.generators.vva import Gen_vva
from ..results.artifacts import write_meta, write_result


def fit_generators_and_metamodels(state, filetme, is_balanced_metamodel):
    for generator in state.generators:
        start = time.time()
        generator.fit(state.X, state.y)
        end = time.time()
        write_meta(filetme, generator.my_name() + "time", end - start)

    for meta_model in state.all_metamodels:
        start = time.time()
        meta_model.fit(state.X, state.y)
        end = time.time()
        write_meta(filetme, meta_model.my_name() + "time", end - start)
        write_meta(filetme, meta_model.my_name() + "acccv", meta_model.fit_score())

        ypredtest = meta_model.predict(state.Xtest)
        write_meta(filetme, meta_model.my_name() + "fid", fidelity_score(ypredtest, state.ydeftest))
        write_meta(filetme, meta_model.my_name() + "acc", accuracy_score(state.ytest, ypredtest))
        write_meta(filetme, meta_model.my_name() + "bac", balanced_accuracy_score(state.ytest, ypredtest))

        for name, model in get_supervised_models(
            meta_model,
            state.tree_models,
            state.balanced_tree_models,
            state.rule_models,
            state.dtvalold,
            state.dtvalbold,
            is_balanced_metamodel,
        ):
            write_meta(filetme, meta_model.my_name() + name + "fid", fidelity_score(model.predict(state.Xtest), ypredtest))


def evaluate_rerx(state, fileres, is_balanced_metamodel):
    for meta_model in state.all_metamodels:
        state.genrerx.fit(state.X, state.y, meta_model)
        ypredtest = meta_model.predict(state.Xtest)
        Xnew = state.genrerx.sample()
        ynew = meta_model.predict(Xnew)

        for name, model in get_supervised_models(
            meta_model,
            state.tree_models,
            state.balanced_tree_models,
            state.rule_models,
            state.dtval,
            state.dtvalb,
            is_balanced_metamodel,
        ):
            score = fit_score_classifier(model, Xnew, ynew, state.X, state.y, state.Xtest, state.ytest)
            write_result(
                fileres,
                name,
                "rerx",
                meta_model.my_name(),
                score["train"],
                score["test"],
                model_size(name, model),
                score["elapsed"],
                fidelity_score(model.predict(state.Xtest), ypredtest),
                score["bactest"],
            )

        if not is_balanced_metamodel(meta_model):
            ynew = meta_model.predict_proba(Xnew)
            for name, model in get_standard_sd_models(state.primcv, state.bicv):
                score = fit_score_sd_model(model, Xnew, ynew, state.X, state.y, state.Xtest, state.ytest)
                write_result(
                    fileres,
                    name,
                    "rerx",
                    meta_model.my_name(),
                    score["train"],
                    score["test"],
                    model_size(name, model),
                    score["elapsed"],
                    "na",
                    "na",
                )


def evaluate_vva(config, state, fileres, filetme, is_balanced_metamodel):
    for meta_model in state.all_metamodels:
        ntrain = int(np.ceil(state.X.shape[0] * 2 / 3))
        Xtrain = state.X[:ntrain, :].copy()
        Xval = state.X[ntrain:, :].copy()
        ytrain = state.y[:ntrain].copy()
        yval = state.y[ntrain:].copy()
        start = time.time()
        state.genvva.fit(Xtrain, meta_model)
        end = time.time()
        write_meta(filetme, meta_model.my_name() + "vva", end - start)
        ypredtest = meta_model.predict(state.Xtest)

        for name, model in get_vva_models(
            meta_model,
            state.tree_models,
            state.balanced_tree_models,
            state.rule_models,
            state.dtval,
            state.dtvalb,
            state.primcv,
            state.bicv,
            is_balanced_metamodel,
        ):
            start = time.time()
            if name in SD_MODEL_NAMES:
                model.fit(Xtrain, meta_model.predict_proba(Xtrain))
            else:
                model.fit(Xtrain, ytrain)
            sctest0 = model.score(Xval, yval)
            ropt = 0

            if state.genvva.will_generate():
                for r in config.vva_grid:
                    Xnew = state.genvva.sample(r)
                    ynew = meta_model.predict(Xnew)
                    Xnew = np.concatenate([Xnew, Xtrain])
                    ynew = np.concatenate([ynew, ytrain])
                    if name in SD_MODEL_NAMES:
                        model.fit(Xnew, meta_model.predict_proba(Xnew))
                    else:
                        model.fit(Xnew, ynew)
                    sctest = model.score(Xval, yval)
                    if sctest > sctest0:
                        sctest0 = sctest
                        ropt = r

            end = time.time()
            write_meta(filetme, name + meta_model.my_name() + "vvaopt", end - start)
            write_meta(filetme, name + meta_model.my_name() + "ropt", ropt)

            start = time.time()
            if ropt > 0:
                Xnew = Gen_vva().fit(state.X, meta_model).sample(ropt)
                ynew = meta_model.predict(Xnew)
                Xnew = np.concatenate([Xnew, state.X])
                ynew = np.concatenate([ynew, state.y])
            else:
                Xnew = state.X.copy()
                ynew = state.y.copy()
            end = time.time()
            write_meta(filetme, name + meta_model.my_name() + "vvagen", end - start)

            if name in SD_MODEL_NAMES:
                score = fit_score_sd_model(model, Xnew, meta_model.predict_proba(Xnew), state.X, state.y, state.Xtest, state.ytest)
                fidelity = "na"
                bactest = "na"
            else:
                score = fit_score_classifier(model, Xnew, ynew, state.X, state.y, state.Xtest, state.ytest)
                fidelity = fidelity_score(model.predict(state.Xtest), ypredtest)
                bactest = score["bactest"]

            write_result(
                fileres,
                name,
                "vva",
                meta_model.my_name(),
                score["train"],
                score["test"],
                model_size(name, model),
                score["elapsed"],
                fidelity,
                bactest,
            )


def evaluate_standard_generator(generator, Xgen, meta_model, state, config, fileres, filetme):
    ypredtest = meta_model.predict(state.Xtest)

    start = time.time()
    Xnew = Xgen.copy()
    predicted_labels = meta_model.predict(Xnew)
    end = time.time()
    write_meta(filetme, generator.my_name() + meta_model.my_name(), end - start)

    if config.include_generated_only_tree_models:
        for name, model in [("dt", state.tree_models["dt"]), ("dtc", state.tree_models["dtc"]), ("dtval", state.dtval)]:
            score = fit_score_classifier(model, Xnew, predicted_labels, state.X, state.y, state.Xtest, state.ytest)
            write_result(
                fileres,
                GENERATED_TREE_ALIASES[name],
                generator.my_name(),
                meta_model.my_name(),
                score["train"],
                score["test"],
                model_size(name, model),
                score["elapsed"],
                fidelity_score(model.predict(state.Xtest), ypredtest),
                score["bactest"],
            )

    Xnew = Xnew[: config.generated_sample_size - len(state.y), :]
    predicted_labels = predicted_labels[: config.generated_sample_size - len(state.y)]
    Xnew = np.concatenate([state.X, Xnew])
    predicted_labels = np.concatenate([state.y, predicted_labels])

    for name, model in [("dt", state.tree_models["dt"]), ("dtc", state.tree_models["dtc"]), ("dtval", state.dtval)]:
        score = fit_score_classifier(model, Xnew, predicted_labels, state.X, state.y, state.Xtest, state.ytest)
        write_result(
            fileres,
            name,
            generator.my_name(),
            meta_model.my_name(),
            score["train"],
            score["test"],
            model_size(name, model),
            score["elapsed"],
            fidelity_score(model.predict(state.Xtest), ypredtest),
            score["bactest"],
        )

    Xnew = Xnew[: config.rules_sample_size, :]
    predicted_labels = predicted_labels[: config.rules_sample_size]

    for name, model in state.rule_models.items():
        score = fit_score_classifier(model, Xnew, predicted_labels, state.X, state.y, state.Xtest, state.ytest)
        write_result(
            fileres,
            name,
            generator.my_name(),
            meta_model.my_name(),
            score["train"],
            score["test"],
            model_size(name, model),
            score["elapsed"],
            fidelity_score(model.predict(state.Xtest), ypredtest),
            score["bactest"],
        )

    ynew = meta_model.predict_proba(Xnew)
    for name, model in get_standard_sd_models(state.primcv, state.bicv):
        score = fit_score_sd_model(model, Xnew, ynew, state.X, state.y, state.Xtest, state.ytest)
        write_result(
            fileres,
            name,
            generator.my_name(),
            meta_model.my_name(),
            score["train"],
            score["test"],
            model_size(name, model),
            score["elapsed"],
            "na",
            "na",
        )


def evaluate_balanced_generator(generator, Xgen, meta_model, state, config, fileres, filetme):
    ypredtest = meta_model.predict(state.Xtest)

    start = time.time()
    Xnew = Xgen[: config.generated_sample_size - len(state.y), :].copy()
    ynew = meta_model.predict(Xnew)
    Xnew = np.concatenate([state.X, Xnew])
    ynew = np.concatenate([state.y, ynew])
    end = time.time()
    write_meta(filetme, generator.my_name() + meta_model.my_name(), end - start)

    for name, model in list(state.balanced_tree_models.items()) + [("dtvalb", state.dtvalb)]:
        score = fit_score_classifier(model, Xnew, ynew, state.X, state.y, state.Xtest, state.ytest)
        write_result(
            fileres,
            name,
            generator.my_name(),
            meta_model.my_name(),
            score["train"],
            score["test"],
            model_size(name, model),
            score["elapsed"],
            fidelity_score(model.predict(state.Xtest), ypredtest),
            score["bactest"],
        )


def evaluate_sampled_generators(config, state, fileres, filetme):
    for generator in state.generators:
        start = time.time()
        Xgen = generator.sample(config.generated_sample_size)
        end = time.time()
        write_meta(filetme, generator.my_name() + "gen", end - start)

        for meta_model in state.standard_metamodels:
            evaluate_standard_generator(generator, Xgen, meta_model, state, config, fileres, filetme)
        for meta_model in state.balanced_metamodels:
            evaluate_balanced_generator(generator, Xgen, meta_model, state, config, fileres, filetme)


def _evaluate_ssl_mode(state, fileres, is_balanced_metamodel, Xtest, ytest, Xgen, pool_labels, mode_name):
    for meta_model in state.all_metamodels:
        ypredtest = meta_model.predict(Xtest)
        ynew = np.concatenate([pool_labels(meta_model, Xgen), state.y])
        Xnew = np.concatenate([Xgen, state.X])

        for name, model in get_supervised_models(
            meta_model,
            state.tree_models,
            state.balanced_tree_models,
            state.rule_models,
            state.dtval,
            state.dtvalb,
            is_balanced_metamodel,
        ):
            score = fit_score_classifier(model, Xnew, ynew, state.X, state.y, Xtest, ytest)
            write_result(
                fileres,
                name,
                mode_name,
                meta_model.my_name(),
                score["train"],
                score["test"],
                model_size(name, model),
                score["elapsed"],
                fidelity_score(model.predict(Xtest), ypredtest),
                score["bactest"],
            )

        if not is_balanced_metamodel(meta_model):
            ynew = meta_model.predict_proba(Xnew)
            for name, model in get_standard_sd_models(state.primcv, state.bicv):
                score = fit_score_sd_model(model, Xnew, ynew, state.X, state.y, Xtest, ytest)
                write_result(
                    fileres,
                    name,
                    mode_name,
                    meta_model.my_name(),
                    score["train"],
                    score["test"],
                    model_size(name, model),
                    score["elapsed"],
                    "na",
                    "na",
                )


def evaluate_ssl(config, state, fileres, is_balanced_metamodel, get_new_test):
    Xtest, ytest, Xgen, ygen_true = get_new_test(
        state.Xtest,
        state.ytest,
        len(state.y),
        new_size=config.ssl_pool_size,
    )
    _evaluate_ssl_mode(
        state,
        fileres,
        is_balanced_metamodel,
        Xtest,
        ytest,
        Xgen,
        lambda meta_model, Xpool: meta_model.predict(Xpool),
        "ssl",
    )
    _evaluate_ssl_mode(
        state,
        fileres,
        is_balanced_metamodel,
        Xtest,
        ytest,
        Xgen,
        lambda meta_model, Xpool: ygen_true,
        "ssl_oracle",
    )
