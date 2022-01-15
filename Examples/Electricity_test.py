from skmultiflow.meta import AdaptiveRandomForestRegressor
from skmultiflow.trees import HoeffdingAdaptiveTreeRegressor, HoeffdingAdaptiveTreeClassifier
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data import DataStream
from src.Environment import *
from sklearn.linear_model import Ridge
import functools
import matplotlib.pyplot as plt
from skmultiflow.meta import *
from skmultiflow.bayes import NaiveBayes
from skmultiflow.data import TemporalDataStream, DataStream

if __name__ == "__main__":
    train_size = 1000

    df_in = pd.read_csv("../Data/electricity.csv", delimiter=",", header=0)
    df_in["label"] = pd.get_dummies(df_in["class"], drop_first=True)
    df_in.drop(columns=["class"], inplace=True)
    df_in = df_in.apply(pd.to_numeric, errors='coerce')
    df = df_in.copy()
    scaler = StandardScaler()
    scaler.fit(df.values)  # [:train_size]
    scaled_features = scaler.transform(df.values)
    df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)
    length = df.shape[0]
    n_dim = df.shape[1]
    df["batch"] = 0
    df["label"] = df_in["label"]
    env = OnlineEnvironment(df, train_size=train_size, length=length, problem="Classification", verbose=True)


    def warn(*args, **kwargs):
        pass


    import warnings

    warnings.warn = warn
    warnings.filterwarnings("ignore")

    likelihood_model = MultivariateNormal()
    likelihood_model_multi = MultivariateNormal()
    likelihood_model_2 = BayesianGaussianMixture(n_components=3, covariance_type='full')
    likelihood_model_3 = KernelDensity()
    likelihood_model_5 = MARFlow(n_dim, 5)

    likelihood_model_1 = MultivariateNormal()
    likelihood_model_1_multi = MultivariateNormal()
    likelihood_model_2_1 = BayesianGaussianMixture(n_components=3, covariance_type='full')
    likelihood_model_3_1 = KernelDensity()
    likelihood_model_5_1 = MARFlow(n_dim, 5)

    likelihood_model_3_bis = KernelDensity()
    likelihood_model_5_bis = MARFlow(n_dim, 5)

    k1 = 1
    window_size = 5

    weight_model_rw = WeightModel(likelihood_model=likelihood_model, reweighting_function=power_law_reweighting)
    weight_model_multi = WeightModel(likelihood_model=likelihood_model_multi, reweighting_function=multiple_reweighting)
    weight_model_rw_3 = WeightModel(likelihood_model=likelihood_model_3, reweighting_function=power_law_reweighting)
    weight_model_rw_2 = WeightModel(likelihood_model=likelihood_model_2, reweighting_function=power_law_reweighting)
    weight_model_rw_5 = WeightModel(likelihood_model=likelihood_model_5, reweighting_function=power_law_reweighting)

    weight_model_adversarial = WeightModel(likelihood_model=None, reweighting_function=adversarial_reweighting)

    weight_model_rw_3_bis = WeightModel(likelihood_model=likelihood_model_3_bis, reweighting_function=reweighting, cv=5,
                                        params={"bandwidth": np.linspace(0.05, 1, 15),
                                                "kernel": ["gaussian", "epanechnikov"]})

    weight_model_rw_5_bis = WeightModel(likelihood_model=likelihood_model_5_bis, reweighting_function=reweighting, cv=3,
                                        params={"num_layers": np.arange(2, 8), "hidden_features": np.arange(1, 10),
                                                "num_iter": np.arange(10, 100)})

    weight_model_rw_1 = WeightModel(likelihood_model=likelihood_model_1, reweighting_function=power_law_reweighting)
    weight_model_multi_1 = WeightModel(likelihood_model=likelihood_model_1, reweighting_function=multiple_reweighting)
    weight_model_rw_3_1 = WeightModel(likelihood_model=likelihood_model_3_1, reweighting_function=power_law_reweighting)
    weight_model_rw_2_1 = WeightModel(likelihood_model=likelihood_model_2_1, reweighting_function=power_law_reweighting)
    weight_model_rw_5_1 = WeightModel(likelihood_model=likelihood_model_5_1, reweighting_function=power_law_reweighting)

    last_1 = functools.partial(retraining_last_k, k1, batch_size=100)
    model_retraining_all = WeightModel(likelihood_model=None, reweighting_function=retraining_all)
    model_retraining_last_k1 = WeightModel(likelihood_model=None, reweighting_function=last_1)

    # ml_models = [LGBMRegressor()]*14
    # ml_models = [Ridge() for _ in range(14)]
    # ml_models = [RandomForestRegressor() for _ in range(14)]
    ml_models = [GaussianNB() for _ in range(14)]
    # ml_models = [LGBMClassifier() for _ in range(14)]

    threshold = 500
    min_instances = 500
    delta = 0.01 * n_dim
    alpha = 0.9999

    drift_detectors = [PageHinkley(min_instances=min_instances, delta=delta, threshold=threshold, alpha=alpha) for _ in
                       range(7)]
    # drift_detectors = [ADWIN() for _ in range(7)]

    # drift_detectors_error = [PageHinkley(min_instances=min_instances, delta=delta, threshold=threshold, alpha=alpha) for _ in range(7)]
    # drift_detectors_error = [ADWIN() for _ in range(7)]
    drift_detectors_error = [HDDM_W() for _ in range(7)]

    # delta deve dipendere da errore su training

    old_to_use = 300
    update_wm = 500

    lgbm_rw = DensityDriftModel(weight_model=weight_model_rw, ml_model=ml_models[0], name="rw_oas",
                                drift_detector=drift_detectors[0],
                                old_to_use=old_to_use, update_wm=update_wm)
    lgbm_multi_rw = DensityDriftModel(weight_model=weight_model_multi, ml_model=ml_models[1], name="multi_rw_oas",
                                      drift_detector=drift_detectors[1],
                                      old_to_use=old_to_use, update_wm=update_wm)
    lgbm_rw_3 = DensityDriftModel(weight_model=weight_model_rw_3, ml_model=ml_models[2], name="rw_kde",
                                  drift_detector=drift_detectors[2],
                                  old_to_use=old_to_use, update_wm=update_wm)
    lgbm_rw_2 = DensityDriftModel(weight_model=weight_model_rw_2, ml_model=ml_models[3], name="rw_gmm",
                                  drift_detector=drift_detectors[3],
                                  old_to_use=old_to_use, update_wm=update_wm)
    lgbm_rw_5 = DensityDriftModel(weight_model=weight_model_rw_5, ml_model=ml_models[4], name="rw_flow",
                                  drift_detector=drift_detectors[4],
                                  old_to_use=old_to_use, update_wm=update_wm)

    lgbm_rw_3_bis = DensityDriftModel(weight_model=weight_model_rw_3_bis, ml_model=ml_models[12], name="rw_kde_bis",
                                      drift_detector=drift_detectors[5],
                                      old_to_use=old_to_use, update_wm=update_wm)

    lgbm_rw_5_bis = DensityDriftModel(weight_model=weight_model_rw_5_bis, ml_model=ml_models[13], name="rw_flow_bis",
                                      drift_detector=drift_detectors[6],
                                      old_to_use=old_to_use, update_wm=update_wm)

    whiten = False
    lgbm_rw_1 = ErrorDriftModel(weight_model=weight_model_rw_1, ml_model=ml_models[5], name="rw_oas_error",
                                drift_detector=drift_detectors_error[0], old_to_use=old_to_use, update_wm=update_wm,
                                whiten=whiten)

    lgbm_multi_rw_1 = ErrorDriftModel(weight_model=weight_model_multi_1, ml_model=ml_models[6],
                                      name="multi_rw_oas_error",
                                      drift_detector=drift_detectors_error[1],
                                      old_to_use=old_to_use, update_wm=update_wm, whiten=whiten)

    lgbm_rw_3_1 = ErrorDriftModel(weight_model=weight_model_rw_3_bis, ml_model=ml_models[7], name="rw_kde_error",
                                  drift_detector=drift_detectors_error[2],
                                  old_to_use=old_to_use, update_wm=update_wm, whiten=whiten)

    lgbm_rw_2_1 = ErrorDriftModel(weight_model=weight_model_rw_2_1, ml_model=ml_models[8], name="rw_gmm_error",
                                  drift_detector=drift_detectors_error[3], old_to_use=old_to_use, update_wm=update_wm,
                                  whiten=whiten)

    lgbm_rw_5_1 = ErrorDriftModel(weight_model=weight_model_rw_5_1, ml_model=ml_models[9], name="rw_flow_error",
                                  drift_detector=drift_detectors_error[4], old_to_use=old_to_use, update_wm=update_wm,
                                  whiten=whiten)

    lgbm_all = ErrorDriftModel(weight_model=model_retraining_all, ml_model=ml_models[10], name="retraining_all",
                               drift_detector=drift_detectors_error[5], old_to_use=old_to_use, update_wm=update_wm,
                               whiten=whiten)

    lgbm_last_1 = ErrorDriftModel(weight_model=model_retraining_last_k1, ml_model=ml_models[11],
                                  name="retraining_last_1",
                                  drift_detector=drift_detectors_error[6], old_to_use=old_to_use, update_wm=update_wm,
                                  whiten=whiten)

    lgbm_adversarial = ErrorDriftModel(weight_model=weight_model_adversarial, ml_model=ml_models[11],
                                       name="rw_adversarial",
                                       drift_detector=drift_detectors_error[4],
                                       old_to_use=old_to_use, update_wm=update_wm, whiten=whiten)

    models_list = [lgbm_all, lgbm_last_1, lgbm_rw_1, lgbm_rw_3_1,
                   lgbm_rw_2_1]  # lgbm_adversarial, lgbm_rw_5_1,
    # lgbm_rw_1,  lgbm_multi_rw_1, lgbm_rw_2_1,lgbm_rw_3_1,
    # lgbm_all, lgbm_last_1,
    # , lgbm_rw_3_bis]#, lgbm_rw_5_bis], lgbm_multi_rw,

    models = dict((k.name, k) for k in models_list)

    env.run_experiment(models)
    models_meta = copy.copy(models)

    a = {}
    for model_name, model in models_meta.items():
        z = env.aucs[model_name]
        a[model_name] = [z[i] * len(z[:i + 1]) - z[i - 1] * len(z[:i]) for i in range(1, len(z))]
        print(model_name, "Done")

    models_used = []
    aucs_top = []
    aucs = []
    preq_aucs = 0
    running_aucs = dict((k, 0) for k in models_meta.keys())
    alpha = 0.5
    auc_mm_1 = metrics.ROCAUC(n_thresholds=20)
    auc_mm = metrics.ROCAUC(n_thresholds=20)

    for i in tqdm(range(1, len(env.aucs["rw_oas"]))):
        for model_name, model in models_meta.items():
            # preds = env.predictions[model_name][i-1]
            # aucs_local = mean_squared_error(df.iloc[train_size+i-1:train_size+i].iloc[:,-1:], preds, squared=False)
            aucs_local = a[model_name][i - 2]  # env.aucs[model_name][i-1]
            running_aucs[model_name] = running_aucs[model_name] * (1 - alpha) + aucs_local * alpha
        model_name_ = max(running_aucs, key=running_aucs.get)
        models_used.append(model_name_)
        # aucs_local = env.aucs[model_name_][i]
        auc_mm_1.update(df["label"][i + 1000], env.predictions[model_name_][i][0], )
        aucs_top.append(auc_mm_1.get())
        # prediction = env.predictions[model_name_][i]
        # aucs_local = mean_squared_error(df.iloc[train_size+i:train_size+i+1].iloc[:,-1:], prediction, squared=False)
        # aucs.append(aucs_local)

    print("Framework runs done")


    aucs_ozabag = []
    aucs_lbag = []
    aucs_srp = []
    aucs_oboost = []

    y_true = []
    y_pred_1 = []
    y_pred_2 = []
    y_pred_3 = []
    y_pred_4 = []

    ozabag = OzaBaggingADWINClassifier(base_estimator=NaiveBayes())
    lbag = LeveragingBaggingClassifier(base_estimator=NaiveBayes())
    srp = StreamingRandomPatchesClassifier(base_estimator=NaiveBayes())
    oboost = OnlineBoostingClassifier(base_estimator=NaiveBayes())

    auc1 = metrics.ROCAUC(n_thresholds=20)
    auc2 = metrics.ROCAUC(n_thresholds=20)
    auc3 = metrics.ROCAUC(n_thresholds=20)
    auc4 = metrics.ROCAUC(n_thresholds=20)

    initial = time.time()
    stream = DataStream(data=df.iloc[:, :-2], y=pd.DataFrame(df["label"]))

    count = 0
    while stream.has_more_samples():
        X, y = stream.next_sample()
        y_true.append(y[0])

        if count >= train_size:
            y_pred_1.append(ozabag.predict_proba(X)[:, 1][0])
            y_pred_2.append(lbag.predict_proba(X)[:, 1][0])
            y_pred_3.append(srp.predict_proba(X)[:,1][0])
            y_pred_4.append(oboost.predict_proba(X)[:, 1][0])

        if count >= train_size:
            auc1.update(y_true[-1], y_pred_1[-1])
            auc2.update(y_true[-1], y_pred_2[-1])
            auc3.update(y_true[-1], y_pred_2[-1])
            auc4.update(y_true[-1], y_pred_4[-1])
            aucs_ozabag.append(auc1.get())
            aucs_lbag.append(auc2.get())
            aucs_srp.append(auc3.get())
            aucs_oboost.append(auc4.get())

        if count >= train_size and count % 10000 == 0:
            print("Sample ", count)
            print("Oza: ", auc1.get())
            print("Lev: ", auc2.get())
            print("Srp: ", auc3.get())
            print("Boo: ", auc4.get())

        ozabag.partial_fit(X, y, classes=[0, 1])
        lbag.partial_fit(X, y, classes=[0, 1])
        srp.partial_fit(X, y, classes=[0, 1])
        oboost.partial_fit(X, y, classes=[0, 1])

        count += 1

    preds = [pd.DataFrame(env.predictions[i], columns=[i]) for i in env.predictions.keys()]
    aucs = [pd.DataFrame(env.aucs[i], columns=[i]) for i in env.aucs.keys()]
    aucs_final = pd.concat(aucs, axis=1)

    plt.rcParams["figure.figsize"] = (7, 6)
    plt.rcParams["savefig.facecolor"] = 'white'

    # exp = ["rw_oas", "multi_rw_oas", "rw_flow", "rw_kde_bis", "rw_gmm", "rw_adversarial"]
    exp = ["rw_oas_error", "Lev", "Oza", "Boo", "retraining_all", "retraining_last_1", "EWMAMM"]

    legenda = {"retraining_last_1": "LAST", "retraining_all": "ALL", "multi_rw_oas": "IWDA(Multi,Norm)",
               "rw_oas": "IWDA(PL., Norm)",
               "EWMAMM": "Expert", "rw_flow": "IWDA(PL,MAF)", "ARF": "ARF", "HRT": "HRT", "Lev": "Lev. Bag.",
               "Boo": "Onl. Boo.",
               "Oza": "OzaBag.", "rw_oas_error": "IWDA(PL,Norm)",
               "HRT": "HRT", "SRP": "SRP"}

    colors = {"retraining_last_1": "goldenrod", "retraining_all": "purple", "multi_rw_oas": "red",
              "rw_oas_error": "red",
              "EWMAMM": "slateblue", "rw_flow": "red", "Lev": "gold", "Oza": "darkslategrey", "Boo": "teal",
              "SRP": "aqua"}

    linestyles = {"retraining_last_1": (0, (2, 2, 2, 2)), "retraining_all": "dotted", "multi_rw_oas": (0, (2, 1, 1, 1)),
                  "EWMAMM": "solid", "rw_flow": (0, (2, 1, 1, 1)), "rw_oas_error": (0, (2, 1, 1, 1)),
                  "Lev": (0, (2, 1, 1, 2)), "SRP": "dashed",
                  "Oza": (0, (1, 1, 1, 2)), "Boo": (0, (0.5, 1, 0.5, 1))}

    fig, ax = plt.subplots()

    for col in exp:
        if col != "rw_oas_error":
            try:
                ax.plot(aucs_final[col][1000:-1000], label=legenda[col], color=colors[col], linestyle=linestyles[col],
                        linewidth=2.5)
            except:
                pass

    ax.plot(aucs_ozabag[1000:], label="OzaBagging", color=colors["Oza"], linestyle=linestyles["Oza"], linewidth=2.5)
    ax.plot(aucs_oboost[1000:], label="OnlineBoosting", color=colors["Boo"], linestyle=linestyles["Boo"], linewidth=2.5)
    ax.plot(aucs_lbag[1000:], label="LeveragingBagging", color=colors["Lev"], linestyle=linestyles["Lev"],
            linewidth=2.5)
    ax.plot(aucs_srp[1000:], label="SRP", color=colors["SRP"], linestyle=linestyles["SRP"], linewidth=2.5)

    for col in exp:
        if col == "rw_oas_error":
            try:
                ax.plot(aucs_final[col][1000:-1000], label=legenda[col], color=colors[col], linestyle=linestyles[col],
                        linewidth=2.5)
            except:
                pass

    ax.plot(aucs_top[1000:], label="Expert", color=colors["EWMAMM"], linestyle=linestyles["EWMAMM"], linewidth=2.5)

    plt.rc('font', family='Times New Roman')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    plt.legend(loc="lower center", bbox_to_anchor=(0.3, 0), prop={'size': 12})
    plt.xlabel("t", size=16)
    plt.ylabel("AUC", size=14)
    plt.margins(0.001)
    plt.savefig("Plots/Elec_cu_arf.pdf", bbox_inches='tight')

# aggiungere gli altri modelli competitor + controllare esperto così!
