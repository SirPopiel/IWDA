from skmultiflow.meta import AdaptiveRandomForestRegressor
from skmultiflow.trees import HoeffdingAdaptiveTreeRegressor
from skmultiflow.data import DataStream
from src.Environment import *
from sklearn.linear_model import Ridge
import functools
import matplotlib.pyplot as plt

if __name__ == "__main__":
    train_size = 1000

    df_in = pd.read_csv("../Data/AirQualityUCI.csv", delimiter=";")
    df_in = df_in[:9357]
    df_in = df_in.drop(columns=["Date", "Unnamed: 15", "Unnamed: 16"])
    df_in["Time"] = pd.to_datetime(df_in["Time"], format="%H.%M.%S").dt.hour
    df_in[df_in.columns[df_in.dtypes.eq('object')]] = df_in[df_in.columns[df_in.dtypes.eq('object')]].\
        apply(lambda a: a.str.replace(',', '.')).apply(pd.to_numeric)
    to_predict = "C6H6(GT)"
    df_in = df_in[df_in[to_predict] != -200]  # there are missing data encoded at -200
    df = df_in.copy()
    df["label"] = df[to_predict]
    df = df.loc[:, ~df.columns.str.contains('GT')]
    df[df.columns[df.dtypes.eq('object')]] = df[df.columns[df.dtypes.eq('object')]].\
        apply(lambda a: a.str.replace(',', '.')).apply(pd.to_numeric)
    df = df.reset_index()
    df.drop(columns="index", inplace=True)
    df_unscaled = df.copy()
    scaler = StandardScaler()
    scaler.fit(df[:train_size].values)  # train size for normalization on only train!!!!
    scaled_features = scaler.transform(df.values)
    df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)

    length = len(df)
    n_dim = df.shape[1]

    df["batch"] = 0
    env = OnlineEnvironment(df, train_size=train_size, length=length, problem="Regression", verbose=True)


    def warn(*args, **kwargs):
        pass

    import warnings

    warnings.filterwarnings("ignore")
    warnings.warn = warn

    likelihood_model = MultivariateNormal()
    likelihood_model_multi = MultivariateNormal()
    likelihood_model_2 = BayesianGaussianMixture(n_components=3, covariance_type='full')
    likelihood_model_3 = KernelDensity()
    likelihood_model_5 = MARFlow(n_dim, 5)
    likelihood_model_3_bis = KernelDensity()

    k1 = 1
    batch_size = None

    weight_model_rw = WeightModel(likelihood_model=likelihood_model, reweighting_function=power_law_reweighting)
    weight_model_multi = WeightModel(likelihood_model=likelihood_model_multi, reweighting_function=multiple_reweighting)
    weight_model_rw_3 = WeightModel(likelihood_model=likelihood_model_3, reweighting_function=power_law_reweighting)
    weight_model_rw_2 = WeightModel(likelihood_model=likelihood_model_2, reweighting_function=power_law_reweighting)
    weight_model_rw_5 = WeightModel(likelihood_model=likelihood_model_5, reweighting_function=power_law_reweighting)
    weight_model_adversarial = WeightModel(likelihood_model=None, reweighting_function=adversarial_reweighting)

    weight_model_rw_3_bis = WeightModel(likelihood_model=likelihood_model_3_bis, reweighting_function=reweighting, cv=5,
                                        params={"bandwidth": np.linspace(0.05, 1, 15),
                                                "kernel": ["gaussian", "epanechnikov"]})

    last_1 = functools.partial(retraining_last_k, k1, batch_size)
    model_retraining_all = WeightModel(likelihood_model=None, reweighting_function=retraining_all)
    model_retraining_last_k1 = WeightModel(likelihood_model=None, reweighting_function=last_1)

    ml_models = [Ridge() for _ in range(14)]

    threshold = 100
    min_instances = 100
    delta = 0.001 * n_dim
    alpha = 0.9999

    drift_detectors = [PageHinkley(min_instances=min_instances, delta=delta, threshold=threshold, alpha=alpha) for _ in
                       range(7)]
    # drift_detectors = [ADWIN() for _ in range(7)]

    drift_detectors_error = [
        PageHinkley(min_instances=min_instances, delta=delta, threshold=threshold / 2, alpha=alpha) for _ in range(3)]
    # drift_detectors_error = [ADWIN() for _ in range(7)]

    old_to_use = 500
    update_wm = 250
    verbose = True

    lgbm_rw = DensityDriftModel(weight_model=weight_model_rw, train_size=train_size, ml_model=ml_models[0],
                                name="rw_oas",
                                drift_detector=drift_detectors[0],
                                old_to_use=old_to_use, update_wm=update_wm)
    lgbm_multi_rw = DensityDriftModel(weight_model=weight_model_multi, train_size=train_size, ml_model=ml_models[1],
                                      name="multi_rw_oas",
                                      drift_detector=drift_detectors[1],
                                      old_to_use=old_to_use, update_wm=update_wm)
    lgbm_rw_3 = DensityDriftModel(weight_model=weight_model_rw_3, train_size=train_size, ml_model=ml_models[2],
                                  name="rw_kde",
                                  drift_detector=drift_detectors[2],
                                  old_to_use=old_to_use, update_wm=update_wm)
    lgbm_rw_2 = DensityDriftModel(weight_model=weight_model_rw_2, train_size=train_size, ml_model=ml_models[3],
                                  name="rw_gmm",
                                  drift_detector=drift_detectors[3],
                                  old_to_use=old_to_use, update_wm=update_wm)
    lgbm_rw_5 = DensityDriftModel(weight_model=weight_model_rw_5, train_size=train_size, ml_model=ml_models[4],
                                  name="rw_flow",
                                  drift_detector=drift_detectors[4],
                                  old_to_use=old_to_use, update_wm=update_wm)

    lgbm_rw_3_bis = DensityDriftModel(weight_model=weight_model_rw_3_bis, train_size=train_size, ml_model=ml_models[5],
                                      name="rw_kde_bis", drift_detector=drift_detectors[5],
                                      old_to_use=old_to_use, update_wm=update_wm)

    lgbm_all = ErrorDriftModel(weight_model=model_retraining_all, train_size=train_size, ml_model=ml_models[6],
                               name="retraining_all",
                               drift_detector=drift_detectors_error[0], old_to_use=old_to_use, update_wm=update_wm,
                               verbose=verbose)

    lgbm_last_1 = ErrorDriftModel(weight_model=model_retraining_last_k1, train_size=train_size,  ml_model=ml_models[7],
                                  name="retraining_last_1",
                                  drift_detector=drift_detectors_error[1], old_to_use=old_to_use, update_wm=update_wm,
                                  verbose=verbose)

    lgbm_adversarial = ErrorDriftModel(weight_model=weight_model_adversarial, train_size=train_size,
                                       ml_model=ml_models[8], name="rw_adversarial",
                                       drift_detector=drift_detectors_error[2],
                                       old_to_use=old_to_use, update_wm=update_wm, verbose=verbose)

    models_list = [lgbm_rw, lgbm_multi_rw, lgbm_rw_2, lgbm_rw_3,
                   lgbm_all, lgbm_last_1, lgbm_rw_5,
                   lgbm_adversarial, lgbm_rw_3_bis]

    models = dict((k.name, k) for k in models_list)

    env.run_experiment(models)

    print("Framework runs done")

    initial = time.time()
    rmse_arf = []
    rmse_tree = []
    y_true = []
    y_pred = []
    y_pred_tree = []
    arf_reg = AdaptiveRandomForestRegressor(random_state=123456)
    tree = HoeffdingAdaptiveTreeRegressor(random_state=123456)

    stream = DataStream(data=df_unscaled.iloc[:, :-1], y=pd.DataFrame(df_unscaled.iloc[:, -1]))

    count = 0
    while stream.has_more_samples():
        X, y = stream.next_sample()
        y_true.append(y[0])
        y_pred.append(arf_reg.predict(X)[0])
        y_pred_tree.append(tree.predict(X)[0])

        if count >= train_size:
            rmse_arf.append(mean_squared_error([y_true[-1]], [y_pred[-1]], squared=False))
            rmse_tree.append(mean_squared_error([y_true[-1]], [y_pred_tree[-1]], squared=False))
        if count > train_size and count % 1000 == 0:
            print("ARF: ", np.mean(rmse_arf))
            print("Tree: ", np.mean(rmse_tree))
        arf_reg.partial_fit(X, y)
        tree.partial_fit(X, y)
        count += 1

    rmse = [pd.DataFrame(env.rmse[i], columns=[i]) for i in env.rmse.keys()]
    rmse_final = pd.concat(rmse, axis=1)
    rmse_cum = rmse_final.cumsum().div(np.cumsum(np.ones(df.shape[0] - train_size)), axis=0)


    def invLabel(scaler, data, idx=-1):
        dummy = pd.DataFrame(np.zeros((len(data), scaler.n_features_in_)))
        dummy.iloc[:, idx] = data
        dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=dummy.columns)
        return dummy.iloc[:, idx].values


    rmse_nonnorm = dict((k, 0) for k in env.rmse.keys())
    for col in rmse_cum:
        for i in tqdm(range(len(env.predictions[col]))):
            rmse_nonnorm[col] = np.append(rmse_nonnorm[col],
                                          mean_squared_error(df_in[train_size + i:train_size + i + 1]["C6H6(GT)"],
                                                             invLabel(scaler, np.array(env.predictions[col][i]), -1),
                                                             squared=False))

    rmse_nonnorm.update({"ARF": np.append(0, rmse_arf)})
    rmse_nonnorm.update({"HRT": np.append(0, rmse_tree)})
    rmse_nn = [pd.DataFrame(rmse_nonnorm[i], columns=[i]) for i in env.rmse.keys()]
    rmse_final_nn = pd.concat(rmse_nn, axis=1)
    rmse_cum_nn = rmse_final_nn.cumsum().div(np.cumsum(np.ones(df.shape[0] - train_size + 1)), axis=0)
    models_meta = copy.copy(models)
    # comment here if you want to exclude some models from the expert!
    # the expert is built afterwards for efficiency, but it is conceptually the same as building it online
    # for e in ["rw_oas_error", "multi_rw_oas_error", "rw_gmm_error", "rw_kde_error",
    # "rw_oas", "multi_rw_oas",,"retraining_all","retraining_last_1","rw_kde", "rw_flow",
    # "rw_kde_bis", "rw_gmm"
    #    models_meta.pop(e)
    models_used = []
    rmse_ewmamm = []
    preq_rmse = 0
    running_rmse = dict((k, 0) for k in models_meta.keys())
    alpha = 0.5
    for i in tqdm(range(0, len(env.predictions["rw_oas"]))):
        for model_name, model in models_meta.items():
            # preds = env.predictions[model_name][i-1]
            # rmse_local = mean_squared_error(df.iloc[train_size+i-1:train_size+i].iloc[:,-1:], preds, squared=False)
            rmse_local = rmse_nonnorm[model_name][i - 1]
            running_rmse[model_name] = running_rmse[model_name] * (1 - alpha) + rmse_local * alpha
        model_name_ = min(running_rmse, key=running_rmse.get)
        models_used.append(model_name_)
        rmse_local = rmse_nonnorm[model_name_][i]
        # prediction = env.predictions[model_name_][i]
        # rmse_local = mean_squared_error(df.iloc[train_size+i:train_size+i+1].iloc[:,-1:], prediction, squared=False)
        rmse_ewmamm.append(rmse_local)
        ewmamm_p = pd.DataFrame(rmse_ewmamm)
        ewmamm = pd.DataFrame(rmse_ewmamm).cumsum().div(np.cumsum(np.ones(df.shape[0] - train_size)), axis=0)
        legenda = {"retraining_last_1": "LAST1", "retraining_all": "ALL", "multi_rw_oas": "M-IW-Norm",
                   "rw_oas": "IW-Norm",
                   "rw_gmm": "IW-GMM", "rw_flow": "IW-MAF",
                   "rw_kde": "IW-KDE", "rw_kde_bis": "IW-KDE2", "rw_oas_error": "IW-Norm-Error",
                   "multi_rw_oas_error": "M-IW-Norm-Error", "rw_gmm_error": "IW-GMM-Error", "rw_adversarial": "Prob",
                   "ARF": "ARF", "HRT": "HRT"}

        plt.style.use('seaborn')
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["savefig.facecolor"] = 'white'

        exp = ["rw_gmm", "HRT", "ARF"]

        fig, ax = plt.subplots()
        for col in rmse_final_nn.columns:  # rank[:3].index:
            if col in exp:  # or col=="retraining_last_1" or col=="retraining_all":
                ax.plot(rmse_final_nn[col].rolling(500).mean(),
                        label=legenda[col] + ": " + str(round(rmse_cum_nn[col][-1:], 3).values[0]))

        ax.plot(ewmamm_p.rolling(500).mean(), label="Expert" + ": " + str(round(ewmamm[-1:].values[0][0], 3)),
                color="k")

        plt.legend(loc="center left", bbox_to_anchor=(1, 0.7), prop={'size': 12})
        plt.xlabel("Time", size=16)
        plt.ylabel("Rolling Window RMSE", size=16)
        plt.title("AirQuality rolling RMSE", size=16)
        plt.show()
        #plt.savefig("Plots/AirQuality_pu_arf.png", bbox_inches='tight')

        fig, ax = plt.subplots()
        for col in rmse_final_nn.columns:  # rank[:3].index:
            if col in exp:
                ax.plot(rmse_cum_nn[col][500:],
                        label=legenda[col] + ": " + str(round(rmse_cum_nn[col][-1:], 3).values[0]))
        ax.plot(ewmamm[500:], label="Expert" + ": " + str(round(ewmamm[-1:].values[0][0], 3)), color="k")

        plt.legend(loc="center left", bbox_to_anchor=(1, 0.7), prop={'size': 12})
        plt.xlabel("Time", size=16)
        plt.ylabel("Cumulative RMSE", size=16)
        plt.title("AirQuality cumulative RMSE", size=16)
        plt.show()

