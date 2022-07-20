from skmultiflow.meta import AdaptiveRandomForestRegressor
from skmultiflow.trees import HoeffdingAdaptiveTreeRegressor
from skmultiflow.data import DataStream
from src.Environment import *
from sklearn.linear_model import Ridge
import functools
import matplotlib.pyplot as plt
from river import ensemble, metrics, evaluate, stream, compat, linear_model, ensemble, neighbors, neural_net, optim, expert, preprocessing
import itertools

if __name__ == "__main__":
    train_size = 2000

    df_in = pd.read_csv('Data/TimeSeries/como_data_1946_2011.txt', sep='  ')
    df_sup = pd.read_csv('Data/TimeSeries/comoDemand.txt', sep='  ', header=None)
    df_sup.reset_index(drop=False, inplace=True)
    df_sup.rename(columns={0:'demand'}, inplace=True)
    df_in['dayoftheyear'] = pd.to_datetime(df_in[['year', 'month', 'day']]).dt.dayofyear
    df_in = pd.merge(df_in, df_sup, left_on=['dayoftheyear'], right_on=['index']).drop(columns=['index','dayoftheyear'])
    scaler = StandardScaler()
    scaler.fit(df.values)
    scaled_features = scaler.transform(df.values)
    df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)
    df["label"] = df['release']
    df.drop(columns=['release','month','day','year'], inplace=True)

    length=df.shape[0]
    n_dim=df.shape[1]

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

    likelihood_model_1 = MultivariateNormal()
    likelihood_model_1_multi = MultivariateNormal()
    likelihood_model_2_1 = BayesianGaussianMixture(n_components=3, covariance_type='full')
    likelihood_model_3_1 = KernelDensity()
    likelihood_model_5_1 = MARFlow(n_dim, 3)

    likelihood_model_3_bis = KernelDensity()
    likelihood_model_5_bis = MARFlow(n_dim, 3)

    k1 = 1
    window_size = 5

    weight_model_rw = WeightModel(likelihood_model=likelihood_model, reweighting_function=power_law_reweighting)
    weight_model_multi = WeightModel(likelihood_model=likelihood_model_multi, reweighting_function=multiple_reweighting )
    weight_model_rw_3 = WeightModel(likelihood_model=likelihood_model_3, reweighting_function=power_law_reweighting)
    weight_model_rw_2 = WeightModel(likelihood_model=likelihood_model_2, reweighting_function=power_law_reweighting)
    weight_model_rw_5 = WeightModel(likelihood_model=likelihood_model_5, reweighting_function=power_law_reweighting)


    adversarial_reweighting_oob = functools.partial(adversarial_reweighting, oob=True)

    weight_model_adversarial = WeightModel(likelihood_model=None, reweighting_function=adversarial_reweighting)

    weight_model_adversarial_oob = WeightModel(likelihood_model=None, reweighting_function=adversarial_reweighting_oob)

    weight_model_rw_3_bis = WeightModel(likelihood_model=likelihood_model_3_bis, reweighting_function=reweighting, cv=5,
                                        params={"bandwidth": np.linspace(0.05, 1, 15),
                                                "kernel": ["gaussian", "epanechnikov"]})

    weight_model_rw_5_bis = WeightModel(likelihood_model=likelihood_model_5_bis, reweighting_function=reweighting, cv=3,
                                        params={"num_layers": np.arange(2, 8), "hidden_features": np.arange(1, 10),
                                                "num_iter": np.arange(10, 100)})

    weight_model_rw_1 = WeightModel(likelihood_model=likelihood_model_1, reweighting_function=power_law_reweighting)
    weight_model_multi_1 = WeightModel(likelihood_model=likelihood_model_1, reweighting_function=multiple_reweighting )
    weight_model_rw_3_1 = WeightModel(likelihood_model=likelihood_model_3_bis, reweighting_function=power_law_reweighting)
    weight_model_rw_2_1 = WeightModel(likelihood_model=likelihood_model_2_1, reweighting_function=power_law_reweighting)
    weight_model_rw_5_1 = WeightModel(likelihood_model=likelihood_model_5_1, reweighting_function=power_law_reweighting)

    last_1 = functools.partial(retraining_last_k, k1, batch_size)
    model_retraining_all = WeightModel(likelihood_model=None, reweighting_function=retraining_all)
    model_retraining_last_k1 = WeightModel(likelihood_model=None, reweighting_function=last_1)

    #ml_models = [LGBMRegressor()]*14
    ml_models = [Ridge() for _ in range(14)]
    #ml_models = [DecisionTreeRegressor()]*14
    #ml_models = [RandomForestRegressor()]*14

    threshold = 30
    min_instances = 30
    delta = 0.001*n_dim
    alpha = 0.9999

    drift_detectors = [PageHinkley(min_instances=min_instances, delta=delta, threshold=threshold, alpha=alpha) for _ in range(7)]
    #drift_detectors = [ADWIN() for _ in range(7)]

    drift_detectors_error = [PageHinkley(min_instances=min_instances, delta=delta, threshold=threshold/2, alpha=alpha) for _ in range(7)]
    #drift_detectors_error = [ADWIN() for _ in range(7)]

    # delta deve dipendere da errore su training

    old_to_use = 100
    update_wm = 80

    verbose=False

    lgbm_rw = DensityDriftModel(weight_model=weight_model_rw, ml_model=ml_models[0], name="rw_oas", drift_detector=drift_detectors[0],
                                old_to_use=old_to_use, update_wm=update_wm, verbose=verbose)
    lgbm_multi_rw = DensityDriftModel(weight_model=weight_model_multi, ml_model=ml_models[1], name="multi_rw_oas", drift_detector=drift_detectors[1],
                                      old_to_use=old_to_use, update_wm=update_wm, verbose=verbose)
    lgbm_rw_3 = DensityDriftModel(weight_model=weight_model_rw_3, ml_model=ml_models[2], name="rw_kde", drift_detector=drift_detectors[2],
                                  old_to_use=old_to_use, update_wm=update_wm, verbose=verbose)
    lgbm_rw_2 = DensityDriftModel(weight_model=weight_model_rw_2, ml_model=ml_models[3], name="rw_gmm", drift_detector=drift_detectors[3],
                                  old_to_use=old_to_use, update_wm=update_wm, verbose=verbose)
    lgbm_rw_5 = DensityDriftModel(weight_model=weight_model_rw_5, ml_model=ml_models[4], name="rw_flow", drift_detector=drift_detectors[4],
                                  old_to_use=old_to_use, update_wm=update_wm, verbose=verbose)

    lgbm_rw_3_bis = DensityDriftModel(weight_model=weight_model_rw_3_bis, ml_model=ml_models[12], name="rw_kde_bis", drift_detector=drift_detectors[5],
                                      old_to_use=old_to_use, update_wm=update_wm, verbose=verbose)

    lgbm_rw_5_bis = DensityDriftModel(weight_model=weight_model_rw_5_bis, ml_model=ml_models[13], name="rw_flow_bis", drift_detector=drift_detectors[6],
                                      old_to_use=old_to_use, update_wm=update_wm, verbose=verbose)

    lgbm_rw_1 = ErrorDriftModel(weight_model=weight_model_rw_1, ml_model=ml_models[5], name="rw_oas_error",
                                drift_detector=drift_detectors_error[0], old_to_use =old_to_use, update_wm=update_wm)

    lgbm_multi_rw_1 = ErrorDriftModel(weight_model=weight_model_multi_1, ml_model=ml_models[6], name="multi_rw_oas_error",
                                      drift_detector=drift_detectors_error[1],
                                      old_to_use=old_to_use, update_wm=update_wm, verbose=verbose)

    lgbm_rw_3_1 = ErrorDriftModel(weight_model=weight_model_rw_3_1, ml_model=ml_models[7], name="rw_kde_error",
                                  drift_detector=drift_detectors_error[2],
                                  old_to_use=old_to_use, update_wm=update_wm, verbose=verbose)

    lgbm_rw_2_1 = ErrorDriftModel(weight_model=weight_model_rw_2_1, ml_model=ml_models[8], name="rw_gmm_error",
                                  drift_detector=drift_detectors_error[3], old_to_use =old_to_use, update_wm=update_wm, verbose=verbose)

    lgbm_rw_5_1 = ErrorDriftModel(weight_model=weight_model_rw_5_1, ml_model=ml_models[9], name="rw_flow_error",
                                  drift_detector=drift_detectors_error[4], old_to_use =old_to_use, update_wm=update_wm, verbose=verbose)

    lgbm_all = ErrorDriftModel(weight_model=model_retraining_all, ml_model=ml_models[10], name="retraining_all",
                                  drift_detector=drift_detectors_error[5], old_to_use =old_to_use, update_wm=update_wm, verbose=verbose)

    lgbm_last_1 = ErrorDriftModel(weight_model=model_retraining_last_k1, ml_model=ml_models[11], name="retraining_last_1",
                                  drift_detector=drift_detectors_error[6], old_to_use =old_to_use, update_wm=update_wm, verbose=verbose)

    lgbm_adversarial = ErrorDriftModel(weight_model=weight_model_adversarial, ml_model=ml_models[8], name="rw_adversarial",
                                      drift_detector=drift_detectors_error[4],
                                       old_to_use=old_to_use, update_wm=update_wm, verbose=verbose)

    lgbm_adversarial_oob = ErrorDriftModel(weight_model=weight_model_adversarial_oob, ml_model=ml_models[3], name="rw_adversarial_oob",
                                      drift_detector=drift_detectors_error[3],
                                       old_to_use=old_to_use, update_wm=update_wm, verbose=verbose)


    #lgbm_retrain_all = DensityDriftModel(weight_model=model_retraining_all, ml_model=ml_model, name="retraining_all")
    #lgbm_retrain_last_k1 = DensityDriftModel(weight_model=model_retraining_last_k1, ml_model=ml_model, name="retraining_last_k="+str(k1))

    #lgbm_rw_sampling, lgbm_multi_rw_sampling
    #lgbm_retrain_all, lgbm_retrain_last_k1,
    models_list = [lgbm_rw,  #lgbm_multi_rw,
                  lgbm_rw_1,  #lgbm_multi_rw_1,lgbm_rw_3_1,
                  lgbm_all, lgbm_last_1,
                   lgbm_adversarial, lgbm_rw_3_bis, lgbm_rw_5]#, lgbm_rw_5_bis],lgbm_rw_5,, lgbm_rw_2, lgbm_rw_2_1, lgbm_rw_3,

    #models_list = [lgbm_rw_2, lgbm_rw_5_1]

    models = dict((k.name, k) for k in models_list)

    #env.run_experiment(models)

    # [400, 900, 1400, 1900, 2300, 2700, 3100, 3500]

    env.run_experiment(models)

    print("Framework runs done")

    def invLabel(scaler, data, idx=-1):
        dummy = pd.DataFrame(np.zeros((len(data), scaler.n_features_in_)))
        dummy.iloc[:, idx] = data
        dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=dummy.columns)
        return dummy.iloc[:, idx].values

    rmse = [pd.DataFrame(env.rmse[i], columns=[i]) for i in env.rmse.keys()]
    rmse_final = pd.concat(rmse, axis=1)
    rmse_cum = rmse_final.cumsum().div(np.cumsum(np.ones(df.shape[0]-train_size)),axis=0)
    rmse_nonnorm = dict((k, 0) for k in env.rmse.keys())
    for col in rmse_cum:
        for i in tqdm(range(len(env.predictions[col]))):
            rmse_nonnorm[col] = np.append(rmse_nonnorm[col], mean_squared_error(df_in[train_size+i:train_size+i+1]["release"], invLabel(scaler,np.array(env.predictions[col][i]),-1), squared=False))

    models_used = []
    rmse_ewmamm = []
    preq_rmse = 0
    running_rmse = dict((k, 0) for k in models_meta.keys())
    alpha = 0.9
    for i in tqdm(range(1,len(env.predictions["rw_oas"]))):
        for model_name, model in models_meta.items():
            #preds = env.predictions[model_name][i-1]
            #rmse_local = mean_squared_error(df.iloc[train_size+i-1:train_size+i].iloc[:,-1:], preds, squared=False)
            rmse_local = rmse_nonnorm[model_name][i-1]
            running_rmse[model_name] = running_rmse[model_name]*(1-alpha) + rmse_local*alpha

    model_name_ = min(running_rmse, key=running_rmse.get)
    models_used.append(model_name_)
    rmse_local = rmse_nonnorm[model_name_][i-1]
    #prediction = env.predictions[model_name_][i]
    #rmse_local = mean_squared_error(df.iloc[train_size+i:train_size+i+1].iloc[:,-1:], prediction, squared=False)
    rmse_ewmamm.append(rmse_local)

    ewmamm_p = pd.DataFrame(rmse_ewmamm)
    ewmamm = pd.DataFrame(rmse_ewmamm).cumsum().div(np.cumsum(np.ones(df.shape[0]-train_size-1)),axis=0)

    initial = time.time()
    rmse_arf = []
    rmse_tree = []
    y_true = []
    y_pred = []
    y_pred_tree = []
    arf_reg = AdaptiveRandomForestRegressor(random_state=123456)
    tree = HoeffdingAdaptiveTreeRegressor(random_state=123456)

    stream = DataStream(data= df.iloc[:,:-2], y = pd.DataFrame(df.iloc[:,-2]))

    count = 0
    while stream.has_more_samples():
        X, y = stream.next_sample()
        y_true.append(invLabel(scaler,np.array([float(y[0])]),-2))
        #y_true.append(y[0])
        prediction = invLabel(scaler,np.array([arf_reg.predict(X)[0]]),-2)
        y_pred.append(prediction)
        prediction_tree = invLabel(scaler,np.array([tree.predict(X)[0]]),-2)
        y_pred_tree.append(prediction_tree)
        #y_pred.append(arf_reg.predict(X)[0])
        #y_pred_tree.append(tree.predict(X)[0])

        if count>=2000:
            rmse_arf.append(mean_squared_error([y_true[-1]],[y_pred[-1]], squared=False))
            rmse_tree.append(mean_squared_error([y_true[-1]],[y_pred_tree[-1]], squared=False))
        if count>2000 and count%500==0:
            print("ARF: ", np.mean(rmse_arf))
            print("Tree: ", np.mean(rmse_tree))
        arf_reg.partial_fit(X, y)
        tree.partial_fit(X, y)
        count+=1

    y_pred_srp = []
    #final = 0
    model = ensemble.SRPRegressor(model=linear_model.LinearRegression(l2 = 1.0), n_models=10, seed=42,
                                drift_detector=drift_detectors_error[4], warning_detector=None)
    metric = metrics.RMSE()
    dataset = stream.iter_pandas(X = df.iloc[:,:-2], y = df.iloc[:,-2] )
    for x, y in itertools.islice(dataset, train_size):
        model.predict_one(x)
        prediction_srp = invLabel(scaler,np.array([model.predict_one(x)]),-2)
        y_pred_srp.append(prediction_srp)
        model.learn_one(x, y)

    for x, y in itertools.islice(dataset, 30000):
        model.predict_one(x)
        prediction_srp = invLabel(scaler,np.array([model.predict_one(x)]),-2)
        y_pred_srp.append(prediction_srp)
        model.learn_one(x, y)

    col = 'ARF'
    rmse_nonnorm['ARF'] = np.array([0])
    for i in tqdm(range(len(env.predictions['rw_oas']))):
        rmse_nonnorm[col] = np.append(rmse_nonnorm[col], mean_squared_error(df_in[train_size+i:train_size+i+1]["release"], [y_pred[i]], squared=False))

    col = 'SRP'
    rmse_nonnorm['SRP'] = np.array([0])
    for i in tqdm(range(len(env.predictions['rw_oas']))):
        rmse_nonnorm[col] = np.append(rmse_nonnorm[col], mean_squared_error(df_in[train_size+i:train_size+i+1]["release"], [y_pred_srp[i]], squared=False))

    col = 'HRT'
    rmse_nonnorm['HRT'] = np.array([0])
    for i in tqdm(range(len(env.predictions['rw_oas']))):
        rmse_nonnorm[col] = np.append(rmse_nonnorm[col], mean_squared_error(df_in[train_size+i:train_size+i+1]["release"], [y_pred_tree[i]], squared=False))

    rmse_nn = [pd.DataFrame(rmse_nonnorm[i], columns=[i]) for i in list(env.rmse.keys())+['HRT','ARF','SRP']]
    rmse_nn_pd = pd.concat(rmse_nn, axis=1)
    rmse_cum = rmse_nn_pd.cumsum().div(np.cumsum(np.ones(df.shape[0]-train_size+1)),axis=0)
    rmse_cum["EWMAMM"] = ewmamm

    plt.rcParams["figure.figsize"] = (7,6)
    plt.rcParams["savefig.facecolor"] = 'white'

    #exp = ["rw_oas", "multi_rw_oas", "rw_flow", "rw_kde_bis", "rw_gmm", "rw_adversarial"]
    exp = ["retraining_all", "retraining_last_1", "HRT", "SRP" , "rw_flow",  "EWMAMM"] #,"retraining_last_1", "ARF"

    legenda = {"retraining_last_1": "LAST", "retraining_all":"ALL","multi_rw_oas":"IWDA(Multi,Norm)", "rw_oas":"IW-Norm-Dens",
              "EWMAMM":"Expert", "rw_kde_bis":"IWDA(PL, KDE)","rw_flow":"IWDA(PL, MAF)", "ARF":"ARF", "HRT":"HRT", "SRP":"SRP"}

    colors = {"retraining_last_1": "goldenrod", "retraining_all":"purple","multi_rw_oas":"red", "rw_oas":"red",
              "EWMAMM":"slateblue", "rw_kde_bis":"red", "rw_flow":"red", "ARF":"darkolivegreen", "HRT":"yellowgreen", "SRP":"darkolivegreen" }

    linestyles = {"retraining_last_1": (0, (2, 2, 2, 2)), "retraining_all":"dotted", "multi_rw_oas":(0, (2,1,1,1)), "rw_oas":(0, (2,1,1,1)),
              "EWMAMM":"solid", "rw_kde_bis":(0, (2,1,1,1)), "rw_flow":(0, (2,1,1,1)), "ARF":(0, (2,1,1,2)), "HRT":(0, (1,1,1,2)), "SRP":(0, (2,1,1,2))}

    fig, ax = plt.subplots()
    for col in exp:
            ax.plot(rmse_cum[col][10:], label=legenda[col], color=colors[col],  linestyle=linestyles[col], linewidth=2.5)
    #ax.plot(ewmamm[500:], label="Expert", color=colors["EWMAMM"],  linestyle=linestyles["EWMAMM"])

    plt.rc('font', family='Times New Roman')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    plt.legend(loc = "upper left", prop={'size': 12})
    plt.xlabel("t", size=16)
    plt.ylabel("ARMSE", size=14)
    plt.margins(0.001)
    plt.savefig("Plots/como_arf.pdf",  bbox_inches = 'tight')
