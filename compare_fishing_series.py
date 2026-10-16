"""Run with results = evaluate_fishing_series(df).
Requires pandas, numpy, sklearn, statsmodels, tensorflow (not xgboost).
"""
import warnings
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf

def evaluate_fishing_series(df, output_dir='fishing_comparison', seeds=(42,123,2026)):
    data=df.copy()
    data.columns=data.columns.str.strip().str.lower()
    for c in ['year','wave','st','sub_reg','mode_fx','area_x','estrips']:
        if c not in data: raise ValueError(f'Missing column {c}')
        data[c]=pd.to_numeric(data[c].astype('string').str.replace(',','',regex=False).str.strip(),errors='coerce').astype('float64')
    dates=pd.date_range('1981-01-01','2025-11-01',freq='2MS')
    lookback,val_n,test_n=12,18,30
    # 2006-2010, 2011-2015, 2016-2020. Final 2021-2025 excluded.
    first_test=len(dates)-4*test_n
    scores,predictions,tuning,coverage,failures=[],[],[],[],[]
    definitions=[('East shore',6,3),('East private boat',6,7),
                 ('West shore',7,3),('West private boat',7,7)]

    def score(actual,pred):
        return mean_absolute_error(actual,pred),np.sqrt(mean_squared_error(actual,pred))

    def fit_sarima(y,order,seasonal):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r=SARIMAX(y,order=order,seasonal_order=seasonal,trend='n',
                enforce_stationarity=True,enforce_invertibility=True).fit(disp=False,maxiter=300)
        if not r.mle_retvals.get('converged',False): raise RuntimeError('Nonconverged SARIMA')
        return r

    def rolling(r,obs):
        out=[]
        for i in range(len(obs)):
            p=float(np.asarray(r.get_forecast(1).predicted_mean)[0])
            out.append(max(0.,p))
            r=r.append(obs.iloc[i:i+1],refit=False)
        return np.asarray(out)

    for name,coast,mode in definitions:
        sub=data.loc[data.st.eq(12)&data.sub_reg.eq(coast)&data.mode_fx.eq(mode)
                     &data.area_x.eq(5)&data.year.between(1981,2025)&data.wave.isin(range(1,7))].copy()
        if sub.empty: raise ValueError(f'No records for {name}')
        sub['date']=pd.to_datetime(dict(year=sub.year.astype(int),month=sub.wave.astype(int)*2-1,day=1))
        if sub.date.duplicated().any(): raise ValueError(f'Duplicate dates in {name}: inspect source aggregation')
        y=sub.set_index('date').estrips.sort_index().reindex(dates)
        if y.dropna().lt(0).any(): raise ValueError(f'Negative effort in {name}')
        coverage.append(dict(series=name,observed=int(y.notna().sum()),missing=int(y.isna().sum()),area_x=5))
        X=pd.DataFrame({f'lag_{k}':y.shift(k) for k in range(1,lookback+1)})
        complete=X.notna().all(axis=1)
        for fold in range(1,4):
            start=first_test+(fold-1)*test_n
            end=start+test_n
            split=start-val_n
            fitting=y.iloc[:split]; validation=y.iloc[split:start]; test=y.iloc[start:end]
            fit_mask=X.index.isin(fitting.index)&complete&y.notna()
            val_mask=X.index.isin(validation.index)&complete&y.notna()
            test_mask=X.index.isin(test.index)&complete&y.notna()&y.shift(6).notna()
            idx=X.index[test_mask]
            if fit_mask.sum()<24 or val_mask.sum()<6 or len(idx)<6:
                raise ValueError(f'{name}, fold {fold}: too few complete windows')
            out=pd.DataFrame({'actual':y.reindex(idx),'seasonal_naive':y.shift(6).reindex(idx)},index=idx)
            # Statistical candidate selection by rolling validation MAE.
            best=np.inf; selected=None
            for p,q,P,Q in product([0,1],repeat=4):
                order,seasonal=(p,1,q),(P,1,Q,6)
                try:
                    r=fit_sarima(fitting,order,seasonal)
                    v=rolling(r,validation)
                    valid=validation.notna().to_numpy(dtype=float)
                    if not np.isfinite(v[valid]).all(): continue
                    mae=mean_absolute_error(validation.to_numpy(dtype=float)[valid],v[valid])
                    tuning.append(dict(series=name,fold=fold,model='sarima',settings=str((order,seasonal)),seed=None,validation_MAE=mae))
                    if mae<best: best=mae;selected=(r,order,seasonal)
                except (RuntimeError,ValueError,np.linalg.LinAlgError) as e:
                    failures.append(dict(series=name,fold=fold,model='sarima',settings=str((order,seasonal)),error=str(e)))
            if selected is None: raise RuntimeError(f'No valid SARIMA for {name}, fold {fold}')
            r,order,seasonal=selected
            r=r.append(validation,refit=False)
            out['sarima']=pd.Series(rolling(r,test),index=test.index).reindex(idx)
            # Tree model, same historical window and log target as BiLSTM.
            best=np.inf; gb=None
            for depth,lr,trees in product([2,3],[.03,.1],[100,300]):
                m=GradientBoostingRegressor(max_depth=depth,learning_rate=lr,n_estimators=trees,
                     min_samples_leaf=3,random_state=42)
                m.fit(X.loc[fit_mask],np.log1p(y.loc[fit_mask]))
                v=np.maximum(0,np.expm1(m.predict(X.loc[val_mask])))
                mae=mean_absolute_error(y.loc[val_mask],v)
                tuning.append(dict(series=name,fold=fold,model='gradient_boosting',settings=str((depth,lr,trees)),seed=42,validation_MAE=mae))
                if mae<best: best=mae;gb=m
            out['gradient_boosting']=np.maximum(0,np.expm1(gb.predict(X.reindex(idx))))
            # Neural tuning: average validation MAE across predefined seeds.
            # All seeds are retained; never pick a seed using test performance.
            scaler=StandardScaler().fit(np.log1p(fitting.dropna().to_numpy(dtype=float)).reshape(-1,1))
            def scaled_features(mask):
                a=X.loc[mask].to_numpy(dtype=float)
                a=scaler.transform(np.log1p(a).reshape(-1,1)).reshape(-1,lookback)
                return a[:,::-1].copy().reshape(-1,lookback,1).astype('float32')
            xf,xv,xt=scaled_features(fit_mask),scaled_features(val_mask),scaled_features(test_mask)
            yf=scaler.transform(np.log1p(y.loc[fit_mask].to_numpy(dtype=float)).reshape(-1,1)).ravel()
            yv=scaler.transform(np.log1p(y.loc[val_mask].to_numpy(dtype=float)).reshape(-1,1)).ravel()
            best=np.inf; best_models=None
            for batch,lr in product([16,35],[.001,.01,.1]):
                models=[]; seed_losses=[]
                for seed in seeds:
                    tf.keras.backend.clear_session()
                    tf.keras.utils.set_random_seed(seed)
                    m=tf.keras.Sequential([tf.keras.Input(shape=(lookback,1)),
                        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8)),tf.keras.layers.Dense(1)])
                    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),loss='mse')
                    m.fit(xf,yf,validation_data=(xv,yv),epochs=200,batch_size=batch,shuffle=False,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=15,restore_best_weights=True)],verbose=0)
                    v=np.maximum(0,np.expm1(scaler.inverse_transform(m.predict(xv,verbose=0)).ravel()))
                    mae=mean_absolute_error(y.loc[val_mask],v) if np.isfinite(v).all() else np.inf
                    seed_losses.append(mae);models.append(m)
                    tuning.append(dict(series=name,fold=fold,model='bilstm',settings=str((batch,lr)),seed=seed,validation_MAE=mae))
                if np.mean(seed_losses)<best: best=np.mean(seed_losses);best_models=models
            if best_models is None: raise RuntimeError(f'No finite BiLSTM for {name}, fold {fold}')
            for seed,m in zip(seeds,best_models):
                out[f'bilstm_seed_{seed}']=np.maximum(0,np.expm1(scaler.inverse_transform(m.predict(xt,verbose=0)).ravel()))
            # Equal-weight ensemble is reported separately from individual seeds.
            out['bilstm_ensemble']=out[[f'bilstm_seed_{s}' for s in seeds]].mean(axis=1)
            if not np.isfinite(out.to_numpy(dtype=float)).all(): raise ValueError('Nonfinite scored predictions')
            scale=(y.iloc[:start]-y.iloc[:start].shift(6)).abs().mean()
            for model in out.columns.drop('actual'):
                mae,rmse=score(out.actual,out[model])
                scores.append(dict(series=name,fold=fold,model=model,MAE=mae,RMSE=rmse,
                    seasonal_MASE=mae/scale if pd.notna(scale) and scale>0 else np.nan,n_test=len(idx)))
            out['series']=name;out['fold']=fold
            predictions.append(out.rename_axis('date').reset_index())
            print(f'{name}, fold {fold}: {len(idx)}/{test_n} waves scored',flush=True)
    scores=pd.DataFrame(scores);predictions=pd.concat(predictions,ignore_index=True)
    summary=[]
    for name,g in predictions.groupby('series'):
        for model in ['seasonal_naive','sarima','gradient_boosting','bilstm_ensemble']+[f'bilstm_seed_{s}' for s in seeds]:
            mae,rmse=score(g.actual,g[model])
            baseline=mean_absolute_error(g.actual,g.seasonal_naive)
            summary.append(dict(series=name,model=model,MAE=mae,RMSE=rmse,n_test=len(g),
                improvement_vs_seasonal_naive_pct=100*(baseline-mae)/baseline if baseline>0 else np.nan))
    summary=pd.DataFrame(summary)
    outdir=Path(output_dir);outdir.mkdir(parents=True,exist_ok=True)
    for filename,table in [('fold_scores',scores),('pooled_by_series',summary),('predictions',predictions),
                           ('validation_tuning',pd.DataFrame(tuning)),('coverage',pd.DataFrame(coverage)),
                           ('failed_candidates',pd.DataFrame(failures))]:
        table.to_csv(outdir/f'{filename}.csv',index=False)
    print(summary.sort_values(['series','MAE']).round(3).to_string(index=False))
    return {'scores':scores,'summary':summary,'predictions':predictions,
            'tuning':pd.DataFrame(tuning),'coverage':pd.DataFrame(coverage)}
