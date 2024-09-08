import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
from itertools import combinations
from xgboost import XGBClassifier
import re
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from pycaret.classification import *
from sklearn.metrics import accuracy_score
from pycaret.classification import setup
import warnings
warnings.filterwarnings("ignore")


class Entry_ML_Flow:
    def __init__(self):
        pass
    def carga_dt(self):
        #path = "C:/Users/MCS/OneDrive - Universidad Santo Tomás/Inteligencia Artificial/Codigos Propios/PARCIAL 1 CORTE/"
        path = "D:/Downloads/PARCIAL IA/"
        train = pd.read_csv(path + "train.csv")
        test = pd.read_csv(path + "test.csv")
        train = train.drop(columns = ["id"])
        test = test.drop(columns = ["id"])
        train.rename(columns = {"Mother's occupation": "Mother occupation",
                            "Father's occupation": "Father occupation",
                            "Mother's qualification": "Mother qualification",
                            "Father's qualification": "Father qualification"}, inplace = True)
        test.rename(columns = {"Mother's occupation": "Mother occupation",
                           "Father's occupation": "Father occupation",
                           "Mother's qualification": "Mother qualification",
                           "Father's qualification": "Father qualification"}, inplace = True)
        categ = ['Course','Marital status','Application mode', 'Application order', 'Daytime/evening attendance', 
             'Nacionality', 'Previous qualification',"Mother qualification","Father qualification", 
             "Mother occupation", "Father occupation", 'Displaced', 'Educational special needs', 
             'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International']
        for k in categ:
            train[k] = train[k].astype("O")
            test[k] = test[k].astype("O")
        return train, test
        
    def procesamiento(self,datos):
        formato = pd.DataFrame({'Variable': list(datos.columns), 'Formato': datos.dtypes })
        categoricas = list(formato.loc[formato["Formato"]=="O","Variable"])
        categoricas = [x for x in categoricas if "Target" not in x]
        numericas = list(formato.loc[formato["Formato"]!="O","Variable"])
        numericas = [x for x in numericas if "Target" not in x]
        datos_num = datos.get(numericas)
        datos_cat = datos.get(categoricas)
        return datos_num, datos_cat
    
    def nombre_(self,x):
        return "C"+str(x)
    
    def indicadora1(self,x):
        if x=="Graduate":
            return 0
        elif x=="Enrolled":
            return 1
        else:
            return 2
    
    def prueba_kr(self,x):
        if x<=0.10:
            return 1
        else:
            return 0
        
    def criterion_(self,datos):
        for k in datos.columns:
            datos[k] = datos[k].map(self.prueba_kr)
            datos["criterio"] = np.sum(datos.get(datos.columns),axis=1)
            datos["criterio"] = datos.apply(lambda row: 1 if row["criterio"]==3 else 0,axis = 1)
            return datos
        
    def indicadora(self,x):
        if x==True:
            return 1
        else:
            return 0
        
    def var_cuadrado(self,datos, cuantitativas):
        base_cuadrado = datos.get(cuantitativas).copy()
        base_cuadrado["Target"] = datos["Target"].copy()
        var_names2, pvalue1 = [], []
        for k in cuantitativas:
            base_cuadrado[k+"_2"] = base_cuadrado[k] ** 2
            mue1 = base_cuadrado.loc[base_cuadrado["Target"]=="Graduate",k+"_2"].to_numpy()
            mue2 = base_cuadrado.loc[base_cuadrado["Target"]=="Dropout",k+"_2"].to_numpy()
            mue3 = base_cuadrado.loc[base_cuadrado["Target"]=="Enrolled",k+"_2"].to_numpy()
            p1 = stats.kruskal(mue1,mue2,mue3)[1]
            var_names2.append(k+"_2")
            pvalue1.append(np.round(p1,2))
        
        pcuadrado1 = pd.DataFrame({'Variable2':var_names2,'p value':pvalue1})
        pcuadrado1["criterio"] = pcuadrado1.apply(lambda row: 1 if row["p value"]<=0.10 else 0,axis = 1)
        
        var_cuad = list(pcuadrado1["Variable2"])
        base_modelo1 = base_cuadrado.get(var_cuad+["Target"])
        base_modelo1["Target"] = base_modelo1["Target"].map(self.indicadora1)
        return base_modelo1
    
    def interacciones(self,datos, cuantitativas):
        lista_inter = list(combinations(cuantitativas,2))
        base_interacciones = datos.get(cuantitativas).copy()
        var_interaccion, pv1 = [], []
        base_interacciones["Target"] = datos["Target"].copy()
        for k in lista_inter:
            base_interacciones[k[0]+"__"+k[1]] = base_interacciones[k[0]] * base_interacciones[k[1]]
            mue1 = base_interacciones.loc[base_interacciones["Target"]=="Graduate",k[0]+"__"+k[1]].to_numpy()
            mue2 = base_interacciones.loc[base_interacciones["Target"]=="Dropout",k[0]+"__"+k[1]].to_numpy()
            mue3 = base_interacciones.loc[base_interacciones["Target"]=="Enrolled",k[0]+"__"+k[1]].to_numpy()
            p1 = stats.kruskal(mue1,mue2,mue3)[1]
            var_interaccion.append(k[0]+"__"+k[1])
            pv1.append(np.round(p1,2))
            
        pxy = pd.DataFrame({'Variable':var_interaccion,'p value':pv1})
        pxy["criterio"] = pxy.apply(lambda row: 1 if row["p value"]<=0.10 else 0, axis = 1)
        
        var_int = list(pxy["Variable"])
        base_modelo2 = base_interacciones.get(var_int+["Target"])
        base_modelo2["Target"] = base_modelo2["Target"].map(self.indicadora1)
        return base_modelo2
    
    def razones(self,datos, cuantitativas):
        raz1 = [(x,y) for x in cuantitativas for y in cuantitativas]
        base_razones1 = datos.get(cuantitativas).copy()
        base_razones1["Target"] = datos["Target"].copy()
        var_nm, pval = [], []
        for j in raz1:
            if j[0]!=j[1]:
                base_razones1[j[0]+"__coc__"+j[1]] = base_razones1[j[0]] / (base_razones1[j[1]]+0.01)
                mue1 = base_razones1.loc[base_razones1["Target"]=="Graduate",j[0]+"__coc__"+j[1]].to_numpy()
                mue2 = base_razones1.loc[base_razones1["Target"]=="Dropout",j[0]+"__coc__"+j[1]].to_numpy()
                mue3 = base_razones1.loc[base_razones1["Target"]=="Enrolled",j[0]+"__coc__"+j[1]].to_numpy()
                p1 = stats.kruskal(mue1,mue2,mue3)[1]
                var_nm.append(j[0]+"__coc__"+j[1])
                pval.append(np.round(p1,2))
        
        prazones = pd.DataFrame({'Variable':var_nm,'p value':pval})
        prazones["criterio"] = prazones.apply(lambda row: 1 if row["p value"]<=0.10 else 0, axis = 1)
        var_raz = list(prazones["Variable"])
        base_modelo3 = base_razones1.get(var_raz+["Target"])
        base_modelo3["Target"] = base_modelo3["Target"].map(self.indicadora1)
        return base_modelo3
    
    def inter_cat(self,datos, categoricas):
        cb = list(combinations(categoricas,2))
        p_value, modalidades, nombre_var = [], [], []
        base2 = datos.get(categoricas).copy()
        for k in base2.columns:
            base2[k] = base2[k].map(self.nombre_)
        base2["Target"] = datos["Target"].copy()
        for k in range(len(cb)):
            base2[cb[k][0]] = base2[cb[k][0]]
            base2[cb[k][1]] = base2[cb[k][1]]
            base2[cb[k][0]+"__"+cb[k][1]] = base2[cb[k][0]] + "__" + base2[cb[k][1]]
            c1 = pd.DataFrame(pd.crosstab(base2["Target"],base2[cb[k][0]+"__"+cb[k][1]]))
            pv = stats.chi2_contingency(c1)[1]
            mod_ = len(base2[cb[k][0]+"__"+cb[k][1]].unique())
            nombre_var.append(cb[k][0]+"__"+cb[k][1])
            modalidades.append(mod_)
            p_value.append(pv)
        
        pc = pd.DataFrame({'Variable':nombre_var,'Num Modalidades':modalidades,'p value':p_value})
        seleccion1 = list(pc.loc[(pc["p value"]<=0.10) & (pc["Num Modalidades"]<=8),"Variable"])
        sel1 = base2.get(seleccion1)
        contador = 0
        for k in sel1:
            if contador==0:
                lb1 = pd.get_dummies(sel1[k],drop_first=True)
                lb1.columns = [k + "_" + x for x in lb1.columns]
            else:
                lb2 = pd.get_dummies(sel1[k],drop_first=True)
                lb2.columns = [k + "_" + x for x in lb2.columns]
                lb1 = pd.concat([lb1,lb2],axis=1)
            contador = contador + 1
        for k in lb1.columns:
            lb1[k] = lb1[k].map(self.indicadora)
        
        lb1["Target"] = datos["Target"].copy()
        lb1["Target"] = lb1["Target"].map(self.indicadora1)
        return lb1
    
    def inte_cat_cuan(self,datos, cuantitativas, categoricas):
        cat_cuanti = [(x,y) for x in cuantitativas for y in categoricas]
        v1, v2, pvalores_min, pvalores_max  = [], [], [], []
        for j in cat_cuanti:
            k1 = j[0]
            k2 = j[1]
            g1 = pd.get_dummies(datos[k2])
            lt1 = list(g1.columns)
            for k in lt1:
                g1[k] = g1[k] * datos[k1]
            g1["Target"] = datos["Target"].copy()
            pvalues_c = []
            for y in lt1:
                mue1 = g1.loc[g1["Target"]=="Graduate",y].to_numpy()
                mue2 = g1.loc[g1["Target"]=="Dropout",y].to_numpy()
                mue3 = g1.loc[g1["Target"]=="Enrolled",y].to_numpy()
                try:
                    pval = (stats.kruskal(mue1,mue2,mue3)[1]<=0.10)
                    if pval==True:
                        pval = 1
                    else:
                        pval = 0
                except ValueError:
                    pval = 0
                pvalues_c.append(pval)
            min_ = np.min(pvalues_c) 
            max_ = np.max(pvalues_c) 
            v1.append(k1) 
            v2.append(k2) 
            pvalores_min.append(np.round(min_,2))
            pvalores_max.append(np.round(max_,2))
        pc2 = pd.DataFrame({'Cuantitativa':v1,'Categórica':v2,'p value':pvalores_min, 'p value max':pvalores_max})
        v1 = list(pc2.loc[(pc2["p value"]==1) & (pc2["p value max"]==1),"Cuantitativa"])
        v2 = list(pc2.loc[(pc2["p value"]==1) & (pc2["p value max"]==1),"Categórica"])
        for j in range(len(v1)):
            if j==0:
                g1 = pd.get_dummies(datos[v2[j]],drop_first=True)
                lt1 = list(g1.columns)
                for k in lt1:
                    g1[k] = g1[k] * datos[v1[j]]
                g1.columns = [v1[j] + "_" + v2[j] + "_" + str(x) for x in lt1]
            else:
                g2 = pd.get_dummies(datos[v2[j]],drop_first=True)
                lt1 = list(g2.columns)
                for k in lt1:
                    g2[k] = g2[k] * datos[v1[j]]
                g2.columns = [v1[j] + "_" + v2[j] + "_" + str(x) for x in lt1]
                g1 = pd.concat([g1,g2],axis=1)
        g1["Target"] = datos["Target"].copy()
        g1["Target"] = g1["Target"].map(self.indicadora1)
        return g1
    
    def xgboost(self,base_modelo):
        cov = list(base_modelo.columns)
        cov = [x for x in cov if x not in ["Target"]]
        X = base_modelo.get(cov)
        y = base_modelo.get(["Target"])
        modelo = XGBClassifier()
        modelo = modelo.fit(X,y)
        importancias = modelo.feature_importances_
        imp = pd.DataFrame({'Variable':X.columns,'Importancia':importancias})
        imp["Importancia"] = imp["Importancia"] * 100 / np.sum(imp["Importancia"])
        imp = imp.sort_values(["Importancia"],ascending=False)
        imp.index = range(imp.shape[0])
        return imp
    
    def preparacion(self, datos, cuantitativas, categoricas, c2, cxy, razxy, catxy, cuactxy):
        D1 = datos.get(cuantitativas).copy()
        D2 = datos.get(categoricas).copy()
        for k in categoricas:
            D2[k] = D2[k].map(self.nombre_)
        D4 = D2.copy()
        cuadrado = [re.findall(r'(.+)_\d+', item) for item in c2]
        cuadrado = [x[0] for x in cuadrado]
        
        for k in cuadrado:
            D1[k+"_2"] = D1[k] ** 2
        
        result = [re.findall(r'([A-Za-z\s\(\)0-9]+)', item) for item in cxy]
        for k in result:
            D1[k[0]+"__"+k[1]] = D1[k[0]] * D1[k[1]]
        
        result2 = [re.findall(r'(.+)__coc__(.+)', item) for item in razxy]
        for k in result2:
            k2 = k[0]
            D1[k2[0]+"__coc__"+k2[1]] = D1[k2[0]] / (D1[k2[1]]+0.01)
        
        result3 = [re.search(r'([^_]+__[^_]+)', item).group(1).split('__') for item in catxy]
        
        for k in result3:
            D4[k[0]+"__"+k[1]] = D4[k[0]] + "_" + D4[k[1]]
        
        D5 = datos.copy()
        result4 = [re.search(r'(.+?)_(.+?)_1', item).groups() for item in cuactxy]
        contador = 0
        for k in result4:
            col1, col2 = k[1], k[0] 
            
            if contador == 0:
                D51 = pd.get_dummies(D5[col1],drop_first=True)
                
                for j in D51.columns:
                    D51[j] = D51[j] * D5[col2]
                    
                D51.columns = [col2+"_"+col1+"_"+ str(x) for x in D51.columns]
            
            else:
                D52 = pd.get_dummies(D5[col1],drop_first=True)
                for j in D52.columns:
                    D52[j] = D52[j] * D5[col2]
                D52.columns = [col2+"_"+col1+"_"+ str(x) for x in D52.columns]
                D51 = pd.concat([D51,D52],axis=1)
            
            contador = contador + 1
        B1 = pd.concat([D1,D4],axis=1)
        base_modelo = pd.concat([B1,D51],axis=1)
        base_modelo["Target"] = datos["Target"].copy()
        base_modelo["Target"] = base_modelo["Target"].map(self.indicadora1)
        return base_modelo, cuadrado, result, result2, result3, result4
    
    def proceso_test(self,datos,cuadrado, result, result2, result3, result4):
        cuantitativas, categoricas = self.procesamiento(datos)

        D1 = datos.get(list(cuantitativas)).copy()
        D2 = datos.get(list(categoricas)).copy()
        for k in categoricas:
            D2[k] = D2[k].map(self.nombre_)
        D4 = D2.copy()
        for k in cuadrado:
            D1[k+"_2"] = D1[k] ** 2
        for k in result:
            D1[k[0]+"__"+k[1]] = D1[k[0]] * D1[k[1]]
        for k in result2:
            k2 = k[0]
            D1[k2[0]+"__coc__"+k2[1]] = D1[k2[0]] / (D1[k2[1]]+0.01)
        for k in result3:
            D4[k[0]+"__"+k[1]] = D4[k[0]] + "_" + D4[k[1]]
        D5 = datos.copy()
        contador = 0
        for k in result4:
            col1, col2 = k[1], k[0] # categórica, cuantitativa
            
            if contador == 0:
                D51 = pd.get_dummies(D5[col1],drop_first=True)
                for j in D51.columns:
                    D51[j] = D51[j] * D5[col2]
                D51.columns = [col2+"_"+col1+"_"+ str(x) for x in D51.columns]
            else:
                D52 = pd.get_dummies(D5[col1],drop_first=True)
                for j in D52.columns:
                    D52[j] = D52[j] * D5[col2]
                D52.columns = [col2+"_"+col1+"_"+ str(x) for x in D52.columns]
                D51 = pd.concat([D51,D52],axis=1)
            contador = contador + 1
        B1 = pd.concat([D1,D4],axis=1)
        base_modelo2 = pd.concat([B1,D51],axis=1)
        return base_modelo2
    ##FUNCION FINAL
    def modelos(self,datos, numericos, categoricos, n_modelo):
        from pycaret.classification import setup, compare_models, create_model
        exp_clf101 = setup(data = datos, target='Target', session_id = 123, train_size = 0.7,
                           numeric_features = list(numericos.columns), 
                           categorical_features = list(categoricos.columns), fix_imbalance=False,
                           verbose = False)
        modelo = create_model(n_modelo, verbose = False)
        return exp_clf101, modelo
        
    def modelo_final(self,modelo, parametros):
        tuned_modelo = tune_model(modelo, custom_grid = parametros,
                                  search_library = 'scikit-optimize', search_algorithm = 'bayesian', 
                                  fold = 5, verbose = False)
        modelo_final1 = finalize_model(tuned_modelo)
        return tuned_modelo, modelo_final1
    
    
    def indicadora2(self,x):
        if x==0:
            return "Graduate"
        elif x== 1:
            return "Enrolled"
        else:
            return "Dropout"
        
    def ML_FLOW(self,ingenieria, n_modelo):
        try:
            train, test = self.carga_dt()
            numericas, categoricas = self.procesamiento(train)
            if ingenieria == False:
                for p in categoricas.columns:
                    train[p] = "C" + train[p].astype("str")
                    test[p] = "C" + test[p].astype("str")
                base = train
                base_test = test
            elif ingenieria ==True:
                cuantitativas = list(numericas.columns)
                categoricas = list(categoricas.columns)
                
                base_modelo1 = self.var_cuadrado(train, cuantitativas)
                imp1 = self.xgboost(base_modelo1)
                c2 = list(imp1.iloc[0:3,0])
            
                base_modelo2 = self.interacciones(train, cuantitativas)
                imp2 = self.xgboost(base_modelo2)
                cxy = list(imp2.iloc[0:10,0])
                
                base_modelo3 = self.razones(train, cuantitativas)
                imp3 = self.xgboost(base_modelo3)
                razxy = list(imp3.iloc[0:10,0])
                
                lb1 = self.inter_cat(train, categoricas)
                imp4 = self.xgboost(lb1)
                catxy = list(imp4.iloc[0:3,0])
                
                g1 = self.inte_cat_cuan(train, cuantitativas, categoricas)
                imp5 = self.xgboost(g1)
                cuactxy = list(imp5.iloc[0:3,0]) 
                base, cuadrado, result, result2, result3, result4 = self.preparacion(train, cuantitativas, categoricas, c2, cxy, razxy, catxy, cuactxy)
                base_test = self.proceso_test(test, cuadrado, result, result2, result3, result4)
                numericas, categoricas = self.procesamiento(base)
            
            if n_modelo == 1:
                exp_clf101, modelo = self.modelos(base, numericas, categoricas, n_modelo= "lightgbm")
                parametros = {'n_estimators': [50,100,200], 'max_depth': [3,5,7], 'min_child_samples': [50,150,200]}
                tuned_modelo, modelo_final1 = self.modelo_final(modelo, parametros)
            
            elif n_modelo ==2:
                exp_clf101, modelo = self.modelos(base, numericas, categoricas, n_modelo = "xgboost")
                parametros = {'n_estimators': [50,100,200], 'max_depth': [3,5,7], 'min_child_samples': [50,150,200]}
                tuned_modelo, modelo_final1 = self.modelo_final(modelo, parametros)
            
            elif n_modelo ==3:
                exp_clf101, modelo = self.modelos(base, numericas, categoricas, n_modelo = "gbc")
                parametros = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 
                                                    'min_samples_split': [2, 10, 20], 'min_samples_leaf': [1, 5, 10], 
                                                    'learning_rate': [0.01, 0.1, 0.2]}
                tuned_modelo, modelo_final1 = self.modelo_final(modelo, parametros)
            
            Predicciones = predict_model(modelo_final1, data = base_test)
            if ingenieria== True:
                Predicciones = Predicciones["prediction_label"].map(self.indicadora2)
            elif ingenieria== False:
                Predicciones = Predicciones["prediction_label"]
            mensaje = "Proceso Exitoso"
            error = "No hubo errores"
            return  Predicciones, modelo_final1, mensaje, error, tuned_modelo, exp_clf101
        except Exception as e:
            Predicciones = 0
            modelo_final1 = 0
            mensaje = "Hubo un error"
            error = str(e)
            tuned_modelo = "No se encontro"
            exp_clf101 ="No se encontro"
            return  Predicciones, modelo_final1, mensaje, error, tuned_modelo, exp_clf101
        
    def precision(self,tuned_modelo, exp_clf101):
        predicciones_train = predict_model(tuned_modelo)
        predicciones_test = predict_model(tuned_modelo, data = exp_clf101.get_config('X_train'))
        
        y_train = exp_clf101.get_config('y_train')
        y_test = exp_clf101.get_config('y_test')
        
        u1 = accuracy_score(y_test,predicciones_train["prediction_label"])
        u2 = accuracy_score(y_train,predicciones_test["prediction_label"])
        return u1, u2
    def resultados(Predicciones):
        path= "D:/Downloads/PARCIAL IA/"
        #path = "C:/Users/MCS/OneDrive - Universidad Santo Tomás/Inteligencia Artificial/Codigos Propios/PARCIAL 1 CORTE/"
        test12 = pd.read_csv(path + "test.csv")
        result = pd.DataFrame({
            'id': test12["id"],
            'Target': Predicciones})
        return result