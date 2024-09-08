import unittest
from ML_FLOW_system import Entry_ML_Flow
import numpy as np


class PruebaMLFlow(unittest.TestCase):
    def test_model(self):
        model = Entry_ML_Flow()
        Predicciones, modelo, mensaje, error, tuned_modelo, exp_clf101 = model.ML_FLOW(ingenieria=True, n_modelo = 3)
        u1,u2 = model.precision(tuned_modelo, exp_clf101)

        if mensaje == "Proceso Exitoso":
            self.assertLessEqual(np.abs(u1-u2),10,print("No presenta Underfitting ni Overfitting"))
            a = "Modelo entrenado correctamente ..."
            return {'Procces':a}
        else:
            a = "No se logro entrenar el modelo de forma satisfactoria ..."
            return {'Procces':a}

pb = PruebaMLFlow()
print(pb.test_model())
if __name__ == "__main__":
    unittest.main()
