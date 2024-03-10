import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, recall_score, 
    plot_confusion_matrix, precision_score, plot_roc_curve
)
import matplotlib.pyplot as plt
from joblib import dump

from pipeline.training.training import x_test
import glob

class Processor():
    def __init__(self) -> None:
        self._x = None
        self._y = None
    
    def load_data(self, path) -> None:

        dataframes = []
        files = glob.glob(path + "/*.parquet")
        for file in files:
            data_partition = pd.read_parquet(file)
            dataframes.append(data_partition)
    
        data = pd.concat(dataframes, ignore_index=True)

        data.fillna(0, inplace=True)

        self._y = data['status']

        data.drop(['status', 'id'], axis=1, inplace=True)
        self._x = data
    
    def resample(self) -> None:
        # Using Synthetic Minority Over-Sampling Technique(SMOTE) to overcome sample imbalance problem.
        self._y = self._y.astype('int')
        x_balance, y_balance = SMOTE().fit_resample(self._x, self._y)

        self._x = pd.DataFrame(x_balance, columns=self._x.columns)
        self._y = y_balance
    
    def split_train_test(self, ratio) -> tuple:
        x_train, x_test, y_train, y_test = train_test_split(self._x,self._y, 
                                                    stratify=self._y, test_size=ratio,
                                                    random_state = 123)
        return x_train, x_test, y_train, y_test


class Trainer():
    def __init__(self, model) -> None:
        self._model = model
    
    def load_data(self, x_train, y_train, x_test, y_test) -> None:
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test
    
    def train(self) -> None:
        self._model.fit(self._x_train, self._y_train)

    def get_metrics(self) -> None:
        y_predict = self._model.predict(self._x_test)

        print('Accuracy Score is {:.5}'.format(accuracy_score(self._y_test, y_predict)))
        print('Precision Score is {:.5}'.format(precision_score(self._y_test, y_predict)))
        print('Recall Score is {:.5}'.format(precision_score(self._y_test, y_predict)))
        print(pd.DataFrame(confusion_matrix(self._y_test,y_predict)))

        plot_confusion_matrix(self._model, self._x_test, self._y_test)  
        plt.show()

        plot_roc_curve(self._model, self._x_test, self._y_test)
        plt.show()

    def save_model(self, path) -> None:
        dump(self._model, path)


if __name__ == '__main__':
    processor = Processor()
    
    input_data_folder = 'train_model_spark'
    processor.load_data(input_data_folder)

    processor.resample()

    data = processor.split_train_test(0.3)


    trainer = Trainer(RandomForestClassifier(n_estimators=5))

    trainer.load_data(data)
    trainer.train()
    trainer.get_metrics()

    output_model = 'model_risk.joblib'
    trainer.save_model(output_model)