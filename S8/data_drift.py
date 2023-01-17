import pandas as pd
from sklearn import datasets

reference_data = datasets.load_iris(as_frame='auto').frame
current_data = pd.read_csv('database.csv')

current_data = current_data.drop(["time"],axis=1)

current_data.columns = reference_data.columns


from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
report = Report(metrics=[DataDriftPreset(),TargetDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data)
report.save_html('report.html')

from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfColumns
data_test = TestSuite(tests=[TestNumberOfColumns()])
data_test.run(reference_data=reference_data, current_data=current_data)

data_test.save_html('data_test.html')