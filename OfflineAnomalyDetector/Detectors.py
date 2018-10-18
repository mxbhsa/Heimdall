import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from luminol.anomaly_detector import AnomalyDetector
from luminol.modules.anomaly import Anomaly
from luminol.modules.time_series import TimeSeries

from AnomalyAlgorithms import anomaly_detect_ts
from Utils import *

from abc import abstractmethod


from keras.layers import Conv1D, GlobalMaxPool1D, Dense, Flatten, LSTM, Bidirectional, RepeatVector, MaxPooling1D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import numpy as np

class BaseDetector(object):
	"""docstring for BaseDetector"""
	def __init__(self):
		super(BaseDetector, self).__init__()

	@abstractmethod
	def fit(self):
		pass
		
	def predict(self, timeSeries):
		pass

	def run(self):
		pass

	def mergeAnomaly(self, mergeLen = 3):
		accum_num = 0
		find = -1
		ret = []
		l = len(self.anoms)
		for i, x in enumerate(anoms):
			if x and find == -1:
				find = i
			if not x or i == l-1:
				if i == l-1:
					i += 1
				if i - find >= mergeLen and find > -1:
					print i, find
					ret.append(Anomaly(find, i, 0, 0))
				find = -1
		return ret

	def alterFilter(score, threshold=0.99, two_sided=True):
	    """

	    :param score: a score, assumed normalised (between 0 and 1) representing anomalousness
	    :param threshold: a user-specified threshold above which an alert should be raised
	    :param two_sided: if True, we flag anomalies that are either smaller than 1-threshold or larger than threhsold
	    :return: a boolean flag

	    >>> decision_rule(score=0.9)
	    False
	    >>> decision_rule(score=0.95, threshold=0.9)
	    True
	    >>> decision_rule(score=0.0001, threshold=0.99)
	    True
	    >>> decision_rule(score=0.001, two_sided=False)
	    False
	    """
	    if two_sided:
	        ans = np.logical_or(score < 1-threshold, score > threshold)
	    else:
	        ans = score > threshold


	    return ans

	def genTranSet(self, npArray, time_window_size):
		ret = np.array(npArray[0,0:time_window_size])
		print npArray.shape
		print ret.shape
		for i in xrange(0,npArray.shape[0]):
			for j in xrange(0,npArray.shape[1]-time_window_size+1,time_window_size):
				#print npArray[j:j+time_window_size,i].shape
				ret = np.vstack((ret,npArray[i,j:j+time_window_size]))
		print ret.shape
		print ret
		return ret


	def printLog(self):
		print "test"

class LstmAutoEncoder(BaseDetector):
    model_name = 'lstm-auto-encoder'
    VERBOSE = 1

    def __init__(self):
        self.model = None
        self.time_window_size = None
        self.config = None
        self.metric = None
        self.threshold = None

    @staticmethod
    def create_model(time_window_size, metric):
        model = Sequential()
        model.add(LSTM(units=128, input_shape=(time_window_size, 1), return_sequences=False))

        model.add(Dense(units=time_window_size, activation='linear'))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metric])
        print(model.summary())
        return model

    def load_model(self, model_dir_path):
        config_file_path = LstmAutoEncoder.get_config_file(model_dir_path)
        self.config = np.load(config_file_path).item()
        self.metric = self.config['metric']
        self.time_window_size = self.config['time_window_size']
        self.threshold = self.config['threshold']
        self.model = LstmAutoEncoder.create_model(self.time_window_size, self.metric)
        weight_file_path = LstmAutoEncoder.get_weight_file(model_dir_path)
        self.model.load_weights(weight_file_path)

    @staticmethod
    def get_config_file(model_dir_path):
        return model_dir_path + '/' + LstmAutoEncoder.model_name + '-config.npy'

    @staticmethod
    def get_weight_file(model_dir_path):
        return model_dir_path + '/' + LstmAutoEncoder.model_name + '-weights.h5'

    @staticmethod
    def get_architecture_file(model_dir_path):
        return model_dir_path + '/' + LstmAutoEncoder.model_name + '-architecture.json'

    def fit(self, timeseries_dataset, model_dir_path, time_window_size = 50, batch_size=None, epochs=None, validation_split=None, metric=None,
            estimated_negative_sample_ratio=None):
        if batch_size is None:
            batch_size = 8
        if epochs is None:
            epochs = 20
        if validation_split is None:
            validation_split = 0.2
        if metric is None:
            metric = 'mean_absolute_error'
        if estimated_negative_sample_ratio is None:
            estimated_negative_sample_ratio = 0.9

        self.metric = metric
        self.time_window_size = time_window_size#timeseries_dataset.shape[1]
        print timeseries_dataset.shape

        timeseries_dataset = self.genTranSet(timeseries_dataset, time_window_size)
        print timeseries_dataset.shape
        input_timeseries_dataset = np.expand_dims(timeseries_dataset, axis=2)

        print timeseries_dataset.shape
        print input_timeseries_dataset.shape

        weight_file_path = LstmAutoEncoder.get_weight_file(model_dir_path=model_dir_path)
        architecture_file_path = LstmAutoEncoder.get_architecture_file(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        self.model = LstmAutoEncoder.create_model(self.time_window_size, metric=self.metric)
        open(architecture_file_path, 'w').write(self.model.to_json())
        self.model.fit(x=input_timeseries_dataset, y=timeseries_dataset,
                       batch_size=batch_size, epochs=epochs,
                       verbose=LstmAutoEncoder.VERBOSE, validation_split=validation_split,
                       callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)

        scores = self.predict(timeseries_dataset)
        scores.sort()
        cut_point = int(estimated_negative_sample_ratio * len(scores))
        self.threshold = scores[cut_point]

        print('estimated threshold is ' + str(self.threshold))

        self.config = dict()
        self.config['time_window_size'] = self.time_window_size
        self.config['metric'] = self.metric
        self.config['threshold'] = self.threshold
        config_file_path = LstmAutoEncoder.get_config_file(model_dir_path=model_dir_path)
        np.save(config_file_path, self.config)

    def predict(self, timeseries_dataset):
        input_timeseries_dataset = np.expand_dims(timeseries_dataset, axis=2)
        target_timeseries_dataset = self.model.predict(x=input_timeseries_dataset)
        print target_timeseries_dataset[:,:]
        dist = np.linalg.norm(timeseries_dataset - target_timeseries_dataset, axis=-1)
        return dist

    def anomaly(self, timeseries_dataset, threshold=None):
        if threshold is not None:
            self.threshold = threshold

        dist = self.predict(timeseries_dataset)
        return zip(dist >= self.threshold, dist)


class CnnLstmAutoEncoder(object):
    model_name = 'cnn-lstm-auto-encoder'
    VERBOSE = 1

    def __init__(self):
        self.model = None
        self.time_window_size = None
        self.config = None
        self.metric = None
        self.threshold = None

    @staticmethod
    def create_model(time_window_size, metric):
        model = Sequential()

        model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu',
                         input_shape=(time_window_size, 1)))
        model.add(MaxPooling1D(pool_size=4))

        model.add(LSTM(64))

        model.add(Dense(units=time_window_size, activation='linear'))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metric])

        # model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metric])
        # model.compile(optimizer="sgd", loss="mse", metrics=[metric])

        print(model.summary())
        return model

    def load_model(self, model_dir_path):
        config_file_path = CnnLstmAutoEncoder.get_config_file(model_dir_path)
        self.config = np.load(config_file_path).item()
        self.metric = self.config['metric']
        self.time_window_size = self.config['time_window_size']
        self.threshold = self.config['threshold']
        self.model = CnnLstmAutoEncoder.create_model(self.time_window_size, self.metric)
        weight_file_path = CnnLstmAutoEncoder.get_weight_file(model_dir_path)
        self.model.load_weights(weight_file_path)

    @staticmethod
    def get_config_file(model_dir_path):
        return model_dir_path + '/' + CnnLstmAutoEncoder.model_name + '-config.npy'

    @staticmethod
    def get_weight_file(model_dir_path):
        return model_dir_path + '/' + CnnLstmAutoEncoder.model_name + '-weights.h5'

    @staticmethod
    def get_architecture_file(model_dir_path):
        return model_dir_path + '/' + CnnLstmAutoEncoder.model_name + '-architecture.json'

    def fit(self, timeseries_dataset, model_dir_path, batch_size=None, epochs=None, validation_split=None, metric=None,
            estimated_negative_sample_ratio=None):
        if batch_size is None:
            batch_size = 8
        if epochs is None:
            epochs = 20
        if validation_split is None:
            validation_split = 0.2
        if metric is None:
            metric = 'mean_absolute_error'
        if estimated_negative_sample_ratio is None:
            estimated_negative_sample_ratio = 0.9

        self.metric = metric
        self.time_window_size = timeseries_dataset.shape[1]

        input_timeseries_dataset = np.expand_dims(timeseries_dataset, axis=2)

        weight_file_path = CnnLstmAutoEncoder.get_weight_file(model_dir_path=model_dir_path)
        architecture_file_path = CnnLstmAutoEncoder.get_architecture_file(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        self.model = CnnLstmAutoEncoder.create_model(self.time_window_size, metric=self.metric)
        open(architecture_file_path, 'w').write(self.model.to_json())
        self.model.fit(x=input_timeseries_dataset, y=timeseries_dataset,
                       batch_size=batch_size, epochs=epochs,
                       verbose=CnnLstmAutoEncoder.VERBOSE, validation_split=validation_split,
                       callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)

        scores = self.predict(timeseries_dataset)
        scores.sort()
        cut_point = int(estimated_negative_sample_ratio * len(scores))
        self.threshold = scores[cut_point]

        print('estimated threshold is ' + str(self.threshold))

        self.config = dict()
        self.config['time_window_size'] = self.time_window_size
        self.config['metric'] = self.metric
        self.config['threshold'] = self.threshold
        config_file_path = CnnLstmAutoEncoder.get_config_file(model_dir_path=model_dir_path)
        np.save(config_file_path, self.config)

    def predict(self, timeseries_dataset):
        input_timeseries_dataset = np.expand_dims(timeseries_dataset, axis=2)
        target_timeseries_dataset = self.model.predict(x=input_timeseries_dataset)
        dist = np.linalg.norm(timeseries_dataset - target_timeseries_dataset, axis=-1)
        return dist

    def anomaly(self, timeseries_dataset, threshold=None):
        if threshold is not None:
            self.threshold = threshold

        dist = self.predict(timeseries_dataset)
        return zip(dist >= self.threshold, dist)


class BidirectionalLstmAutoEncoder(object):
    model_name = 'bidirectional-lstm-auto-encoder'
    VERBOSE = 1

    def __init__(self):
        self.model = None
        self.time_window_size = None
        self.config = None
        self.metric = None
        self.threshold = None

    @staticmethod
    def create_model(time_window_size, metric):
        model = Sequential()

        model.add(Bidirectional(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2), input_shape=(time_window_size, 1)))

        model.add(Dense(units=time_window_size, activation='linear'))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metric])

        # model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metric])
        # model.compile(optimizer="sgd", loss="mse", metrics=[metric])

        print(model.summary())
        return model

    def load_model(self, model_dir_path):
        config_file_path = BidirectionalLstmAutoEncoder.get_config_file(model_dir_path)
        self.config = np.load(config_file_path).item()
        self.metric = self.config['metric']
        self.time_window_size = self.config['time_window_size']
        self.threshold = self.config['threshold']
        self.model = BidirectionalLstmAutoEncoder.create_model(self.time_window_size, self.metric)
        weight_file_path = BidirectionalLstmAutoEncoder.get_weight_file(model_dir_path)
        self.model.load_weights(weight_file_path)

    @staticmethod
    def get_config_file(model_dir_path):
        return model_dir_path + '/' + BidirectionalLstmAutoEncoder.model_name + '-config.npy'

    @staticmethod
    def get_weight_file(model_dir_path):
        return model_dir_path + '/' + BidirectionalLstmAutoEncoder.model_name + '-weights.h5'

    @staticmethod
    def get_architecture_file(model_dir_path):
        return model_dir_path + '/' + BidirectionalLstmAutoEncoder.model_name + '-architecture.json'

    def fit(self, timeseries_dataset, model_dir_path, batch_size=None, epochs=None, validation_split=None, metric=None,
            estimated_negative_sample_ratio=None):
        if batch_size is None:
            batch_size = 8
        if epochs is None:
            epochs = 20
        if validation_split is None:
            validation_split = 0.2
        if metric is None:
            metric = 'mean_absolute_error'
        if estimated_negative_sample_ratio is None:
            estimated_negative_sample_ratio = 0.9

        self.metric = metric
        self.time_window_size = timeseries_dataset.shape[1]

        input_timeseries_dataset = np.expand_dims(timeseries_dataset, axis=2)

        weight_file_path = BidirectionalLstmAutoEncoder.get_weight_file(model_dir_path=model_dir_path)
        architecture_file_path = BidirectionalLstmAutoEncoder.get_architecture_file(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        self.model = BidirectionalLstmAutoEncoder.create_model(self.time_window_size, metric=self.metric)
        open(architecture_file_path, 'w').write(self.model.to_json())
        self.model.fit(x=input_timeseries_dataset, y=timeseries_dataset,
                       batch_size=batch_size, epochs=epochs,
                       verbose=BidirectionalLstmAutoEncoder.VERBOSE, validation_split=validation_split,
                       callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)

        scores = self.predict(timeseries_dataset)
        scores.sort()
        cut_point = int(estimated_negative_sample_ratio * len(scores))
        self.threshold = scores[cut_point]

        print('estimated threshold is ' + str(self.threshold))

        self.config = dict()
        self.config['time_window_size'] = self.time_window_size
        self.config['metric'] = self.metric
        self.config['threshold'] = self.threshold
        config_file_path = BidirectionalLstmAutoEncoder.get_config_file(model_dir_path=model_dir_path)
        np.save(config_file_path, self.config)

    def predict(self, timeseries_dataset):
        input_timeseries_dataset = np.expand_dims(timeseries_dataset, axis=2)
        target_timeseries_dataset = self.model.predict(x=input_timeseries_dataset)
        dist = np.linalg.norm(timeseries_dataset - target_timeseries_dataset, axis=-1)
        return dist

    def anomaly(self, timeseries_dataset, threshold=None):
        if threshold is not None:
            self.threshold = threshold

        dist = self.predict(timeseries_dataset)
        return zip(dist >= self.threshold, dist)

#from keras_anomaly_detection.library.recurrent import LstmAutoEncoder


def rolling_window_update(old, new, w=100):
    """
    :param old: Old data
    :param new: New data
    :param w: Controls the size of the rolling window
    :return: The w most recent datapoints from the concatenation of old and new
    >>> rolling_window_update(old=[1,2,3], new=[4,5,6,7],w=5)
    array([3,4,5,6,7])

    """
    out = np.concatenate((old, new))
    if len(out) > w:
        out = out[(len(out)-w):]
    return out



class Percentile1D(BaseEstimator, BaseDetector):

    def __init__(
        self,
        ff=1.0,
        window_size=300,
        threshold=0.99
    ):
        self.ff = ff
        self.window_size = window_size
        self.threshold = threshold
        self.sample_ = []

    def fit(self, x):
        x = pd.Series(x)
        self.__setattr__('sample_', x[:int(np.floor(self.window_size))])

    def update(self, x):  # allows mini-batch
        x = pd.Series(x)
        window = rolling_window_update(
            old=self.sample_, new=x,
            w=int(np.floor(self.window_size))
        )
        self.__setattr__('sample_', window)

    def score_anomaly(self, x):
        x = pd.Series(x)
        scores = pd.Series([0.01*percentileofscore(self.sample_, z) for z in x])
        return scores

    def get_anomalies(self, x):
        return decision_rule(self.score_anomaly(x), self.threshold)

class LIDetector(BaseDetector):
	def __init__(self):
		super(LIDetector, self).__init__()

	def get_anomalies(self):
		return self.detector.get_anomalies()

	def get_all_scores(self):
		return self.detector.get_all_scores()

	def load_series(self, ts):
		ts = self.cvtTimeSeries(ts)
		self.detector.time_series = ts

	def detect_anomalies(self, ts):
		self.detector.time_series = self.cvtTimeSeries(ts)
		self.detector._detect(False)
		return self.detector.get_anomalies()

	def cvtTimeSeries(self, ts):
		if not isinstance(ts, TimeSeries):
			if isinstance(ts, list):
				tmp = {}
				for i, item in enumerate(ts):
					tmp[i] = item
				return TimeSeries(tmp)
		return ts

	def read_csv(self, csv_name):
	    """
	    Read data from a csv file into a dictionary.
	    :param str csv_name: path to a csv file.
	    :return dict: a dictionary represents the data in file.
	    """
	    data = {}
	    if int(sys.version[0]) == 2:
	        str_types = (str, unicode)
	    else:
	        str_types = (bytes, str)
	    if not isinstance(csv_name, str_types):
	        raise exceptions.InvalidDataFormat('luminol.utils: csv_name has to be a string!')
	    with open(csv_name, 'r') as csv_data:
	        reader = csv.reader(csv_data, delimiter=',', quotechar='|')
	        for row in reader:
	            try:
	                key = to_epoch(row[0])
	                value = float(row[1])
	                data[key] = value
	            except ValueError:
	                pass
	    return data

class BitmapD(LIDetector):
	"""docstring for BitmapD"""
	def __init__(self, ts = {}, param = None):
		#pass
		#super(AnomalyDetector, self).__init__(ts, algorithm_name = 'bitmap_detector')
		self.detector = AnomalyDetector(self.cvtTimeSeries(ts), algorithm_name = 'bitmap_detector', algorithm_params = param)

class DerivativeD(LIDetector):
	"""docstring for DerivativeD"""
	def __init__(self, ts = {}, param = None):
		super(DerivativeD, self).__init__()
		self.detector = AnomalyDetector(self.cvtTimeSeries(ts), algorithm_name = 'derivative_detector', algorithm_params = param)


class ExpAvgD(LIDetector):
	"""docstring for ExpAvgD"""
	def __init__(self, ts = {}, param = None):
		super(ExpAvgD, self).__init__()
		self.detector = AnomalyDetector(self.cvtTimeSeries(ts), algorithm_name = 'exp_avg_detector', algorithm_params = param)

class AbsThresholdD(LIDetector):
	"""docstring for AbsThresholdD"""
	def __init__(self, ts = {}, param = None):
		super(AbsThresholdD, self).__init__()
		self.detector = AnomalyDetector(self.cvtTimeSeries(ts), algorithm_name = 'absolute_threshold', algorithm_params = param)


class DiffPercentD(LIDetector):
	"""docstring for DiffPercentD"""
	def __init__(self, ts = {}, param = None):
		super(DiffPercentD, self).__init__()
		print self.cvtTimeSeries(ts)
		self.detector = AnomalyDetector(self.cvtTimeSeries(ts), algorithm_name = 'diff_percent_threshold', algorithm_params = param)


class SignTestD(LIDetector):
	"""docstring for SignTestD"""
	def __init__(self, ts = {}, param = None):
		super(SignTestD, self).__init__()
		self.detector = AnomalyDetector(self.cvtTimeSeries(ts), algorithm_name = 'sign_test', algorithm_params = param)


#class DNNDetector():

class ShesdD(object):
	"""docstring for ShesdD"""
	def __init__(self, ts = {}, param = None):
		super(ShesdD, self).__init__()
		self.param = param
		print "init"
		
	def mergeAnomaly(self, anoms, mergeLen = 10):
		accum_num = 0
		find = 0
		ret = []
		l = len(anoms)
		if l == 0:
			return []
		timediff = pd.Timedelta('15s') * 30

		base = 10**9
		for i, x in enumerate(anoms.index):
			if x - anoms.index[i-1] > timediff and i > 0:
				ret.append(Anomaly(anoms.index[find].value/base, anoms.index[i-1].value/base, 0, 0))
				find = i
		print find
		ret.append(Anomaly(anoms.index[find].value/base, anoms.index[l-1].value/base, 0, 0))
		return ret

	def get_anomalies(self):
		return self.anoms

	def get_all_scores(self):
		return self.detector.get_all_scores()

	def load_series(self, ts):
		ts = cvt2s(ts)
		self.time_series = ts

	def detect_anomalies(self, ts):
		ts = cvt2s(ts)
		anoms = anomaly_detect_ts(ts, max_anoms=0.02, direction="both", plot=False, alpha = 0.05)
		#print anoms
		tmp = anoms['anoms'].sort_index(ascending=True)
		#print tmp
		self.anoms = self.mergeAnomaly(tmp)
		return self.anoms
		

'''
class DnnD(object):
	"""docstring for DnnD"""
	def __init__(self):
		super(DnnD, self).__init__()
		model_dir_path = './models'
		self.ae = LstmAutoEncoder()
		ae.load_model(model_dir_path)
		
	def fit(self):
		ae.fit(np_data[:20, :], model_dir_path=model_dir_path, estimated_negative_sample_ratio=0.9)


	def detect_anomalies(self):
		anomaly_information = ae.anomaly(ecg_np_data[:20, :])

'''

def testHere():
	print "test"

if __name__ == '__main__':
	test1 = [1,1,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,0,6,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,1]
	test = [1,1,1,1,0,0,3,5,5,5,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,5,1,1,0,1,1,0,1,1,0,1,0,0,0,1,0,1,1,0,1,5,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,1,1]
	'''
	test = cvt2s(readcsv2list('../../0.084_0.244_0.600_1.520_5.000_4.189.csv'))
	print test.axes
	xx = ShesdD()
	#print xx.detect_anomalies(test)
	print [a.get_time_window() for a in xx.detect_anomalies(test)]
	'''
	xx = ShesdD()
	a = xx.detect_anomalies(test)
	print [a.get_time_window() for a in xx.get_anomalies()]


