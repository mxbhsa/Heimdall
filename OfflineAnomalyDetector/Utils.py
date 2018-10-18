import pandas as pd
import os

def listpath(path):
	dirs = os.listdir( path )
	ret = []
	for file in dirs:
		if not ".csv" in file:
			continue
		fullfn = os.path.join(path, file)
		ret.append(fullfn)
	return ret

def readcsv2list(fn):
	df2 = pd.read_csv(fn,header=None)
	df2 = df2.T
	a = df2[1:2]

	a = a.values.tolist()[0]
	return a

def cvt2s(ts, freq = '1s'):
	if not isinstance(ts, pd.Series):
		if isinstance(ts, list):
			ser1 = pd.Series(ts)
			ser1.index=pd.DatetimeIndex(freq=freq,start=0,periods=len(ts))
			return ser1
		'''
		elif isinstance(ts, pd.DataFrame):
			ts = pd.Series(ts.values)
			return 
		'''
	return ts


if __name__ == '__main__':
	x = [1,2,3]
	print cvt2s(x)