# -*- encoding:utf-8 -*-
import pandas as pd
import os
import Utils
import ray
import happybase, time

class DataFetcher(object):
	"""docstring for DataFetcher"""
	def __init__(self):
		super(DataFetcher, self).__init__() 
		self.pool = []
		#ray.init()

	def _fetchData(self):
		pass

	def getData(self):
		pass


class CsvFetcher(DataFetcher):
	"""docstring for CsvFetcher"""
	def __init__(self, csvdir, batchSize = 10, startTime = 0):
		super(CsvFetcher, self).__init__()
		self.starKeys = Utils.listpath(csvdir)
		self.batchSize = batchSize
		self.startTime = startTime

	def _fetchData(self):
		count = self.batchSize 
		while count > 0 and any(self.starKeys):
			fn = self.starKeys.pop()
			data = Utils.readcsv2list(fn)
			job = {"name" : os.path.basename(fn), "data" : data}
			#jobId = ray.put(job)
			self.pool.append(job) 
			count -= 1

	def getData(self, num = 1):
		#print self.r.llen(self.dbKey)
		ret = []
		count = num
		while count > 0 :
			if not any(self.pool):
				self._fetchData()
			if any(self.pool):
				ret.append(self.pool.pop())
			else:
				ret.append(None)
			count -= 1
		if num == 1:
			return ret[0]
		return ret

class HBaseFetcher(DataFetcher):
	"""docstring for HBaseFetcher"""
	def __init__(self, ip = '10.0.82.111', port = 9090, batchSize = 10, startTime = None,  endTime = None, startlist = None):
		self.batchSize = batchSize

		self.startTime = str(startTime)
		self.endTime = str(endTime)
		self.connection = happybase.Connection(ip,port)
		self.connection.open()
		self.starKeys = startlist or self._getStarList() 
		self.pool = []
		self.magTableColumn = "cf1:c"
		print self.starKeys
		print self.connection.tables()

	def _getStarList(self):

		tb = self.connection.table("startemplate")

		xx = tb.scan(row_start=None, row_stop=None,
		 row_prefix=None, columns=['cf1:c1'], filter=None, 
		 timestamp=None, include_timestamp=False, 
		 batch_size=10000, scan_batching=None, limit=None, 
		 sorted_columns=False, reverse=False)

		ret = [x[0][:-2] for x in xx]

		return ret

	def parseRegion(self, ans, region_star_id):
		for star in ans.split("*"):
			tstar = star.split("_")
			if tstar[0] == region_star_id:
				return float(tstar[1])
		return 0

	def _getSeries(self, table, starId):
		tb = self.connection.table(table)
		#print tb.families()
		#print tb.regions()
		region_id = starId.split("_")[-2]
		region_star_id = starId.split("_")[-1]
		row_start = region_id+"_"+self.startTime
		row_stop = region_id+"_"+self.endTime+"_"

		xx = tb.scan(row_start=row_start, row_stop=row_stop,
		 row_prefix=None, columns=[self.magTableColumn], filter=None, 
		 timestamp=None, include_timestamp=False, 
		 batch_size=1000, scan_batching=None, limit=None, 
		 sorted_columns=False, reverse=False)
		ret = []
		try:
			ret = [self.parseRegion(x[1][self.magTableColumn], region_star_id) for x in xx]
		except Exception as e:
			print 'invalid star'+str(e)
		else:
			pass
		x[0].split("_")[-1]
		return x[0].split("_")[-1], ret

	def getData(self, num = 1):
		ret = []
		count = num
		while count > 0 :
			if not any(self.pool):
				self._fetchData()
			if any(self.pool):
				ret.append(self.pool.pop())
			else:
				ret.append(None)
			count -= 1
		if num == 1:
			return ret[0]
		return ret

	def _fetchData(self):
		count = self.batchSize 
		while count > 0 and any(self.starKeys):
			fn = self.starKeys.pop()
			#data = Utils.readcsv2list(fn)
			ccd, data = self._getSeries('magTable', fn)
			print fn, data
			job = {"name" : os.path.basename(fn+"_"+ccd), "data" : data}
			#jobId = ray.put(job)
			self.pool.append(job) 
			count -= 1


if __name__ == '__main__':
	#x = CsvFetcher("../../data")
	#print x.getData()
	x = HBaseFetcher(ip = '10.0.82.111', port = 9090, startlist = ['1666899_2027'], startTime = 1121239,  endTime = 1121253)
	print x.getData()
		