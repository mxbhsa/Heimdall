# -*- encoding:utf-8 -*-
import pandas as pd
import os
import Utils
import ray
import happybase, time

class DataWriter(object):
	"""docstring for DataWriter"""
	def __init__(self):
		super(DataWriter, self).__init__() 
		self.pool = []

	def writeData(self):
		pass


class CsvFetcher(DataWriter):
	"""docstring for CsvFetcher"""
	def __init__(self, csvdir, batchSize = 10):
		super(CsvFetcher, self).__init__()
		self.starKeys = Utils.listpath(csvdir)
		self.batchSize = batchSize

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

class HBaseWriter(DataWriter):
	"""docstring for HBaseWriter"""
	def __init__(self, ip = '10.0.82.111', port = 9090, startTime = None, anomalyTable = None):
		self.connection = happybase.Connection(ip,port)
		self.connection.open()
		self.pool = []
		self.magTableColumn = "cf1:c"
		self.anomalyTable = anomalyTable

		self.tables = self.connection.tables()
		if not self.anomalyTable in self.tables:
			self.connection.create_table(self.anomalyTable, {"cf1:c" : None})

	def writeAnomaly(self, starId, baseTime, startTime, endTime, data):
		tb = self.connection.table(self.anomalyTable)
		region_id, star_region_id, ccd = starId.split("_")
		data = "*".join(['{:.2f}'.format(x) for x in data])
		row = "_".join([region_id, str(baseTime + startTime),  str(baseTime + endTime), ccd, star_region_id])
		anomaly_data = {'cf1:c': data}
		tb.put(row=row, data=anomaly_data)




if __name__ == '__main__':
	#x = CsvFetcher("../../data")
	#print x.getData()
	x = HBaseWriter(ip = '10.0.82.111', port = 9090, anomalyTable = "offlineAnomalytest")
	print x.writeAnomaly("1666899_2027_1*1", 1440052, 0, 2, [1.1,2.3,1.2])
		