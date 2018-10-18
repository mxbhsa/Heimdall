import ray
from Fetchers import CsvFetcher, HBaseFetcher
from Detectors import ExpAvgD, ShesdD
from DataWriters import HBaseWriter
import time, sys
import ConfigParser


@ray.remote
class Actor(object):
	def __init__(self, detector):
		self.value = 1
		self.det = detector

	def execute(self, ts):
		if ts == None:
			return None
		#return ts['name']
		a = self.det.detect_anomalies(ts['data'])
		return [a.get_time_window() for a in self.det.get_anomalies()]
		

class Dispatcher(object):
	"""docstring for Dispatcher"""
	def __init__(self, detector, datafetcher, datawriter = None,  numthreads = 0):
		#super(Dispatcher, self).__init__()
		#self.det = detector
		self.numthreads = numthreads
		self.daf = datafetcher
		self.daw = datawriter
		self.actors = [Actor.remote(detector) for _ in range(numthreads)]

	def run(self):
		while 1:
			data = self.daf.getData(self.numthreads)
		 	results = ray.get([self.actors[i].execute.remote(data[i]) for i in range(self.numthreads) ])
		 	#print results
		 	if self.daw:
		 		for i, result in enumerate(results):
		 			if result:
			 			for anomaly in result:
			 				#print data[i]['name'], self.daf.startTime, anomaly[0], anomaly[1]
			 				#print data[i]['data'][anomaly[0]: anomaly[1]+1]
			 				self.daw.writeAnomaly(data[i]['name'], self.daf.startTime, anomaly[0], anomaly[1],data[i]['data'][anomaly[0]: anomaly[1]+1])
		 	if None in results:
		 		break
		 		


if __name__ == '__main__':
	
	if len(sys.argv) < 5 :
		print "Usage: [startlist_file_name] [config_file_name] [startTime] [endTime] "
		exit()
	fn = sys.argv[1]
	startlist = []
	with open(fn, 'r') as f:
		for line in f:
			startlist.append(line.strip())
	Config = ConfigParser.ConfigParser()

	Config.read(sys.argv[2])

	db_host = Config.get("db", "ip")
	db_port = Config.get("db", "hbase_thrift_port")
	num_thread = Config.get('detector', "num_thread")

	ray.init()
	dt = ShesdD()
	
	df = HBaseFetcher(ip = db_host, port = int(db_port), startlist = startlist, startTime = int(sys.argv[3]),  endTime = int(sys.argv[4]))
	dw = HBaseWriter(ip = db_host, port = int(db_port), anomalyTable = "offlineAnomalytest")
	a2 = Dispatcher(dt, df, dw, int(num_thread))
	a2.run()


