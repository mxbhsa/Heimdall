# -*- encoding:utf-8 -*-
import pandas as pd
import os
import Utils
import redis
from abc import abstractmethod

class BaseWriter(object):
	"""docstring for BaseWriter"""
	def __init__(self):
		super(BaseWriter, self).__init__()
		#ray.init()

	@abstractmethod
	def write(self):
		pass


class CsvWriter(BaseWriter):
	"""docstring for CsvWriter"""
	def __init__(self, rootdir):
		super(CsvWriter, self).__init__()
		self.rootdir = rootdir

	def write(self, fn):
		f = open(os.path.join(self.rootdir, fn))
		

class HBaseWriter(object):
	"""docstring for HBaseWriter"""
	def __init__(self):
		super(HBaseWriter, self).__init__()

	def write(self):
		pass
		
		
