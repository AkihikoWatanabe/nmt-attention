# coding=utf-8

import cPickle

class Backup:
	@staticmethod
	def dump(frms, file_path):
		opened = open(file_path, 'w')
		for frm in frms:
			cPickle.dump(frm, opened)
		opened.close()
		
	@staticmethod
	def load(file_path):
		opened = open(file_path, 'r')
		load_data = []
		while True:
			try:
				load_data.append(cPickle.load(opened))
			except EOFError:
				break
		opened.close()
		if len(load_data)==1:
			return load_data[0]
		return load_data
