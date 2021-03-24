import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

class PlotData:
	def __init__(self, savedir):
		sns.set()
		sns.set_context('paper')
		sns.set_style('white')

		self.savedir = savedir

	def plot_scan(self, arr, fname, text):
		# plot scan (arr) of size (channels x height x width)
		plt.imshow(arr[0,0], cmap='Greys_r')
		plt.savefig(self.savedir+fname+'_'+text+'.png')
		plt.show()
		plt.close()

class PlotResults:
	def __init__(self, savedir):
		sns.set()
		sns.set(rc={'figure.figsize':(6,3)})
		sns.set_context('paper')
		sns.set_style('white')

		self.savedir = savedir

	def plot_history(self, history, m, epoch_max=1000):
		sns.lineplot(data=history, x=history.index, y="value", hue="split", lw=.1)
		plt.xlabel('epoch')
		plt.ylabel(m)
		plt.xlim((0,epoch_max))
		plt.savefig(self.savedir+'/history_'+m+'.pdf', bbox_inches='tight')
		plt.close()