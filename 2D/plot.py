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

	def plot_scan(self, arr, text):
		# plot scan (arr) of size (channels x height x width)
		plt.imshow(arr[0], cmap='Greys_r')
		plt.savefig(self.savedir+'scan_'+text+'.png')
		plt.show()
		plt.close()

	def plot_scan_mask_gray(self, inp, tar, text):
		plt.imshow(inp[0], cmap='Greys_r')
		plt.imshow(tar[0], alpha=.5, cmap='Reds')
		plt.savefig(self.savedir+'scanmask_'+text+'.png')
		plt.show()
		plt.close()

	def plot_scan_mask(self, inp, tar, text):
		# create rgb channels for mask
		mask = tar[0].reshape(*tar[0].shape,1)
		mask = np.concatenate([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=-1) 
		mask[np.where(mask == 0)] = np.nan
		
		plt.imshow(inp[0], cmap='Greys_r')
		plt.imshow(mask, alpha=.5)
		plt.savefig(self.savedir+'scanmask_'+text+'.png')
		plt.show()
		plt.close()

class PlotResults:
	def __init__(self, savedir, savetext=''):
		sns.set()
		sns.set(rc={'figure.figsize':(5,3)})
		sns.set_context('paper')
		sns.set_style('white')

		self.savedir = savedir
		self.savetext = savetext

	def plot_history(self, history, m):
		sns.lineplot(data=history, x=history.index, y="value", hue="split")
		plt.xlabel('epoch')
		plt.ylabel(m)
		plt.savefig(self.savedir+'/history_'+m+'.pdf', bbox_inches='tight')
		plt.close()