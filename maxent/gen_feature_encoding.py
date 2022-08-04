import re
from nltk import TaggerI, FreqDist, untag
from nltk.classify.maxent import BinaryMaxentFeatureEncoding
from nltk.classify.megam import call_megam, parse_megam_weights, write_megam_file, config_megam
from collections import defaultdict
import sys, os
import tempfile
import numpy as np
import json
from tqdm import tqdm
from sklearn.cluster import KMeans
import pickle
import umap
import plotly.express as px

PATH_TO_MEGAM_EXECUTABLE = "/mnt/c/Users/Asus/Downloads/megam-64.opt"
config_megam(PATH_TO_MEGAM_EXECUTABLE)

def main():
	assert len(sys.argv) > 2, "Need to specify training file, n_clusters as command line argument"
	train_file = sys.argv[1]
	n_clusters = int(sys.argv[2])
	train_sents, unique_tags = preprocess(train_file)
	print(f"Found {len(unique_tags)} unique_tags:", unique_tags)
	f = FeatureExtractor(train_sents)

	word_embs = np.zeros((len(f.featuresets), len(f.encoding.labels())), dtype=np.float16)
	for i,x in tqdm(enumerate(f.featuresets)):
		for j,label in enumerate(f.encoding.labels()):
			v = f.encoding.encode(x[0], label)
			total = np.float16(0.0)
			for (f_id, f_val) in v:
				total += f.weights[f_id]*f_val
			word_embs[i][j] = total

	# standard_embedding = umap.UMAP(random_state=42).fit_transform(feature_embs.reshape(-1,1))
	kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(word_embs)
	labels = kmeans.predict(word_embs)
	print(labels.shape)
	print(labels[:10])
	# px.scatter(x=standard_embedding[:,0].reshape(-1,1), y=standard_embedding[:,1].reshape(-1,1), color=kmeans_labels.reshape(-1,1))
	np.save(f"{n_clusters}/feature_.npy", f.weights)
	np.save(f"{n_clusters}/labels_.npy", f.encoding.labels())
	with open(f"{n_clusters}/kmeans_.pkl", "wb") as f:
		pickle.dump(kmeans, f)
	# kmeans = KMeans(n_clusters=n_clusters).fit(feature_embs)



def preprocess(train_file):
	if os.path.exists(train_file[:-7]+'toks.json'):
		print('toks already saved')
		return json.load(open(train_file[:-7]+'toks.json')), json.load(open(train_file[:-7]+'unique_tags.json'))
	f = open(train_file)
	sentences = []
	sentence = []
	unique_tags = set()
	for x in f.readlines():
		if x.startswith('#'):
			continue
		elif x == '\n':
			sentences.append(sentence)
			sentence = []
			continue
		# '3\tGoogle\tGoogle\tPROPN\tNNP\tNumber=Sing\t4\tnsubj\t4:nsubj\t_\n'
		x = x.split('\t')
		# id	WORD		word	POS		POS2	descr	connected_to connection	ex		ex2\n
		# ['3', 'Google', 'Google', 'PROPN', 'NNP', 'Number=Sing', '4', 'nsubj', '4:nsubj', '_\n']
		if x[3] in ['PUNCT', '_'] :
			continue
		sentence.append((x[2],x[3]))
		unique_tags.add(x[3])
	x = open(train_file[:-7]+'toks.json','w')
	u = open(train_file[:-7]+'unique_tags.json','w')
	json.dump(sentences, x)
	json.dump(list(unique_tags), u)
	return sentences, unique_tags


class FeatureExtractor:
	def __init__(self, train_sents, labels=None, algorithm='megam', rare_word_cutoff=5,
			rare_feat_cutoff=5, uppercase_letters='[A-Z]', trace=2, train_megam=True, **cutoffs):
		self.uppercase_letters = uppercase_letters
		self.word_freqdist = self.gen_word_freqs(train_sents)
		self.featuresets = self.gen_featsets(train_sents, rare_word_cutoff)
		self.features_freqdist = self.gen_feat_freqs(self.featuresets)
		self.cutoff_rare_feats(self.featuresets, rare_feat_cutoff)
		
		print(len(self.featuresets))
		# from train_maxent_classifier_with_megam
		explicit = True
		bernoulli = True
		if "explicit" in cutoffs:
			explicit = cutoffs["explicit"]
		if "bernoulli" in cutoffs:
			bernoulli = cutoffs["bernoulli"]
		gaussian_prior_sigma=0

		count_cutoff = cutoffs.get("count_cutoff", 0)
		# self.classifier = MaxentClassifier.train(self.featuresets, algorithm, trace, **cutoffs)
		self.encoding = BinaryMaxentFeatureEncoding.train(self.featuresets, count_cutoff, labels=labels, alwayson_features=True)
		if train_megam:
			try:
				fd, trainfile_name = tempfile.mkstemp(prefix="nltk-")
				with open(trainfile_name, "w") as trainfile:
					write_megam_file(self.featuresets, self.encoding, trainfile, explicit=explicit, bernoulli=bernoulli)
					os.close(fd)
			except (OSError, ValueError) as e:
				raise ValueError("Error while creating megam training file: %s" % e) from e

			# Run megam on the training file.
			options = []
			options += ["-nobias", "-repeat", "10"]
			if explicit:
				options += ["-explicit"]
			if not bernoulli:
				options += ["-fvals"]
			if gaussian_prior_sigma:
				# Lambda is just the precision of the Gaussian prior, i.e. it's the
				# inverse variance, so the parameter conversion is 1.0/sigma**2.
				# See https://users.umiacs.umd.edu/~hal/docs/daume04cg-bfgs.pdf
				inv_variance = 1.0 / gaussian_prior_sigma ** 2
			else:
				inv_variance = 0
			options += ["-lambda", "%.2f" % inv_variance, "-tune"]
			if trace < 3:
				options += ["-quiet"]
			if "max_iter" in cutoffs:
				options += ["-maxi", "%s" % cutoffs["max_iter"]]
			if "ll_delta" in cutoffs:
				# [xx] this is actually a perplexity delta, not a log
				# likelihood delta
				options += ["-dpp", "%s" % abs(cutoffs["ll_delta"])]
			if hasattr(self.encoding, "cost"):
				options += ["-multilabel"]  # each possible la
			options += ["multiclass", trainfile_name]
			stdout = call_megam(options)
			# print(stdout)
			# print('./megam_i686.opt ', ' '.join(options))
			# Delete the training file
			try:
				os.remove(trainfile_name)
			except OSError as e:
				print(f"Warning: unable to delete {trainfile_name}: {e}")

			# Parse the generated weight vector.
			self.weights = parse_megam_weights(stdout, self.encoding.length(), explicit)

			# Convert from base-e to base-2 weights.
			self.weights *= np.log2(np.e)



	def gen_feat_freqs(self, featuresets):
		"""
		Generates a frequency distribution of joint features (feature, tag)
		tuples. The frequency distribution will be used by the tagger to
		determine which (rare) features should not be considered during
		training (feature cutoff).
		This is how joint features look like::
			(('t-2 t-1', 'IN DT'), 'NN')
			(('w-2', '<START>'), 'NNP')
			(('w+1', 'of'), 'NN')
		@type featuresets: {list} of C{tuples} of (C{dict}, C{str})
		@param featuresets: a list of tuples that contain the featureset of a
		word from the training set and its POS tag.
		@rtype: C{FreqDist}
		@return: a L{frequency distribution<nltk.FreqDist()>},
		counting how often each (context information feature, tag) tuple occurs
		in the training sentences.
		"""
		features_freqdist = defaultdict(int)
		for (feat_dict, tag) in featuresets:
			for (feature, value) in feat_dict.items():
				features_freqdist[ ((feature, value), tag) ] += 1
		return features_freqdist

	def gen_word_freqs(self, train_sents):
		"""
		Generates word frequencies from the training sentences for the feature
		extractor.
		@type train_sents: C{list} of C{list} of tuples of (C{str}, C{str})
		@param train_sents: A list of tagged sentences.
		@rtype: C{FreqDist}
		@return: a L{frequency distribution<nltk.FreqDist()>},
		counting how often each word occurs in the training sentences.
		"""
		word_freqdist = FreqDist()
		for tagged_sent in tqdm(train_sents):
			for (word, _tag) in tagged_sent:
				word_freqdist[word] += 1
		return word_freqdist

	def gen_featsets(self, train_sents, rare_word_cutoff):
		"""
		Generates featuresets for each token in the training sentences.
		@type train_sents: C{list} of C{list} of tuples of (C{str}, C{str})
		@param train_sents: A list of tagged sentences.
		@type rare_word_cutoff: C{int}
		@param rare_word_cutoff: Words with less occurrences than
		C{rare_word_cutoff} will be treated differently by L{extract_feats}
		than non-rare words (cf. Ratnaparkhi 1996).
		@rtype: {list} of C{tuples} of (C{dict}, C{str})
		@return:  a list of tuples that contains the featureset of
		a token and its POS-tag.
		"""
		featuresets = []
		for tagged_sent in train_sents:
			history = []
			untagged_sent = untag(tagged_sent)
			for (i, (_word, tag)) in enumerate(tagged_sent):
				featuresets.append( (self.extract_feats(untagged_sent, i,
					history, rare_word_cutoff), tag) )
				history.append(tag)
		return featuresets


	def cutoff_rare_feats(self, featuresets, rare_feat_cutoff):
		"""
		Cuts off rare features to reduce training time and prevent overfitting.
		Example
		=======
			Let's say, the suffixes of this featureset are too rare to learn.
			>>> featuresets[46712]
			({'suffix(1)': 't',
			'prefix(1)': 'L',
			'prefix(2)': 'Le',
			'prefix(3)': 'Lem',
			'suffix(3)': 'ont',
			'suffix(2)': 'nt',
			'contains-uppercase': True,
			'prefix(4)': 'Lemo',
			'suffix(4)': 'mont'},
			'NNP')
			C{cutoff_rare_feats} would then remove the rare joint features::
				(('suffix(1)', 't'), 'NNP')
				(('suffix(3)', 'ont'), 'NNP')
				((suffix(2)': 'nt'), 'NNP')
				(('suffix(4)', 'mont'), 'NNP')
			and return a featureset that only contains non-rare features:
			>>> featuresets[46712]
			({'prefix(1)': 'L',
			'prefix(2)': 'Le',
			'prefix(3)': 'Lem',
			'contains-uppercase': True,
			'prefix(4)': 'Lemo'},
			'NNP')
		@type featuresets: {list} of C{tuples} of (C{dict}, C{str})
		@param featuresets: a list of tuples that contain the featureset of a
		word from the training set and its POS tag
		@type rare_feat_cutoff: C{int}
		@param rare_feat_cutoff: if a (context information feature, tag)
		tuple occurs less than C{rare_feat_cutoff} times in the training
		set, then its corresponding feature will be removed from the
		C{featuresets} to be learned.
		"""
		never_cutoff_features = set(['w','t'])

		for (feat_dict, tag) in featuresets:
			for (feature, value) in list(feat_dict.items()):
				feat_value_tag = ((feature, value), tag)
				if self.features_freqdist[feat_value_tag] < rare_feat_cutoff:
					if feature not in never_cutoff_features:
						feat_dict.pop(feature)


	def extract_feats(self, sentence, i, history, rare_word_cutoff=5):
		"""
		Generates a featureset from a word (in a sentence). The features
		were chosen as described in Ratnaparkhi (1996) and his Java
		software package U{MXPOST<ftp://ftp.cis.upenn.edu/pub/adwait/jmx>}.
		The following features are extracted:
			- features for all words: last tag (C{t-1}), last two tags (C{t-2
				t-1}), last words (C{w-1}) and (C{w-2}), next words (C{w+1}) and
				(C{w+2})
			- features for non-rare words: current word (C{w})
			- features for rare words: word suffixes (last 1-4 letters),
				word prefixes (first 1-4 letters),
				word contains number (C{bool}), word contains uppercase character
				(C{bool}), word contains hyphen (C{bool})
		Ratnaparkhi experimented with his tagger on the Wall Street Journal
		corpus (Penn Treebank project). He found that the tagger yields
		better results when words which occur less than 5 times are treated
		as rare. As your mileage may vary, please adjust
		L{rare_word_cutoff} accordingly.
		Examples
		========
			1. This is a featureset extracted from the nonrare (word, tag)
			tuple ('considerably', 'RB')
			>>> featuresets[22356]
			({'t-1': 'VB',
			't-2 t-1': 'TO VB',
			'w': 'considerably',
			'w+1': '.',
			'w+2': '<END>',
			'w-1': 'improve',
			'w-2': 'to'},
			'RB')
			2. A featureset extracted from the rare tuple ('Lemont', 'NN')
			>>> featuresets[46712]
			({'suffix(1)': 't',
			'prefix(1)': 'L',
			'prefix(2)': 'Le',
			'prefix(3)': 'Lem',
			'suffix(3)': 'ont',
			'suffix(2)': 'nt',
			'contains-uppercase': True,
			'prefix(4)': 'Lemo',
			'suffix(4)': 'mont'},
			'NNP')
		@type sentence: C{list} of C{str}
		@param sentence: A list of words, usually a sentence.
		@type i: C{int}
		@param i: The index of a word in a sentence, where C{sentence[0]} would
		represent the first word of a sentence.
		@type history: C{int} of C{str}
		@param history: A list of POS-tags that have been assigned to the
		preceding words in a sentence.
		@type rare_word_cutoff: C{int}
		@param rare_word_cutoff: Words with less occurrences than
		C{rare_word_cutoff} will be treated differently than non-rare words
		(cf. Ratnaparkhi 1996).
		@rtype: C{dict}
		@return: a dictionary of features extracted from a word's
		context.
		"""
		features = {}
		hyphen = re.compile("-")
		number = re.compile("\d")
		uppercase = re.compile(self.uppercase_letters)

		#get features: w-1, w-2, t-1, t-2.
		#takes care of the beginning of a sentence
		if i == 0: #first word of sentence
			features.update({"w-1": "<START>", "t-1": "<START>",
								"w-2": "<START>", "t-2 t-1": "<START> <START>"})
		elif i == 1: #second word of sentence
			features.update({"w-1": sentence[i-1], "t-1": history[i-1],
								"w-2": "<START>",
								"t-2 t-1": "<START> %s" % (history[i-1])})
		else:
			features.update({"w-1": sentence[i-1], "t-1": history[i-1],
				"w-2": sentence[i-2],
				"t-2 t-1": "%s %s" % (history[i-2], history[i-1])})

		#get features: w+1, w+2. takes care of the end of a sentence.
		for inc in [1, 2]:
			try:
				features["w+%i" % (inc)] = sentence[i+inc]
			except IndexError:
				features["w+%i" % (inc)] = "<END>"

		if self.word_freqdist[sentence[i]] >= rare_word_cutoff:
			#additional features for 'non-rare' words
			features["w"] = sentence[i]

		else: #additional features for 'rare' or 'unseen' words
			features.update({"suffix(1)": sentence[i][-1:],
				"suffix(2)": sentence[i][-2:], "suffix(3)": sentence[i][-3:],
				"suffix(4)": sentence[i][-4:], "prefix(1)": sentence[i][:1],
				"prefix(2)": sentence[i][:2], "prefix(3)": sentence[i][:3],
				"prefix(4)": sentence[i][:4]})
			if hyphen.search(sentence[i]) != None:
				#set True, if regex is found at least once
				features["contains-hyphen"] = True
			if number.search(sentence[i]) != None:
				features["contains-number"] = True
			if uppercase.search(sentence[i]) != None:
				features["contains-uppercase"] = True

		return features

	sentence = []
	pos_tags = []
	history = []
	rare_word_cutoff = 5

	
if __name__ == '__main__':
	main()

# for i in range(len(sentence)):
# 	featureset = extract_feats(sentence, i, history,
# 										rare_word_cutoff)
# 	tag = classifier.classify(featureset)
# 	history.append(pos_tags[i])
# 	