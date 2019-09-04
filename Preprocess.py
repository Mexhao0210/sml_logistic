import os
import re
import tldextract
import math
import nltk
import csv
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
# import enchant
from textblob import TextBlob


class Preprocess:
	def __init__(self):
		self.raw_data_path = 'test_tweets_unlabeled.txt'
		self.emoji_path = './icon.txt'
		self.topic_url_path = './topic_url.txt'
		self.keyword_path = './keyword.txt'
		self.output_path = 'processed_test.txt'
		self.feature_list = ['id', 'retweet', 'length', 'uppercase', 'typo', 'sentiment', '!','?', '$']
		self.content_list = []
		self.sentence_feature = dict()
		self.isTest=True
		# nltk.download('punkt')
		# nltk.download('stopwords')
		self.r4 = "\\【.*?】…»“”–#‹+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}]+|[——！，。=？、:""''￥……（）《》【】]…»“”–#‹"
		self.twitter_username_re = re.compile(r'@([A-Za-z0-9_]+)')
		self.twitter_tag_re = re.compile(r'#([A-Za-z0-9_]+)')
		self.dot=['[',']','“','”','|','-','`','\'','\"','‹']
		self.stop_words = set(stopwords.words('english'))
		self.punctuation=['!','?','$']
		self.positive={}
		self.negative={}
		self.neutral={}
		self.emoji=[]
		self.topic_url = []
		self.keyword = []
		self.read_raw_data()
		self.read_emoji()
		self.read_topic_url()
		self.read_keywords()

	def read_raw_data(self):
		raw_data_file = open(self.raw_data_path,encoding='utf-8')
		for line in raw_data_file.readlines():
			line = line.strip()
			self.content_list.append(line)
		raw_data_file.close()

	def read_emoji(self):
		with open(self.emoji_path,'r',errors='ignore') as emo:
			for line in emo:
				line = line.strip()
				self.feature_list.append(line)
				self.emoji.append(line)
		emo.close()

	def read_topic_url(self):
		with open(self.topic_url_path,'r',errors='ignore') as tu:
			for line in tu:
				line = line.strip()
				self.topic_url.append(line)
				self.feature_list.append(line)
		tu.close()

	def read_keywords(self):
		with open(self.keyword_path,'r',errors='ignore') as kw:
			for line in kw:
				line = line.strip()
				self.keyword.append(line)
				self.feature_list.append(line)
		kw.close()

	def process_RT(self):
		#remove empty sentence
		temp_content_list = list()
		for i in range(len(self.content_list)):
			line = self.content_list[i]
			item = dict()
			if self.isTest:
				user_id=''
				content=line
			else: user_id, content = line.split('\t', 1)
			item['id'] = user_id
			item['retweet'] = 0
			if 'RT @handle' in content:
				item['retweet'] = 1
				content = ((content.split('RT @handle'))[0])
			#replace @handle
			content = content.replace('@handle', '').strip()
			item['length'] = len(content.split())
			if content == "":
				continue
			else:
				self.sentence_feature[len(temp_content_list)] = item
				temp_content_list.append(user_id + '\t' + content)
		self.content_list = temp_content_list

	def url_to_domain_topic(self):
		for i in range(len(self.content_list)):
			line = self.content_list[i]
			item = self.sentence_feature[i]
			if self.isTest:
				user_id = ''
				content = line
			else:
				user_id, content = line.split('\t', 1)
			#grep https?://
			url_list = re.findall(r'(https?://\S+)', content)
			if(len(url_list) != 0):
				for url in url_list:
					domain = (tldextract.extract(url).domain)
					if domain == "":
						content = content.replace(url, '')
						continue
					else:
						domain = "@" + domain
						content = content.replace(url, domain)
					if domain in self.topic_url:
						if domain not in item.keys():
							item[domain] = 1

		    #grep www.
			url_list = re.findall(r'(www\.\S+)', content)
			if(len(url_list) != 0):
				for url in url_list:
					utl = url.strip()
					domain = (tldextract.extract(url).domain)
					if domain == "":
						content = content.replace(url, '')
						continue
					else:
						omain = "@" + domain
						content = content.replace(url, domain)
					if domain in self.topic_url:
						if domain not in item.keys():
							item[domain] = 1

		    # grep topic
			topic_list = re.findall(r"#\w+", content)
			if(len(topic_list) != 0):
				for topic in topic_list:
					if topic in self.topic_url:
						if topic not in item.keys():
							item[topic] = 1

			self.content_list[i] = user_id + '\t' + content
			self.sentence_feature[i] = item

	def get_emoji_punctuation(self):
		for i in range(len(self.content_list)):
			line = self.content_list[i]
			item = self.sentence_feature[i]
			if self.isTest:
				user_id = ''
				sentence = line
			else:
				user_id, sentence = line.split('\t', 1)
			for emoji in self.emoji:
				if sentence.find(emoji)>0: 
					item[emoji] = 1
			for p in self.punctuation:
				c = sentence.count(p)
				item[p] = c
			self.sentence_feature[i] = item

	def get_typo_feature(self):
		# d = enchant.Dict("en_US")
		for i in range(len(self.content_list)):
			line = self.content_list[i]
			item = self.sentence_feature[i]
			if self.isTest:
				user_id = ''
				sentence = line
			else:
				user_id, sentence = line.split('\t', 1)
			# sentence=line
			sentence = re.sub(self.twitter_username_re, '', sentence)
			sentence = re.sub(self.twitter_tag_re, '', sentence)
			sentence = re.sub(self.r4, ' ', sentence)
			for x in self.dot:
				sentence=sentence.replace(x,'')
			inccorect_word_num = 0
			sentence_length = len(sentence.split(" "))
			# for word in sentence.split(" "):
			# 	if word.isalpha() == False:
			# 		continue
			# 	if d.check(word) == False:
			# 		inccorect_word_num += 1
			item['typo'] = inccorect_word_num
			self.content_list[i] = user_id + '\t' + sentence
			self.sentence_feature[i] = item

	def get_upper_and_keyword(self):
		ps = PorterStemmer()
		for i in range(len(self.content_list)):
			line = self.content_list[i]
			item = self.sentence_feature[i]
			if self.isTest:
				user_id = ''
				sentence = line
			else:
				user_id, sentence = line.split('\t', 1)
			word_tokens = word_tokenize(sentence)
			upper=0
			for w in word_tokens:
				if re.search('^[A-Z]+$', w): 
					upper+=1
			item['uppercase'] = upper
			sentence = sentence.lower()
			word_tokens = word_tokenize(sentence)
			tokens = [ps.stem(w) for w in word_tokens if not w in self.stop_words]
			for kw in tokens:
				if kw in self.keyword:
					item[kw] = 1
			sentence = " ".join(tokens)
			self.content_list[i] = user_id + '\t' + sentence
			self.sentence_feature[i] = item

	def get_sentiment(self):
		ps = PorterStemmer()
		for i in range(len(self.content_list)):
			line = self.content_list[i]
			item = self.sentence_feature[i]
			if self.isTest:
				user_id = ''
				sentence = line
			else:
				user_id, sentence = line.split('\t', 1)
			tb = TextBlob(line)
			polarity = round(tb.sentiment.polarity, 4)
			item['sentiment'] = polarity
			self.sentence_feature[i] = item

	def save_feature(self):
		output_file = open(self.output_path, 'w')
		output_file.write(",".join(self.feature_list) + '\n')
		for i in range(len(self.sentence_feature)):
			item = self.sentence_feature[i]
			cur_line = str(item['id'])
			for j in range(1, len(self.feature_list)):
				fea = self.feature_list[j]
				if fea in item.keys():
					cur_line = cur_line + ',' + fea + ":" + str(item[fea])
				else:
					cur_line = cur_line + ',' + '0'					
			output_file.write(cur_line + '\n')

	def save_for_logistic(self):
		output_file = open(self.output_path, 'w',encoding='utf-8')
		for i in self.content_list:
			if self.isTest: i=i.strip()
			output_file.write(i+'\n')
		output_file.close()

	def process(self):
		self.process_RT()
		self.url_to_domain_topic()
		self.get_emoji_punctuation()
		self.get_typo_feature()
		# self.get_upper_and_keyword()
		# self.get_sentiment()
		# self.save_feature()
		self.save_for_logistic()


if __name__ == '__main__':
	preprocess = Preprocess()
	preprocess.process()