import urllib
import requests
from lxml import etree
plist=[]
def get_song(lists):
	id=1
	for x in lists:
		urllib.urlretrieve(x,str(id)+'.mid')
		id=id+1
def get_list(url):
	htm=requests.get(url)
	res=etree.HTML(htm.content)
	p=res.xpath('//*[@id="page"]/ul[1]/li/a/@href')
	return p

for x in range(8):
	x=x+1
	plist=plist+get_list('http://www.midiworld.com/search/'+str(x)+'/?q=dance')

get_song(plist)


