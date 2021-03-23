import scipy.io
data = scipy.io.loadmat('C:/Users/as722/Downloads/setid.mat') 
# 假設檔名為1.mat4 
# data型別為dictionary5 
print (data.keys() )
# 即可知道Mat檔案中存在資料名,假設存在'x', 'y'兩列資料6 
# print(data['__header__'])
# print(data['__version__'])
# print(data['__globals__'])
# print(data['labels'])
label = data['trnid']
