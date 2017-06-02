import jieba

filePath='/home/dingji/Desktop/Cut.txt'
fileSegWordDonePath ='/home/dingji/Desktop/new.txt'    #segmented file
# read the file by line
fileTrainRead = []
#fileTestRead = []
with open(filePath) as fileTrainRaw:
    for line in fileTrainRaw:
       fileTrainRead.append(line)

def PrintListChinese(list):
    for i in range(len(list)):
         print (list[i])
 # segment word with jieba
fileTrainSeg=[]
for i in range(len(fileTrainRead)):
    fileTrainSeg.append([' '.join(list(jieba.cut(fileTrainRead[i][9:-11],cut_all=False)))])
    if i % 100 == 0 :
         print (i)
 # to test the segment result
 #PrintListChinese(fileTrainSeg[10])

 # save the result
with open(fileSegWordDonePath,'wb') as fW:
    for i in range(len(fileTrainSeg)):
         fW.write(fileTrainSeg[i][0].encode('utf-8'))
#         fW.write('\n')