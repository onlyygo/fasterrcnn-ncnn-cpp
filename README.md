# faster-rcnn-base-on-NCNN

因为caffe的依赖库太多了，win下使用起来很繁琐，移动端部署更加麻烦。故将其caffe替换为ncnn库。
ncnn：https://github.com/Tencent/ncnn

linux：
下载后将src/CMakeLists.txt的82行到121行的“OFF”删掉，让所有的层都编译到库文件里面。

windows：
将ncnn的源代码都添加进项目里面，并添加上本项目win下的三个文件。编译即可。
PS：使用自己的模型，注意修改Config.h文件中的anchors信息等。生成新的anchors的脚本见于我的项目rfcn-caffe-cpp。

有问题欢迎邮件咨询：onlyygo@qq.com

模型可以在这里下载：
https://drive.google.com/open?id=0B9OXtJAN3-R8UDZIZVRzTU4yU1E

1, delete the “OFF” in src/CMakeLists.txt from line 82 to line 121, to make all layer will be compiled in lib.

2, If you want to train model youself. You should modify the anchors setting in Config.h. The anchors can be genrate in generateanchors.py at "https://github.com/onlyygo/rfcn-caffe-cpp"

3, If you have some question, please email 'onlyygo@qq.com'.

4, model can download here

https://drive.google.com/open?id=0B9OXtJAN3-R8UDZIZVRzTU4yU1E
