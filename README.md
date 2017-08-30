# faster-rcnn-base-on-NCNN

因为caffe的依赖库太多了，win下使用起来很繁琐，移动端部署更加麻烦。故将其caffe替换为ncnn库。
ncnn：https://github.com/Tencent/ncnn
linux：
下载后将src/CMakeLists.txt的82行到121行的“OFF”删掉，让所有的层都编译到库文件里面。
windows：
将ncnn的源代码都添加进项目里面，并添加上本项目win下的三个文件。编译即可。
有问题欢迎邮件咨询：onlyygo@qq.com
