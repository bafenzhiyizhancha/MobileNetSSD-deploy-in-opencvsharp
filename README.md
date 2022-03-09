
## MobileNetSSD模型在c#上的调用模块，使用opencvsharp进行模型加载推理
![image](https://user-images.githubusercontent.com/26215301/157416176-96522d92-0e1d-47f4-9baf-e2512830c699.png)

### 注意事项：  
1、src是在anycpu debug下编译的，如需切换到x64下编译，需要复制原debug下image于weights文件夹。  
2、运行环境nuget包：OpenCvSharp与OpenCvSharp4.runtime.win   


#### 其他：  
1、模型属于caffe训练，可到github：chuanqi305/MobileNet-SSD，使用其代码训练自己的数据。  
2、若想要使用tensorflow、pytorch训练的模型，可以改变加载函数：Net net = CvDnn.ReadNetFromCaffe(pathConfig, pathModel);   
可自行搜索opencv.dnn模块中关于加载不同模型的函数。
