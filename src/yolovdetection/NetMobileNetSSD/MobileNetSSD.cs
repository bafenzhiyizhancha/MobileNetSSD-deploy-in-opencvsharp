using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace OpenCVCSharpDNN
{
    public class MobileNetSSD
    {
        private readonly MobileNetSSDConfig _config;

        /// <summary>
        /// 初始化
        /// </summary>
        /// <param name="pathModel"> 权重路径</param>
        /// <param name="pathConfig">权重参数路径</param>
        /// <param name="labelsFile">标签路径</param>
        /// <param name="imgWidth">模型要求的的图片大小</param>
        /// <param name="imgHigh">模型要求的的图片大小</param>
        /// <param name="threshold">置信度阈值</param>
        public MobileNetSSD(string pathModel, string pathConfig, string labelsFile,
                       int imgWidth = 300, int imgHigh = 300, float threshold = 0.5f)
        {
            _config = new MobileNetSSDConfig
            {
                ModelWeights        = pathModel,
                ModelConfiguaration = pathConfig,
                LabelsFile          = labelsFile,
                ImgWidth            = imgWidth,
                ImgHight            = imgHigh,
                Threshold           = threshold,
            };
        }

        /// <summary>
        /// 检测
        /// </summary>
        /// <param name="imgpath">图像路径</param>
        /// <returns></returns>
        public NetResult[] Detect(string imgpath)
        {
            Mat img = Cv2.ImRead(imgpath);
            return Process(img);
        }

        /// <summary>
        /// 检测
        /// </summary>
        /// <param name="img"></param>
        /// <returns></returns>
        public NetResult[] Detect(System.Drawing.Bitmap img)
        {
            Mat mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(img);
            return Process(mat);
        }

        #region way
        /// <summary>
        /// 图片预处理
        /// </summary>
        /// <param Mat img</param>
        /// <returns></returns>
        private Mat ImagePretreatment(Mat img)
        {
            double scale = 1.0 / 255;                                     //像素大小由0~255范围变为0~1，深度学习中的输入都是0~1范围的，便于优化，暂时每一其他数值大小的    
            Size size   = new Size(_config.ImgWidth, _config.ImgHight);  //模型网络结果是使用同大小一图像进行训练的所以需要将图像转为对应的大小

            //将二维图像转换为CNN输入的张量Tensor,作为网络的输入
            Mat blob = CvDnn.BlobFromImage(img, scale, size, new Scalar(), true, false);

            return blob;
        }

        /// <summary>
        /// 读入标签
        /// </summary>
        /// <param name="pathLabels"></param>
        /// <returns></returns>
        private string[] ReadLabels(string pathLabels)
        {
            if (!File.Exists(pathLabels))
                throw new FileNotFoundException("The file of labels not foud", pathLabels);

            string[] classNames = File.ReadAllLines(pathLabels).ToArray();

            return classNames;
        }

        /// <summary>
        /// 初始化模型
        /// </summary>
        private Net InitializeModel(string pathModel, string pathConfig)
        {
            if (!File.Exists(pathModel))
                throw new FileNotFoundException("The file model has not found", pathModel);
            if (!File.Exists(pathConfig))
                throw new FileNotFoundException("The file config has not found", pathConfig);

            Net net = CvDnn.ReadNetFromCaffe(pathConfig, pathModel);
            if (net == null || net.Empty())
                throw new Exception("The model has not yet initialized or is empty.");

            //读入模型和设置
            net.SetPreferableBackend(Backend.OPENCV);     // 3:DNN_BACKEND_OPENCV 
            net.SetPreferableTarget(Target.CPU);          //dnn target cpu

            return net;
        }

        /// <summary>
        /// 检测的后处理
        /// </summary>
        /// <param name="image"></param>
        /// <param name="results"></param>
        /// <returns></returns>
        private NetResult[] Postprocess(ref Mat image, Mat result)
        {
            var netResults = new List<NetResult>();         //存储检测结果

            var w = image.Width;                            //图像的宽高
            var h = image.Height;

            /* 
            MobileNetSSD 模型输出output的格式：矩阵形式
            呈[1, 1, 100, 7]形式
            其中一维，二维的1是在这里没有意义，第三为中100代表共检测出100个可能的对象，会随图像变化数值
            第四维7含义：此维度不会发生改变，必然存在7个数值：[0,x1,x2，x3,x4,x5,x6],代表
            着可能对象的信息。
            列的各个维的信息为：
            0  : 无意义                   x1 : 标签序号 
            x2 : confidence  对象框的置信度               
            x3 : 对象框的左
            x4 : 对象框的顶
            x5 : 对象框的右
            x6 : 对象框的底
           */

            //从[1, 1, 100, 7]中取出第三、第四维：[100，7]
            Mat detectionMat = new Mat(result.Size(2), result.Size(3), MatType.CV_32F, result.Ptr());

            for (int i = 0; i < detectionMat.Rows; i++)                         // detectionMat.Rows=[100,7]中的100
            {
                float confidence = detectionMat.At<float>(i, 2);                //置信度
                if (confidence > _config.Threshold)
                {
                    var classID = (int)(detectionMat.At<float>(i, 1));          //对应的标签序号    

                    //x,y,width,height 都是相对于输入图片的比例，所以需要乘以相应的宽高进行复原
                    var left    = detectionMat.At<float>(i, 3) * w;             //方框左边
                    var top     = detectionMat.At<float>(i, 4) * h;
                    var right   = detectionMat.At<float>(i, 5) * w;
                    var bottom  = detectionMat.At<float>(i, 6) * h;


                    var label   = _config.Labels[classID];
                    var width   = right - left;
                    var height  = bottom - top;

                    //存储最终结果
                    netResults.Add(NetResult.Add((int)left, (int)top, (int)width, (int)height, label, confidence));

                    if (true == _config.IsDraw)
                    {
                        Draw(ref image, classID, confidence, left, top, width, height);
                    }
                }
            }

            return netResults.ToArray();                // 返回检测结果
        }

        /// <summary>
        /// 将结果在图像上画出
        /// </summary>
        /// <param name="image"></param>
        /// <param name="classes"> 在标签中的序号</param>
        /// <param name="confidence">置信度</param>
        /// <param name="left">对象框左边距离</param>
        /// <param name="top">对象框顶边距离</param>
        /// <param name="width">对象框宽度</param>
        /// <param name="height">对象框高度</param>
        private void Draw(ref Mat image, int classes, float confidence, double left, double top, double width, double height)
        {
            //标签字符串
            var label = string.Format("{0} {1:0.0}%", _config.Labels[classes], confidence * 100);
            //画方框
            Cv2.Rectangle(image, new Point(left, top), new Point(left + width, top + height), _config.Colors[classes], 1);

            //标签字符大小
            var textSize = Cv2.GetTextSize(label, HersheyFonts.HersheyTriplex, 0.5, 1, out var baseline);

            //画标签背景框
            var x1 = left < 0 ? 0 : left;
            Cv2.Rectangle(image, new Rect(new Point(x1, top - textSize.Height - baseline),
                new Size(textSize.Width, textSize.Height + baseline)), _config.Colors[classes], Cv2.FILLED);
            Cv2.PutText(image, label, new Point(x1, top - baseline), HersheyFonts.HersheyTriplex, 0.5, Scalar.White);

            Cv2.ImShow("图片展示：", image);
        }

        /// <summary>
        /// 整个处理过程
        /// </summary>
        /// <param name="img"></param>
        /// <returns></returns>
        private NetResult[] Process(Mat img)
        {
            Mat blob = ImagePretreatment(img);

            Net net = InitializeModel(_config.ModelWeights, _config.ModelConfiguaration);
            _config.Labels = ReadLabels(_config.LabelsFile);


            net.SetInput(blob);                                  //输入网络

            var outs = net.Forward();                          //CNN网络前向计算-

            NetResult[] netResults = Postprocess(ref img, outs);


            return netResults;
        }
        #endregion


        #region demo
        private void Demo()
        {
            string dir                  = System.IO.Directory.GetCurrentDirectory();
            string modelWeights         = System.IO.Path.Combine(dir, "MobileNetSSD_deploy.caffemodel");
            string modelConfiguaration  = System.IO.Path.Combine(dir, "MobileNetSSD_deploy.prototxt");
            string labelFile            = System.IO.Path.Combine(dir, "coconames2.txt");
            string imgpath              = System.IO.Path.Combine(dir, "img.jpeg");
            MobileNetSSD netMobile   = new MobileNetSSD(modelWeights, modelConfiguaration, labelFile, 300, 300, 0.2f);
            NetResult[] netResults      = netMobile.Detect(imgpath);
            foreach (var result in netResults)
            {
                Console.WriteLine("类别，置信度，方框坐标[左边、顶点、长、宽]");
                Console.WriteLine(result.Label + "  " + result.Probability.ToString() + " [" +
                    result.Rectangle.Left.ToString() + " " + result.Rectangle.Top.ToString() + " " +
                    result.Rectangle.Width.ToString() + " " + result.Rectangle.Height.ToString() + "]");
            }
        }

        private void Demo2()
        {
            string dir                  = System.IO.Directory.GetCurrentDirectory();
            string modelWeights         = System.IO.Path.Combine(dir, "mobilenet_iter_73000.caffemodel");
            string modelConfiguaration  = System.IO.Path.Combine(dir, "deploy.prototxt");
            string labelFile            = System.IO.Path.Combine(dir, "coconames2.txt");
            string imgpath              = System.IO.Path.Combine(dir, "img.jpeg");
            MobileNetSSD netMobile = new MobileNetSSD(modelWeights, modelConfiguaration, labelFile, 300, 300, 0.2f);
            NetResult[] netResults = netMobile.Detect(imgpath);
            foreach (var result in netResults)
            {
                Console.WriteLine("类别，置信度，方框坐标[左边、顶点、长、宽]");
                Console.WriteLine(result.Label + "  " + result.Probability.ToString() + " [" +
                    result.Rectangle.Left.ToString() + " " + result.Rectangle.Top.ToString() + " " +
                    result.Rectangle.Width.ToString() + " " + result.Rectangle.Height.ToString() + "]");
            }
        }
        #endregion
    }
}
