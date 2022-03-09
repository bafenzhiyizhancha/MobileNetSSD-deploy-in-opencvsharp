using System;
using System.Windows;

using OpenCVCSharpDNN;

namespace yolovdetection
{
    /// <summary>
    /// MainWindow.xaml 的交互逻辑
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            Init();
        }

        private void Init()
        {

            string dir                  = System.IO.Directory.GetCurrentDirectory();
            string weightpath           = dir + "/weights/";
            string imgagepath           = dir + "/image/";
            string modelWeights         = System.IO.Path.Combine(weightpath, "mobilenet_iter_73000.caffemodel");
            string modelConfiguaration  = System.IO.Path.Combine(weightpath, "deploy.prototxt");
            string labelFile            = System.IO.Path.Combine(weightpath, "coconames.txt");
            string imgpath              = System.IO.Path.Combine(imgagepath, "img.jpeg");

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


        private void Init2()
        {

            string dir = System.IO.Directory.GetCurrentDirectory();
            string weightpath = dir + "/weights/";
            string imgagepath = dir + "/image/";
            string modelWeights = System.IO.Path.Combine(weightpath, "MobileNetSSD_deploy.caffemodel");
            string modelConfiguaration = System.IO.Path.Combine(weightpath, "MobileNetSSD_deploy.prototxt");
            string labelFile = System.IO.Path.Combine(weightpath, "coconames.txt");
            string imgpath = System.IO.Path.Combine(imgagepath, "img.jpeg");

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
    }
}
