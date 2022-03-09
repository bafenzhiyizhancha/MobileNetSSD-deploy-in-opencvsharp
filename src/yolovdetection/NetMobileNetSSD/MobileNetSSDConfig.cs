using OpenCvSharp;
using System.Linq;

namespace OpenCVCSharpDNN
{
    /// <summary>
    /// 参数
    /// </summary>
    public class MobileNetSSDConfig
    {
        /// <summary>
        /// 模型权重文件路径
        /// </summary>
        public string ModelWeights { set; get; }


        /// <summary>
        /// 模型权重参数文件路径
        /// </summary>
        public string ModelConfiguaration { set; get; }

        /// <summary>
        /// 标签文件路径
        /// </summary>
        public string LabelsFile { set; get; }

        /// <summary>
        /// 标签
        /// </summary>
        public string[] Labels { set; get; }

        /// <summary>
        /// 置信度阈值
        /// </summary>
        public float Threshold { set; get; }

        /// <summary>
        /// 模型要求的的图片大小
        /// </summary>
        public int ImgWidth { set; get; }

        /// <summary>
        ///模型要求的的图片大小
        /// </summary>
        public int ImgHight { set; get; }

        /// <summary>
        /// 是否显示图像
        /// </summary>
        public bool IsDraw { set; get; }

        /// <summary>
        /// 画图的颜色
        /// </summary>
        public Scalar[] Colors;

        /// <summary>
        /// 初始化
        /// </summary>
        public MobileNetSSDConfig()
        {

            IsDraw = true;
            Colors = Enumerable.Repeat(false, 20).Select(x => Scalar.RandomColor()).ToArray();

            
        }
    }
}
