using System;
using System.Collections.Generic;
using System.Text;

namespace ObjectDetection_Common
{
    public struct TinyYoloModelSettings
    {
        // for checking TIny yolo2 Model input and  output  parameter names,
        //you can use tools like Netron, 
        // which is installed by Visual Studio AI Tools

        // input tensor name
        public const string ModelInput = "image";

        // output tensor name
        public const string ModelOutput = "grid";
    }
}
