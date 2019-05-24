using System;
using System.Collections.Generic;
using System.Text;

namespace MulticlassClassification_Mnist
{
    public class DebugLambdaInput
    {
        public float Count { get; set; }
    }

    public class DebugLambdaOutput
    {
        public string DebugText { get; set; }
    }

    public class DebugLambda
    {
        public static void MyAction(DebugLambdaInput input, DebugLambdaOutput output)
        {
            output.DebugText = "";
            Console.WriteLine($"DebugLambda.MyAction Debug:{input.Count}");
        }
       
    }

}
