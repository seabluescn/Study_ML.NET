using System;

namespace BinaryClassification_TextFeaturize
{
    public class JiebaLambdaInput
    {
        public string Text { get; set; }
    }

    public class JiebaLambdaOutput
    {
        public string JiebaText { get; set; }
    }

    public class JiebaLambda
    {       
        public static void MyAction(JiebaLambdaInput input, JiebaLambdaOutput output)
        {
            JiebaNet.Segmenter.JiebaSegmenter jiebaSegmenter = new JiebaNet.Segmenter.JiebaSegmenter();
            output.JiebaText = string.Join(" ", jiebaSegmenter.Cut(input.Text));

            Count++;
            //Console.WriteLine($"JiebaLambda.MyAction Debug:{Count}");
        }

        static int Count = 0;
    }
}
