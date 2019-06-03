using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace ObjectDetection_Common
{
    public class ImageNetPrediction
    {
        [ColumnName(TinyYoloModelSettings.ModelOutput)]
        public float[] PredictedLabels;

        public void PrintToConsole()
        {
            Console.WriteLine($"PredictedLabels(length:{PredictedLabels.Length}):");
            foreach (var f in PredictedLabels)
            {
                Console.Write($"{f},");
            }
            Console.WriteLine();
        }
    }
}
