using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;

namespace BinaryClassification_TextFeaturize
{
    class Program
    {
        static readonly string DataPath = Path.Combine(Environment.CurrentDirectory, "Data", "meeting_data_full.csv");

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();
            var fulldata = mlContext.Data.LoadFromTextFile<MeetingInfo>(DataPath, separatorChar: ',', hasHeader: false);
            var trainTestData = mlContext.Data.TrainTestSplit(fulldata, testFraction: 0.15);
            var trainData = trainTestData.TrainSet;
            var testData = trainTestData.TestSet;

            var trainingPipeline = mlContext.Transforms.CustomMapping<JiebaLambdaInput, JiebaLambdaOutput>(mapAction: JiebaLambda.MyAction, contractName: "JiebaLambda")
                .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: "JiebaText"))
                .Append(mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features"));
            ITransformer trainedModel = trainingPipeline.Fit(trainData);

            //评估
            var predictions = trainedModel.Transform(testData);
            //DebugData(mlContext, predictions);
            var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label");
            Console.WriteLine($"Evalution Accuracy: {metrics.Accuracy:P2}");


            //创建预测引擎
            var predEngine = mlContext.Model.CreatePredictionEngine<MeetingInfo, PredictionResult>(trainedModel);

            MeetingInfo sampleStatement1 = new MeetingInfo { Text = "支委会。" };
            var predictionresult1 = predEngine.Predict(sampleStatement1);
            Console.WriteLine($"{sampleStatement1.Text}:{predictionresult1.PredictedLabel}");
            predictionresult1.PrintToConsole();

            MeetingInfo sampleStatement2 = new MeetingInfo { Text = "开展新时代中国特色社会主义思想三十讲党员答题活动。" };
            var predictionresult2 = predEngine.Predict(sampleStatement2);
            Console.WriteLine($"{sampleStatement2.Text}:{predictionresult2.PredictedLabel}");
            predictionresult2.PrintToConsole();

            Console.WriteLine("Press any to exit!");
            Console.ReadKey();
        }

        private static void DebugData(MLContext mlContext, IDataView predictions)
        {
            var trainDataShow = new List<PredictionResult>(mlContext.Data.CreateEnumerable<PredictionResult>(predictions, false, true));

            foreach (var dataline in trainDataShow)
            {
                dataline.PrintToConsole();
            }
        }
    }

    public class MeetingInfo
    {
        [LoadColumn(0)]
        public bool Label { get; set; }
        [LoadColumn(1)]
        public string Text { get; set; }
    }

    public class PredictionResult : MeetingInfo
    {
        public string JiebaText { get; set; }
        public float[] Features { get; set; }
        public bool PredictedLabel;
        public float Score;
        public float Probability;
        public void PrintToConsole()
        {
            Console.WriteLine($"JiebaText={JiebaText}");
            Console.WriteLine($"PredictedLabel:{PredictedLabel},Score:{Score},Probability:{Probability}");
            Console.WriteLine($"TextFeatures Length:{Features.Length}");
            if (Features != null)
            {
                foreach (var f in Features)
                {
                    Console.Write($"{f},");
                }
                Console.WriteLine();
            }
            Console.WriteLine();
        }
    }
}
