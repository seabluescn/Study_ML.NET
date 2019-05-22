using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;

namespace BinaryClassification_Figure
{
    class Program
    {
        static readonly string DataPath = Path.Combine(Environment.CurrentDirectory, "Data", "figure_full.csv");
        static readonly string ModelPath = Path.Combine(Environment.CurrentDirectory, "Data", "FastTree_Model.zip");

        static void Main(string[] args)
        {
            //TrainAndSave();
            LoadAndPrediction();

            Console.WriteLine("Press any to exit!");
            Console.ReadKey();
        }

        static void TrainAndSave()
        {
            MLContext mlContext = new MLContext();

            //准备数据
            var fulldata = mlContext.Data.LoadFromTextFile<FigureData>(path: DataPath, hasHeader: true, separatorChar: ',');           
            var trainTestData = mlContext.Data.TrainTestSplit(fulldata,testFraction:0.2);
            var trainData = trainTestData.TrainSet;
            var testData = trainTestData.TestSet;

            //训练 
            IEstimator<ITransformer> dataProcessPipeline = mlContext.Transforms.Concatenate("Features", new[] { "Height", "Weight" })
                .Append(mlContext.Transforms.NormalizeMeanVariance(inputColumnName: "Features", outputColumnName: "FeaturesNormalizedByMeanVar"));
            IEstimator<ITransformer> trainer = mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Result", featureColumnName: "FeaturesNormalizedByMeanVar");
            IEstimator<ITransformer> trainingPipeline = dataProcessPipeline.Append(trainer); 
            ITransformer model = trainingPipeline.Fit(trainData);

            //评估
            var predictions = model.Transform(testData);
            var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Result", scoreColumnName: "Score");
            PrintBinaryClassificationMetrics(trainer.ToString(), metrics);

            //保存模型
            mlContext.Model.Save(model, trainData.Schema, ModelPath);
            Console.WriteLine($"Model file saved to :{ModelPath}");
        }

        static void LoadAndPrediction()
        {
            var mlContext = new MLContext();
            ITransformer model = mlContext.Model.Load(ModelPath, out var inputSchema);
            var predictionEngine = mlContext.Model.CreatePredictionEngine<FigureData, FigureDatePredicted>(model);

            FigureData test = new FigureData();
            test.Weight = 115;
            test.Height = 171;

            var prediction = predictionEngine.Predict(test);
            Console.WriteLine($"Predict Result :{prediction.PredictedLabel}");
        }

        public static void PrintBinaryClassificationMetrics(string name, CalibratedBinaryClassificationMetrics metrics)
        {
            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*       Metrics for {name} binary classification model      ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"*       Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"*       Area Under Curve:      {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"*       Area under Precision recall Curve:  {metrics.AreaUnderPrecisionRecallCurve:P2}");
            Console.WriteLine($"*       F1Score:  {metrics.F1Score:P2}");
            Console.WriteLine($"*       LogLoss:  {metrics.LogLoss:#.##}");
            Console.WriteLine($"*       LogLossReduction:  {metrics.LogLossReduction:#.##}");
            Console.WriteLine($"*       PositivePrecision:  {metrics.PositivePrecision:#.##}");
            Console.WriteLine($"*       PositiveRecall:  {metrics.PositiveRecall:#.##}");
            Console.WriteLine($"*       NegativePrecision:  {metrics.NegativePrecision:#.##}");
            Console.WriteLine($"*       NegativeRecall:  {metrics.NegativeRecall:P2}");
            Console.WriteLine($"************************************************************");
        }
    }

    public class FigureData
    {
        [LoadColumn(0)]
        public float Height { get; set; }

        [LoadColumn(1)]
        public float Weight { get; set; }

        [LoadColumn(2)]
        public bool Result { get; set; }       
    }

    public class FigureDatePredicted : FigureData
    {
        public bool PredictedLabel;
    }
}
