using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;

namespace Regression_WineQuality
{
    public class WineData
    {
        [LoadColumn(0)]
        public float FixedAcidity;

        [LoadColumn(1)]
        public float VolatileAcidity;

        [LoadColumn(2)]
        public float CitricACID;

        [LoadColumn(3)]
        public float ResidualSugar;

        [LoadColumn(4)]
        public float Chlorides;

        [LoadColumn(5)]
        public float FreeSulfurDioxide;

        [LoadColumn(6)]
        public float TotalSulfurDioxide;

        [LoadColumn(7)]
        public float Density;

        [LoadColumn(8)]
        public float PH;

        [LoadColumn(9)]
        public float Sulphates;

        [LoadColumn(10)]
        public float Alcohol;
      
        [LoadColumn(11)]
        [ColumnName("Label")]
        public float Quality;
       
        [LoadColumn(12)]
        public float Id;
    }

    public class WinePrediction
    {
        [ColumnName("Score")]
        public float PredictionQuality;
    }

    class Program
    {
        static readonly string ModelFilePath = Path.Combine(Environment.CurrentDirectory, "MLModel", "model.zip");

        static void Main(string[] args)
        { 
            Train();
            Prediction();

            Console.WriteLine("Hit any key to finish the app");
            Console.ReadKey();
        }

        public static void Train()
        {
            MLContext mlContext = new MLContext(seed: 1);

            // 准备数据
            string TrainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "winequality-data-full.csv");
            var fulldata = mlContext.Data.LoadFromTextFile<WineData>(path: TrainDataPath, separatorChar: ',', hasHeader: true);

            var trainTestData = mlContext.Data.TrainTestSplit(fulldata, testFraction: 0.2);
            var trainData = trainTestData.TrainSet;
            var testData = trainTestData.TestSet;

            // 创建学习管道并通过训练数据调整模型  
            var dataProcessPipeline = mlContext.Transforms.DropColumns("Id")
                .Append(mlContext.Transforms.NormalizeMeanVariance(nameof(WineData.FreeSulfurDioxide)))
                .Append(mlContext.Transforms.NormalizeMeanVariance(nameof(WineData.TotalSulfurDioxide)))
                .Append(mlContext.Transforms.Concatenate("Features", new string[] { nameof(WineData.FixedAcidity),
                                                                                    nameof(WineData.VolatileAcidity),
                                                                                    nameof(WineData.CitricACID),
                                                                                    nameof(WineData.ResidualSugar),
                                                                                    nameof(WineData.Chlorides),
                                                                                    nameof(WineData.FreeSulfurDioxide),
                                                                                    nameof(WineData.TotalSulfurDioxide),
                                                                                    nameof(WineData.Density),
                                                                                    nameof(WineData.PH),
                                                                                    nameof(WineData.Sulphates),
                                                                                    nameof(WineData.Alcohol)}));

            var trainer = mlContext.Regression.Trainers.LbfgsPoissonRegression(labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer);
            var trainedModel = trainingPipeline.Fit(trainData);

            // 评估
            var predictions = trainedModel.Transform(testData);
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");
            PrintRegressionMetrics(trainer.ToString(), metrics);

            // 保存模型
            Console.WriteLine("====== Save model to local file =========");
            mlContext.Model.Save(trainedModel, trainData.Schema, ModelFilePath);
        }

        static void Prediction()
        {
            MLContext mlContext = new MLContext(seed: 1);

            ITransformer loadedModel = mlContext.Model.Load(ModelFilePath, out var modelInputSchema);
            var predictor = mlContext.Model.CreatePredictionEngine<WineData, WinePrediction>(loadedModel);

            WineData wineData = new WineData
            {
                FixedAcidity = 7.6f,
                VolatileAcidity = 0.33f,
                CitricACID = 0.36f,
                ResidualSugar = 2.1f,
                Chlorides = 0.034f,
                FreeSulfurDioxide = 26f,
                TotalSulfurDioxide = 172f,
                Density = 0.9944f,
                PH = 3.42f,
                Sulphates = 0.48f,
                Alcohol = 10.5f
            };

            var wineQuality = predictor.Predict(wineData);
            Console.WriteLine($"Wine Data  Quality is:{wineQuality.PredictionQuality} ");           
        }

        public static void PrintRegressionMetrics(string name, RegressionMetrics metrics)
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Metrics for {name} regression model      ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       LossFn:        {metrics.LossFunction:0.##}");
            Console.WriteLine($"*       R2 Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Absolute loss: {metrics.MeanAbsoluteError:#.##}");
            Console.WriteLine($"*       Squared loss:  {metrics.MeanSquaredError:#.##}");
            Console.WriteLine($"*       RMS loss:      {metrics.RootMeanSquaredError:#.##}");
            Console.WriteLine($"*************************************************");
        }
    }
}
