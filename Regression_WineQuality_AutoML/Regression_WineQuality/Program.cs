using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Regression_WineQuality.Common;
using System;
using System.IO;
using System.Linq;

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
        public float ID; 
    }

    public class WinePrediction
    {
        [ColumnName("Score")]
        public float PredictionQuality;
    }
 

    class Program
    {
        static readonly string ModelFilePath = Path.Combine(Environment.CurrentDirectory, "MLModel", "model.zip");
        static readonly string TrainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "winequality-data-train.csv");
        static readonly string TestDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "winequality-data-test.csv");

        static void Main(string[] args)
        {           
            TrainAndSave();
            LoadAndPrediction();

            Console.WriteLine("Hit any key to finish the app");
            Console.ReadKey();
        }

        public static void TrainAndSave()
        {
            MLContext mlContext = new MLContext(seed: 1);

            // 准备数据 
            var trainData = mlContext.Data.LoadFromTextFile<WineData>(path: TrainDataPath, separatorChar: ',', hasHeader: true);
            var testData = mlContext.Data.LoadFromTextFile<WineData>(path: TestDataPath, separatorChar: ',', hasHeader: true);
         
            var progressHandler = new RegressionExperimentProgressHandler();
            uint ExperimentTime = 200;

            ExperimentResult<RegressionMetrics> experimentResult = mlContext.Auto()
               .CreateRegressionExperiment(ExperimentTime)
               .Execute(trainData, "Label", progressHandler: progressHandler);           

            Debugger.PrintTopModels(experimentResult);

            RunDetail<RegressionMetrics> best = experimentResult.BestRun;
            ITransformer trainedModel = best.Model;

            // 评估 BestRun
            var predictions = trainedModel.Transform(testData);
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");
            Debugger.PrintRegressionMetrics(best.TrainerName, metrics);

            // 保存模型
            Console.WriteLine("====== Save model to local file =========");
            mlContext.Model.Save(trainedModel, trainData.Schema, ModelFilePath);           
        }
       

        static void LoadAndPrediction()
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
    }
}
