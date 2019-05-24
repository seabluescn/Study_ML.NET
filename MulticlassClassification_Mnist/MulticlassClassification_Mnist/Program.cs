using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.IO;

namespace MulticlassClassification_Mnist
{
    class Program
    {
        static readonly string DataPath = Path.Combine(Environment.CurrentDirectory, "Data", "optdigits-full.csv");
        static readonly string ModelPath = Path.Combine(Environment.CurrentDirectory, "Data", "SDCA-Model.zip");

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            TrainAndSaveModel(mlContext);

           // LoadAndPrediction();

            Console.WriteLine("Hit any key to finish the app");
            Console.ReadKey();
        }

        public static void TrainAndSaveModel(MLContext mlContext)
        {
            // STEP 1: 准备数据
            var fulldata = mlContext.Data.LoadFromTextFile(path: DataPath,
                    columns: new[]
                    {
                        new TextLoader.Column("Count", DataKind.Single, 0),
                        new TextLoader.Column("PixelValues", DataKind.Single, 1, 64),
                        new TextLoader.Column("Number", DataKind.Single, 65)
                    },
                    hasHeader: true,
                    separatorChar: ','
                    );

            var trainTestData = mlContext.Data.TrainTestSplit(fulldata, testFraction: 0.2);
            var trainData = trainTestData.TrainSet;
            var testData = trainTestData.TestSet;

            // STEP 2: 配置数据处理管道  mlContext.Transforms.CustomMapping<DebugLambdaInput, DebugLambdaOutput>(mapAction: DebugLambda.MyAction, contractName: "DebugLambda")          
            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Number", keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
                .Append(mlContext.Transforms.Concatenate("Features", "PixelValues")
                .AppendCacheCheckpoint(mlContext));     // AppendCacheCheckpoint Use in-memory cache for small/medium datasets to lower training time. Do NOT use it when handling very large datasets.

            // STEP 3: 配置训练算法
            var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("Number", "Label"));

           
            // STEP 4: 训练模型使其与数据集拟合
            Console.WriteLine("=============== Train the model fitting to the DataSet ===============");
            ITransformer trainedModel = trainingPipeline.Fit(trainData);

            /*
           // STEP 5:评估模型的准确性
           Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
           var predictions = trainedModel.Transform(testData);
           var metrics = mlContext.MulticlassClassification.Evaluate(data: predictions, labelColumnName: "Number", scoreColumnName: "Score");
           PrintMultiClassClassificationMetrics(trainer.ToString(), metrics);

           // STEP 6:保存模型
           mlContext.Model.Save(trainedModel, trainData.Schema, ModelPath);
           Console.WriteLine("The model is saved to {0}", ModelPath);
           */
        }

        public static void PrintMultiClassClassificationMetrics(string name, MulticlassClassificationMetrics metrics)
        {
            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*    Metrics for {name} multi-class classification model   ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"    AccuracyMacro = {metrics.MacroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    AccuracyMicro = {metrics.MicroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    LogLoss = {metrics.LogLoss:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 1 = {metrics.PerClassLogLoss[0]:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 2 = {metrics.PerClassLogLoss[1]:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 3 = {metrics.PerClassLogLoss[2]:0.####}, the closer to 0, the better");
            Console.WriteLine($"************************************************************");
        }
    }
}
