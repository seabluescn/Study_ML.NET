using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using static Microsoft.ML.Transforms.Image.ImagePixelExtractingEstimator;

namespace MulticlassClassification_Mnist
{
    class Program
    {
        static readonly string AssetsFolder = @"D:\StepByStep\Blogs\ML_Assets\MNIST";
        static readonly string TrainTagsPath = Path.Combine(AssetsFolder, "train_tags.tsv");
        static readonly string TrainDataFolder = Path.Combine(AssetsFolder, "train");
        static readonly string ModelPath = Path.Combine(Environment.CurrentDirectory, "Data", "SDCA-Model.zip");

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 1);
          
            TrainAndSaveModel(mlContext);
            TestSomePredictions(mlContext);

            Console.WriteLine("Hit any key to finish the app");
            Console.ReadKey();
        }

        public static void TrainAndSaveModel(MLContext mlContext)
        {
            // STEP 1: 准备数据
            var fulldata = mlContext.Data.LoadFromTextFile<InputData>(path: TrainTagsPath, separatorChar: '\t', hasHeader: false);

            var trainTestData = mlContext.Data.TrainTestSplit(fulldata, testFraction: 0.1);
            var trainData = trainTestData.TrainSet;
            var testData = trainTestData.TestSet;

            // STEP 2: 配置数据处理管道        
            var dataProcessPipeline = mlContext.Transforms.CustomMapping(new DebugConversion().GetMapping(), contractName: "DebugConversionAction")
               .Append(mlContext.Transforms.CustomMapping(new LoadImageConversion().GetMapping(), contractName: "LoadImageConversionAction"))
               .Append(mlContext.Transforms.Conversion.MapValueToKey("Label", "Number", keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue))
               .Append(mlContext.Transforms.Concatenate("Features", new string[] { "ImagePixels", "DebugFeature" }))
               .Append(mlContext.Transforms.NormalizeMeanVariance(inputColumnName: "Features", outputColumnName: "FeaturesNormalizedByMeanVar"));


            // STEP 3: 配置训练算法 (using a maximum entropy classification model trained with the L-BFGS method)
            var trainer = mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "Label", featureColumnName: "FeaturesNormalizedByMeanVar");
            var trainingPipeline = dataProcessPipeline.Append(trainer)
                 .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictNumber", "Label"));


            // STEP 4: 训练模型使其与数据集拟合
            Console.WriteLine("=============== Train the model fitting to the DataSet ===============");
            Stopwatch stopWatch = new Stopwatch();
            stopWatch.Start();
            ITransformer trainedModel = trainingPipeline.Fit(trainData);
            stopWatch.Stop();
            Console.WriteLine($"Used time : {stopWatch.Elapsed}");

            // STEP 5:评估模型的准确性
            Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
            var predictions = trainedModel.Transform(testData);
            var metrics = mlContext.MulticlassClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");
            PrintMultiClassClassificationMetrics(trainer.ToString(), metrics);

            Console.WriteLine("===== Debug Data =====");
            DebugData(mlContext, predictions);


            // STEP 6:保存模型              
            mlContext.ComponentCatalog.RegisterAssembly(typeof(DebugConversion).Assembly);
            mlContext.ComponentCatalog.RegisterAssembly(typeof(LoadImageConversion).Assembly);
            mlContext.Model.Save(trainedModel, trainData.Schema, ModelPath);
            Console.WriteLine("The model is saved to {0}", ModelPath);
        }

        private static void TestSomePredictions(MLContext mlContext)
        {
            // Load Model           
            ITransformer trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);

            // Create prediction engine 
            var predEngine = mlContext.Model.CreatePredictionEngine<InputData, OutPutData>(trainedModel);

            Console.WriteLine("===== Test =====");
            DirectoryInfo TestFolder = new DirectoryInfo(Path.Combine(AssetsFolder, "test"));
            int count = 0;
            int success = 0;
            foreach(var image in TestFolder.GetFiles())
            {
                count++;

                InputData img = new InputData()
                {
                    FileName = image.Name
                };
                var result = predEngine.Predict(img);

                if(int.Parse(image.Name.Substring(0,1))==result.GetPredictResult())
                {
                    success++;
                }

                if(count%100==1)
                {
                    Console.WriteLine($"Current Source={img.FileName},PredictResult={result.GetPredictResult()},Success rate={success*100/count}%");
                }
            }
        }
         
        private static void DebugData(MLContext mlContext, IDataView predictions)
        {
            Console.WriteLine("DebugData...");
            Console.WriteLine("OutputColumns:");

            var OutputColumnNames = predictions.Schema.Where(col => !col.IsHidden).Select(col => col.Name);
            foreach (string column in OutputColumnNames)
            {
                Console.WriteLine($"OutputColumnName:{ column }");
            }

            var DataShowList = new List<OutPutData>(mlContext.Data.CreateEnumerable<OutPutData>(predictions, false, true));
            int Count = 0;          
            foreach (var dataline in DataShowList)
            { 
                Count++;
                if (Count % 1000 == 1)
                {
                    Console.WriteLine($"Curent Coun={Count}:");
                    dataline.PrintToConsole();
                }                
            }
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

    class InputData
    {
        [LoadColumn(0)]
        public string FileName;

        [LoadColumn(1)]
        public string Number;

        [LoadColumn(1)]
        public float Serial;       
    }

    class OutPutData : InputData
    {
        public string PredictNumber;
        public string ImagePath;
        public float[] ImagePixels;
        public float[] Score;
        public int GetPredictResult()
        {
            float max = 0;
            int index = 0;
            for (int i = 0; i < Score.Length; i++)
            {
                if (Score[i] > max)
                {
                    max = Score[i];
                    index = i;
                }
            }
            return index;
        }

        public void PrintToConsole()
        {  
            Console.WriteLine($"ImagePath={ImagePath},Number={Number},PredictNumber={PredictNumber}");

            int PredictResult = GetPredictResult();
            Console.WriteLine($"PredictResult={PredictResult}");
            
            Console.Write($"ImagePixels.Length={ImagePixels.Length},ImagePixels=[");
            for(int i=0;i<ImagePixels.Length;i++)
            {
                Console.Write($"{ImagePixels[i]},");
            }
            Console.WriteLine("]");

            Console.Write($"Score.Length={Score.Length},Score=[");
            for (int i = 0; i < Score.Length; i++)
            {
                Console.Write($"{Score[i]},");
            }
            Console.WriteLine("]");            
        }
    }   
}
