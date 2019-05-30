using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace TensorFlow_ImageClassification
{    

    class Program
    {
        //Assets files download from:https://gitee.com/seabluescn/ML_Assets
        static readonly string AssetsFolder = @"D:\StepByStep\Blogs\ML_Assets";
        static readonly string TrainDataFolder = Path.Combine(AssetsFolder, "FaceValueDetection", "SCUT-FBP5500");
        static readonly string TrainTagsPath = Path.Combine(AssetsFolder, "FaceValueDetection", "SCUT-FBP5500_asia_full.csv");
        static readonly string TestDataFolder = Path.Combine(AssetsFolder, "FaceValueDetection", "testimages");
        static readonly string inceptionPb = Path.Combine(AssetsFolder, "TensorFlow", "tensorflow_inception_graph.pb");
        static readonly string imageClassifierZip = Path.Combine(Environment.CurrentDirectory, "MLModel", "imageClassifier.zip");

        //配置用常量
        private struct ImageNetSettings
        {
            public const int imageHeight = 224;
            public const int imageWidth = 224;
            public const float mean = 117;
            public const float scale = 1;
            public const bool channelsLast = true;
        }

        static void Main(string[] args)
        {
            // TrainAndSaveModel();
            LoadAndPrediction();

            Console.WriteLine("Hit any key to finish the app");
            Console.ReadKey();
        }

        public static void TrainAndSaveModel()
        {
            MLContext mlContext = new MLContext(seed: 1);

            // STEP 1: 准备数据
            var fulldata = mlContext.Data.LoadFromTextFile<ImageNetData>(path: TrainTagsPath, separatorChar: ',', hasHeader: true);

            var trainTestData = mlContext.Data.TrainTestSplit(fulldata, testFraction: 0.2);
            var trainData = trainTestData.TrainSet;
            var testData = trainTestData.TestSet;

            // STEP 2：创建学习管道
            var pipeline = mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: TrainDataFolder, inputColumnName: nameof(ImageNetData.ImagePath))
                .Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: ImageNetSettings.imageWidth, imageHeight: ImageNetSettings.imageHeight, inputColumnName: "input"))
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: ImageNetSettings.channelsLast, offsetImage: ImageNetSettings.mean))
                .Append(mlContext.Model.LoadTensorFlowModel(inceptionPb).
                     ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: true))
                .Append(mlContext.Regression.Trainers.LbfgsPoissonRegression(labelColumnName: "Label", featureColumnName: "softmax2_pre_activation"));


            // STEP 3：通过训练数据调整模型  
            Console.WriteLine("======= training =========");
            Stopwatch stopWatch = new Stopwatch();
            stopWatch.Start();
            ITransformer model = pipeline.Fit(trainData);
            stopWatch.Stop();
            Console.WriteLine($"Used time : {stopWatch.Elapsed}");

            // STEP 4：评估模型
            Console.WriteLine("===== Evaluate model =======");
            stopWatch.Start();
            var predictions = model.Transform(testData); 
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");
            PrintRegressionMetrics( metrics);
            stopWatch.Stop();
            Console.WriteLine($"Used time : {stopWatch.Elapsed}");

            //STEP 5：保存模型
            Console.WriteLine("====== Save model to local file =========");
            mlContext.Model.Save(model, trainData.Schema, imageClassifierZip);
        }

        static void LoadAndPrediction()
        {
            MLContext mlContext = new MLContext(seed: 1);

            // Load the model
            ITransformer loadedModel = mlContext.Model.Load(imageClassifierZip, out var modelInputSchema);

            // Make prediction function (input = ImageNetData, output = ImageNetPrediction)
            var predictor = mlContext.Model.CreatePredictionEngine<ImageNetData, ImageNetPrediction>(loadedModel);
            
            DirectoryInfo testdir = new DirectoryInfo(TestDataFolder);
            foreach (var jpgfile in testdir.GetFiles("*.jpg"))
            {
                ImageNetData image = new ImageNetData();
                image.ImagePath = jpgfile.FullName;
                var pred = predictor.Predict(image);

                Console.WriteLine($"Filename:{jpgfile.Name}:\tPredict:{pred.FaceValue}");
            }
        }

        public static void PrintRegressionMetrics(RegressionMetrics metrics)
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Metrics for L-BFGS regression model      ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       LossFn:        {metrics.LossFunction:0.##}");
            Console.WriteLine($"*       R2 Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Absolute loss: {metrics.MeanAbsoluteError:#.##}");
            Console.WriteLine($"*       Squared loss:  {metrics.MeanSquaredError:#.##}");
            Console.WriteLine($"*       RMS loss:      {metrics.RootMeanSquaredError:#.##}");
            Console.WriteLine($"*************************************************");
        }
    }

    public class ImageNetData
    {
        [LoadColumn(0)]
        public string ImagePath;

        [LoadColumn(1)]
        public float Label;
    }

    public class ImageNetPrediction
    {
        [ColumnName("Score")]
        public float FaceValue;
    }   
}
