using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace MulticlassClassification_Mnist
{
    class Program
    {
        static readonly string TrainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "optdigits-full.csv");
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
            var fulldata = mlContext.Data.LoadFromTextFile(path: TrainDataPath,
                    columns: new[]
                    {
                        new TextLoader.Column("Serial", DataKind.Single, 0),
                        new TextLoader.Column("PixelValues", DataKind.Single, 1, 64),
                        new TextLoader.Column("Number", DataKind.Single, 65)
                    },
                    hasHeader: true,
                    separatorChar: ','
                    );

            var trainTestData = mlContext.Data.TrainTestSplit(fulldata, testFraction: 0.2);
            var trainData = trainTestData.TrainSet;
            var testData = trainTestData.TestSet;

            // STEP 2: 配置数据处理管道        
            var dataProcessPipeline = mlContext.Transforms.CustomMapping(new DebugConversion().GetMapping(), contractName: "DebugConversionAction")
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Label", "Number", keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue))
                .Append(mlContext.Transforms.Concatenate("Features", new string[] { "PixelValues", "DebugFeature" }));

            // STEP 3: 配置训练算法
            var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer)
              .Append(mlContext.Transforms.Conversion.MapKeyToValue("Number", "Label"));
            
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
            var metrics = mlContext.MulticlassClassification.Evaluate(data: predictions, labelColumnName: "Number", scoreColumnName: "Score");
            PrintMultiClassClassificationMetrics(trainer.ToString(), metrics);
            DebugData(mlContext, predictions);

            // STEP 6:保存模型              
            mlContext.ComponentCatalog.RegisterAssembly(typeof(DebugConversion).Assembly);
            mlContext.Model.Save(trainedModel, trainData.Schema, ModelPath);
            Console.WriteLine("The model is saved to {0}", ModelPath);
        }

        private static void TestSomePredictions(MLContext mlContext)
        {
            // Load Model           
            ITransformer trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);

            // Create prediction engine 
            var predEngine = mlContext.Model.CreatePredictionEngine<InputData, OutPutData>(trainedModel);

            //num 1
            InputData MNIST1 = new InputData()
            {               
                PixelValues = new float[] { 0, 0, 0, 0, 14, 13, 1, 0, 0, 0, 0, 5, 16, 16, 2, 0, 0, 0, 0, 14, 16, 12, 0, 0, 0, 1, 10, 16, 16, 12, 0, 0, 0, 3, 12, 14, 16, 9, 0, 0, 0, 0, 0, 5, 16, 15, 0, 0, 0, 0, 0, 4, 16, 14, 0, 0, 0, 0, 0, 1, 13, 16, 1, 0 }
            }; 
            var resultprediction1 = predEngine.Predict(MNIST1);
            resultprediction1.PrintToConsole();

            //num 7
            InputData MNIST2 = new InputData()
            {               
                PixelValues = new float[] { 0, 0, 1, 8, 15, 10, 0, 0, 0, 3, 13, 15, 14, 14, 0, 0, 0, 5, 10, 0, 10, 12, 0, 0, 0, 0, 3, 5, 15, 10, 2, 0, 0, 0, 16, 16, 16, 16, 12, 0, 0, 1, 8, 12, 14, 8, 3, 0, 0, 0, 0, 10, 13, 0, 0, 0, 0, 0, 0, 11, 9, 0, 0, 0 }
            };
            var resultprediction2 = predEngine.Predict(MNIST2);
            resultprediction2.PrintToConsole();
        }

        private static void DebugData(MLContext mlContext, IDataView predictions)
        {
            var loadedModelOutputColumnNames = predictions.Schema.Where(col => !col.IsHidden).Select(col => col.Name);
            foreach (string column in loadedModelOutputColumnNames)
            {
                Console.WriteLine($"loadedModelOutputColumnNames:{ column }");
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
        public float Serial;
        [VectorType(64)]
        public float[] PixelValues;               
        public float Number;       
    }

    class OutPutData : InputData
    {  
        public float[] Score;       

        public void PrintToConsole()
        {  
            Console.WriteLine($"Predicted probability:     zero:  {Score[0]:0.####}");
            Console.WriteLine($"                           One :  {Score[1]:0.####}");
            Console.WriteLine($"                           two:   {Score[2]:0.####}");
            Console.WriteLine($"                           three: {Score[3]:0.####}");
            Console.WriteLine($"                           four:  {Score[4]:0.####}");
            Console.WriteLine($"                           five:  {Score[5]:0.####}");
            Console.WriteLine($"                           six:   {Score[6]:0.####}");
            Console.WriteLine($"                           seven: {Score[7]:0.####}");
            Console.WriteLine($"                           eight: {Score[8]:0.####}");
            Console.WriteLine($"                           nine:  {Score[9]:0.####}");           
        }
    }   
}
