using Microsoft.ML;
using ObjectDetection_Common;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;

namespace YOLO_ObjectDetection
{
    class Program
    {
        //Assets files download from:https://gitee.com/seabluescn/ML_Assets
        static readonly string AssetsFolder = @"D:\StepByStep\Blogs\ML_Assets";
        static readonly string YOLO_ModelFilePath = Path.Combine(AssetsFolder, "ObjectDetection", "YoloModel", "TinyYolo2_model.onnx");       
        static readonly string trainImagesFolder = Path.Combine(AssetsFolder, "ObjectDetection", "train");       
        static readonly string tagsTsv = Path.Combine(trainImagesFolder,  "tags.tsv");
        static readonly string ObjectDetectionModelFilePath = Path.Combine(Environment.CurrentDirectory, "MLModel", "ObjectDetectionModel.zip");
        static readonly string testimagesFolder = Path.Combine(AssetsFolder, "ObjectDetection", "testimages");

        static void Main(string[] args)
        {
            //TrainAndSave();
            LoadAndPredict();

            Console.WriteLine("Press any key to exit!");
            Console.ReadKey();
        }

        private static void TrainAndSave()
        {
            var mlContext = new MLContext();
            var trainData = mlContext.Data.LoadFromTextFile<ImageNetData>(tagsTsv);

            var pipeline = mlContext.Transforms.LoadImages(outputColumnName: "image", imageFolder: trainImagesFolder, inputColumnName: nameof(ImageNetData.ImagePath))
                            .Append(mlContext.Transforms.ResizeImages(outputColumnName: "image", imageWidth: ImageNetSettings.imageWidth, imageHeight: ImageNetSettings.imageHeight, inputColumnName: "image"))
                            .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "image"))
                            .Append(mlContext.Transforms.ApplyOnnxModel(modelFile: YOLO_ModelFilePath, outputColumnNames: new[] { TinyYoloModelSettings.ModelOutput }, inputColumnNames: new[] { TinyYoloModelSettings.ModelInput }));

            var model = pipeline.Fit(trainData);


            using (var file = File.OpenWrite(ObjectDetectionModelFilePath))
                mlContext.Model.Save(model, trainData.Schema, file);

            Console.WriteLine("Save Model success!");
        }

        private static void LoadAndPredict()
        {
            var mlContext = new MLContext();

            ITransformer trainedModel;
            using (var stream = File.OpenRead(ObjectDetectionModelFilePath))
            {
                trainedModel = mlContext.Model.Load(stream, out var modelInputSchema);
                Console.WriteLine("Load Model file success!");
            }
            var predictionEngine = mlContext.Model.CreatePredictionEngine<ImageNetData, ImageNetPrediction>(trainedModel);

            DirectoryInfo testdir = new DirectoryInfo(testimagesFolder);
            foreach (var jpgfile in testdir.GetFiles("*.jpg"))
            {  
                ImageNetData image = new ImageNetData
                {
                    ImagePath = jpgfile.FullName
                };

                Console.WriteLine($".....PredictImage: {image.ImagePath} ....");
                
                var Predicted = predictionEngine.Predict(image);
                PredictImage(image.ImagePath, Predicted);              

                Console.WriteLine("");
            }
        }

        private static void PredictImage(string filename, ImageNetPrediction Predicted)
        {
            //获取对象信息
            Console.WriteLine(".....The objects in the image are detected as below....");

            YoloWinMlParser _parser = new YoloWinMlParser();
            IList<YoloBoundingBox> boundingBoxes = _parser.ParseOutputs(Predicted.PredictedLabels, 0.4f);
            foreach (var box in boundingBoxes)
            {
                Console.WriteLine(box.Label + " and its Confidence score: " + box.Confidence + "Location: " + box.Rect);
            }

            //过滤
            Console.WriteLine(".....The filtered objects as below....");

            var filteredBoxes = _parser.NonMaxSuppress(boundingBoxes, 5, 0.6F);

            Bitmap bitmapSource = Image.FromFile(filename) as Bitmap;
            Bitmap bitmapOut = new Bitmap(bitmapSource, ImageNetSettings.imageWidth, ImageNetSettings.imageHeight);
            Graphics graphics = Graphics.FromImage(bitmapOut);
            
            foreach (var box in filteredBoxes)
            {
                Console.WriteLine(box.Label + " and its Confidence score: " + box.Confidence + "Location: " + box.Rect);
                graphics.DrawRectangle(Pens.Red, box.Rect.X, box.Rect.Y, box.Rect.Width, box.Rect.Height);
                graphics.DrawString(box.Label, new Font("宋体", 15), Brushes.Red, box.X + 10, box.Y);
            }           

            bitmapOut.Save(filename + "_predicted.png");
            bitmapOut?.Dispose();
            bitmapSource?.Dispose();
        }
    }
}
