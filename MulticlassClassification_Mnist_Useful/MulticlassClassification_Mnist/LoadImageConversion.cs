using System;
using System.Drawing;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace MulticlassClassification_Mnist
{
    public class LoadImageConversionInput
    {
        public string  FileName { get; set; }
    }
 
    public class LoadImageConversionOutput
    {
        [VectorType(400)]
        public float[] ImagePixels { get; set; }

        public string ImagePath;
    }

    [CustomMappingFactoryAttribute("LoadImageConversionAction")]
    public class LoadImageConversion : CustomMappingFactory<LoadImageConversionInput, LoadImageConversionOutput>
    {
        static long Count = 0;
        static long TotalCount = 0;
        static readonly string TrainDataFolder = @"D:\StepByStep\Blogs\ML_Assets\MNIST\train";

        public void CustomAction(LoadImageConversionInput input, LoadImageConversionOutput output)
        {  
            string ImagePath = Path.Combine(TrainDataFolder, input.FileName);
            output.ImagePath = ImagePath;

            Bitmap bmp = Image.FromFile(ImagePath) as Bitmap;           

            output.ImagePixels = new float[400];
            for (int x = 0; x < 20; x++)
                for (int y = 0; y < 20; y++)
                {
                    var pixel = bmp.GetPixel(x, y);
                    var gray = (pixel.R + pixel.G + pixel.B) / 3 / 16;
                    output.ImagePixels[x + y * 20] = gray;
                }           
            bmp.Dispose();

            Count++;
            if (Count / 10000 > TotalCount)
            {
                TotalCount = Count / 10000;
                Console.WriteLine($"LoadImageConversion.CustomAction's debug info.TotalCount={TotalCount}0000 ");
            }            
        }

        public override Action<LoadImageConversionInput, LoadImageConversionOutput> GetMapping()
              => CustomAction;
    }
}
