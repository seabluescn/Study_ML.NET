using System;
using System.IO;

namespace MakeData
{
    class Program
    {
        static void Main(string[] args)
        {
            string Filename = "./figure_full.csv";
            StreamWriter sw = new StreamWriter(Filename, false);
            sw.WriteLine("Height,Weight,Result");

            Random random = new Random();

            float height, weight;
            Result result;

            for (int i = 0; i < 2000; i++)
            {
                height = random.Next(150, 195);
                weight = random.Next(70, 200);

                if (height > 170 && weight < 120)
                    result = Result.Good;
                else
                    result = Result.Bad;

                Console.WriteLine($"{height},{weight},{(int)result}");
                sw.WriteLine($"{height},{weight},{(int)result}");
            }

            sw.Close();
            sw?.Dispose();

            Console.ReadKey();
        }
    }

    enum Result
    {
        Bad=0,
        Good=1
    }
}
