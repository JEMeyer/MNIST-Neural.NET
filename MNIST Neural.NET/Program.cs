using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MNIST_Neural.NET
{
    using System;
    using System.IO;
    using System.Runtime.Serialization;
    using System.Runtime.Serialization.Formatters.Binary;
    using Neural.NET;

    /// <summary>
    /// Class used to rest the Neural.NET network.
    /// </summary>
    public class Program
    {
        /// <summary>
        /// The main function to run. Will likely change over time depending on what stage of development is being tested.
        /// </summary>
        public static void Main()
        {
            //Network _network = Initialize();
            Network _network = new Network(784, new[] { 400, 100 }, 10);
            Train(_network);
            IFormatter _formatter = new BinaryFormatter();
            using (Stream _stream = new FileStream($"C:/Temp/NeuralNetworks/Network-{DateTime.Now.ToShortDateString()}.bin",
                                     FileMode.Create,
                                     FileAccess.Write, FileShare.None))
            {
                _formatter.Serialize(_stream, _network);
            }
            Console.Read();
        }

        /// <summary>
        /// Trains the network
        /// </summary>
        /// <param name="network">the network to train</param>
        private static void Train(Network network)
        {
            Tuple<double[], double[]>[] _trainingData = GetMNISTData(true, 60000);
            Tuple<double[], double[]>[] _testingData = GetMNISTData(false, 5000);

            NetworkTrainer _trainer = new NetworkTrainer(network);
            foreach (Tuple<int, double?> _result in _trainer.StochasticGradientDescent(10000, _trainingData, 1000, 100, .05, _testingData))
            {
                Console.WriteLine($"Epoch {_result.Item1}: {_result.Item2}% error");
            }
        }

        private static Tuple<double[], double[]>[] GetMNISTData(bool training, int recordCount)
        {
            // We have caps people!
            if (training && recordCount > 60000)
            {
                recordCount = 60000;
            }
            else if (!training && recordCount > 10000)
            {
                recordCount = 10000;
            }

            string _labelsLocation = training ?
                @"C:\MNIST\train-labels.idx1-ubyte" :
                @"C:\MNIST\t10k-labels.idx1-ubyte";

            string _imagesLocation = training ?
                @"C:\MNIST\train-images.idx3-ubyte" :
                @"C:\MNIST\t10k-images.idx3-ubyte";

            FileStream _labels = new FileStream(_labelsLocation, FileMode.Open);
            FileStream _images = new FileStream(_imagesLocation, FileMode.Open);

            BinaryReader _brLabels = new BinaryReader(_labels);
            BinaryReader _brImages = new BinaryReader(_images);

            // Discard
            int _magic1 = ReverseBytes(_brImages.ReadInt32());
            int _magic2 = ReverseBytes(_brLabels.ReadInt32());

            int _numImages = ReverseBytes(_brImages.ReadInt32());
            int _numRows = ReverseBytes(_brImages.ReadInt32());
            int _numCols = ReverseBytes(_brImages.ReadInt32());

            int _numLabels = ReverseBytes(_brLabels.ReadInt32());

            Tuple<double[], double[]>[] _data = new Tuple<double[], double[]>[_numImages];

            int _index = 0;

            for (int i = 0; i < _numImages; i++)
            {
                double[] _image = new double[784];
                for (int j = 0; j < 28; j++)
                {
                    for (int k = 0; k < 28; k++)
                    {
                        _index = j * 28 + k;
                        //Console.WriteLine($"Pixel {_index}:{_brImages.ReadByte()}");
                        _image[_index] = Convert.ToDouble(_brImages.ReadByte()) / 100;
                    }
                }

                int _label = Convert.ToInt32(_brLabels.ReadByte());

                _data[i] = new Tuple<double[], double[]>(
                    _image,
                    new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
                _data[i].Item2[_label] = 1;
            }

            _labels.Close();
            _images.Close();
            _brImages.Close();
            _brLabels.Close();

            return _data;
        }

        public static int ReverseBytes(int v)
        {
            byte[] intAsBytes = BitConverter.GetBytes(v);
            Array.Reverse(intAsBytes);
            return BitConverter.ToInt32(intAsBytes, 0);
        }
    }
}