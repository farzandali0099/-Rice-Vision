using System;
using System.IO;
using System.Threading.Tasks;
using Tensorflow;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace RiceVision.Services
{
    public class RiceDetectionService : IDisposable
    {
        private SavedModelBundle _modelBundle;
        private Session _session;
        private Graph _graph;
        private readonly string _modelPath = @"D:\EAD\RiceVision\Models";

        // Get these names from saved_model_cli output
        private const string InputTensorName = "serving_default_input_1:0";
        private const string OutputTensorName = "StatefulPartitionedCall:0";

        public RiceDetectionService()
        {
            LoadModel();
        }

        private void LoadModel()
        {
            try
            {
                if (!File.Exists(Path.Combine(_modelPath, "saved_model.pb")))
                    throw new FileNotFoundException($"Model not found at {_modelPath}");

                _modelBundle = SavedModelBundle.Load(_modelPath, new[] { "serve" });
                _session = _modelBundle.Session;
                _graph = _modelBundle.Graph;

                Console.WriteLine($"Model loaded successfully. Input: {InputTensorName}, Output: {OutputTensorName}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Model load error: {ex.Message}");
                Dispose();
            }
        }

        public async Task<bool> PredictInfection(byte[] imageData)
        {
            if (_session == null)
                throw new InvalidOperationException("Model not loaded");

            try
            {
                var inputTensor = await Task.Run(() => PreprocessImage(imageData));

                var runner = _session.Runner()
                    .AddInput(_graph[InputTensorName][0], inputTensor)
                    .Fetch(_graph[OutputTensorName][0]);

                var results = await Task.Run(() => runner.Run());
                var result = results[0].GetValue() as float[,];

                return result[0, 0] > 0.5f;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Inference error: {ex.Message}");
                return false;
            }
        }

        private Tensor PreprocessImage(byte[] imageData)
        {
            using var ms = new MemoryStream(imageData);
            using var image = Image.Load<Rgb24>(ms);

            // Ensure this matches model expectations
            image.Mutate(x => x.Resize(100, 100).Grayscale());

            var array = new float[1, 100, 100, 3];
            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < accessor.Height; y++)
                {
                    var row = accessor.GetRowSpan(y);
                    for (int x = 0; x < accessor.Width; x++)
                    {
                        array[0, y, x, 0] = row[x].R / 255.0f;
                        array[0, y, x, 1] = row[x].G / 255.0f;
                        array[0, y, x, 2] = row[x].B / 255.0f;
                    }
                }
            });

            return tf.constant(array);
        }

        public void Dispose()
        {
            _session?.Dispose();
            _graph?.Dispose();
            _modelBundle?.Dispose();
        }
    }
}