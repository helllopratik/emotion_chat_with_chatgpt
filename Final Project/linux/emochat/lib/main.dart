import 'dart:convert';
import 'dart:typed_data';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:http/http.dart' as http;
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

Future<void> main() async {
  await dotenv.load(fileName: ".env");
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  final firstCamera = cameras.first;
  runApp(MyApp(camera: firstCamera));
}

class MyApp extends StatelessWidget {
  final CameraDescription camera;

  MyApp({required this.camera});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData.dark(),
      home: EmotionDetectionScreen(camera: camera),
    );
  }
}

class EmotionDetectionScreen extends StatefulWidget {
  final CameraDescription camera;

  EmotionDetectionScreen({required this.camera});

  @override
  _EmotionDetectionScreenState createState() => _EmotionDetectionScreenState();
}

class _EmotionDetectionScreenState extends State<EmotionDetectionScreen> {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;
  String _detectedEmotion = "No emotion detected";
  final TextEditingController _questionController = TextEditingController();
  String _chatResponse = '';

  @override
  void initState() {
    super.initState();
    _controller = CameraController(
      widget.camera,
      ResolutionPreset.medium,
    );
    _initializeControllerFuture = _controller.initialize();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Future<String> detectEmotion(String imagePath) async {
    final interpreter =
        await tfl.Interpreter.fromAsset('emotiondetector.tflite');
    final labels = [
      'angry',
      'disgust',
      'fear',
      'happy',
      'neutral',
      'sad',
      'surprise'
    ];

    final imageBytes = await File(imagePath).readAsBytes();
    final img.Image? image = img.decodeImage(imageBytes);
    if (image == null) return 'Error loading image';

    final resizedImage = img.copyResize(image, width: 48, height: 48);
    final input = imageToByteList(resizedImage, 48);
    final output = List.filled(7, 0.0).reshape([1, 7]);

    interpreter.run(input, output);
    final emotionIndex =
        output[0].indexWhere((element) => element == output[0].reduce(max));
    return labels[emotionIndex];
  }

  Uint8List imageToByteList(img.Image image, int inputSize) {
    var convertedBytes = Float32List(1 * inputSize * inputSize * 1);
    var buffer = Float32List.view(convertedBytes.buffer);
    int pixelIndex = 0;
    for (var i = 0; i < inputSize; i++) {
      for (var j = 0; j < inputSize; j++) {
        var pixel = image.getPixel(j, i);
        buffer[pixelIndex++] = (img.getRed(pixel) / 255.0);
      }
    }
    return convertedBytes.buffer.asUint8List();
  }

  Future<String> getChatResponse(
      String question, String detectedEmotion) async {
    final apiKey = dotenv.env['GEMINI_API_KEY'];
    final response = await http.post(
      Uri.parse(
          'https://api.example.com/gemini-pro/chat'), // Update with the correct API endpoint
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer $apiKey',
      },
      body: jsonEncode({
        'input': {
          'text': question,
          'emotion': detectedEmotion,
        },
      }),
    );
    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return data['response']; // Adjust based on actual API response structure
    } else {
      throw Exception('Failed to get response from the chat model');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Emotion Detection and Chat')),
      body: SingleChildScrollView(
        child: Column(
          children: [
            FutureBuilder<void>(
              future: _initializeControllerFuture,
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.done) {
                  return CameraPreview(_controller);
                } else {
                  return Center(child: CircularProgressIndicator());
                }
              },
            ),
            SizedBox(height: 20),
            Text('Detected Emotion: $_detectedEmotion'),
            ElevatedButton(
              onPressed: () async {
                try {
                  await _initializeControllerFuture;
                  final image = await _controller.takePicture();
                  final detectedEmotion = await detectEmotion(image.path);
                  setState(() {
                    _detectedEmotion = detectedEmotion;
                  });
                } catch (e) {
                  print(e);
                }
              },
              child: Text('Detect Emotion'),
            ),
            Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                children: [
                  TextField(
                    controller: _questionController,
                    decoration: InputDecoration(labelText: 'Ask a question'),
                  ),
                  SizedBox(height: 20),
                  ElevatedButton(
                    onPressed: () async {
                      final question = _questionController.text;
                      final response =
                          await getChatResponse(question, _detectedEmotion);
                      setState(() {
                        _chatResponse = response;
                      });
                    },
                    child: Text('Send'),
                  ),
                  SizedBox(height: 20),
                  Text('Response: $_chatResponse'),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
