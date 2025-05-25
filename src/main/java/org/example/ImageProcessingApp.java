package org.example;

import javax.swing.*;
import org.opencv.core.*;
import nu.pattern.OpenCV;
import org.opencv.features2d.SIFT;

public class ImageProcessingApp extends JFrame {
    private JTabbedPane tabbedPane;

    // Статический блок для загрузки OpenCV
    static {
        try {
            // Попробуем сначала загрузить локально (работает с Java 12+)
            OpenCV.loadLocally();
            System.out.println("OpenCV loaded successfully using loadLocally()");
        } catch (UnsatisfiedLinkError e) {
            try {
                // Fallback 1: стандартный способ
                System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
                System.out.println("OpenCV loaded using System.loadLibrary()");
            } catch (UnsatisfiedLinkError e2) {
                // Fallback 2: явное указание пути к DLL
                try {
                    System.load("C:/opencv/build/java/x64/opencv_java455.dll");
                    System.out.println("OpenCV loaded from explicit path");
                } catch (UnsatisfiedLinkError e3) {
                    JOptionPane.showMessageDialog(null,
                            "Failed to load OpenCV library:\n" +
                                    "1. " + e.getMessage() + "\n" +
                                    "2. " + e2.getMessage() + "\n" +
                                    "3. " + e3.getMessage(),
                            "OpenCV Error", JOptionPane.ERROR_MESSAGE);
                    System.exit(1);
                }
            }
        }
    }

    public ImageProcessingApp() {
        setTitle("Image Processing Application - OpenCV " + Core.VERSION);
        setSize(800, 600);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null); // Центрируем окно

        // Проверка функций OpenCV при запуске
        testOpenCVFunctionality();

        tabbedPane = new JTabbedPane();

        // Добавляем вкладки
        tabbedPane.addTab("Keypoint Detection", new KeypointDetectionPanel());
        tabbedPane.addTab("Image Comparison", new ImageComparisonPanel());
        tabbedPane.addTab("Background Subtraction", new BackgroundSubtractionPanel());
        tabbedPane.addTab("Motion Blur", new MotionBlurPanel());

        add(tabbedPane);
    }

    private void testOpenCVFunctionality() {
        try {
            // Создаем тестовое изображение
            Mat testImage = new Mat(100, 100, CvType.CV_8UC3, new Scalar(100, 100, 100));

            // Проверяем детектор SIFT
            SIFT sift = SIFT.create();
            MatOfKeyPoint keypoints = new MatOfKeyPoint();
            sift.detect(testImage, keypoints);

            System.out.println("OpenCV test passed. Detected keypoints: " + keypoints.size().height);
        } catch (Exception e) {
            JOptionPane.showMessageDialog(this,
                    "OpenCV functionality test failed:\n" + e.getMessage(),
                    "Test Failed", JOptionPane.ERROR_MESSAGE);
        }
    }

    public static void main(String[] args) {
        // Выводим информацию о версии Java
        System.out.println("Java version: " + System.getProperty("java.version"));

        SwingUtilities.invokeLater(() -> {
            try {
                ImageProcessingApp app = new ImageProcessingApp();
                app.setVisible(true);
            } catch (Exception e) {
                JOptionPane.showMessageDialog(null,
                        "Application failed to start:\n" + e.getMessage(),
                        "Fatal Error", JOptionPane.ERROR_MESSAGE);
                e.printStackTrace();
            }
        });
    }
}