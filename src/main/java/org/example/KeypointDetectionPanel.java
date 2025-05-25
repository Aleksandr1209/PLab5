package org.example;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;
import org.opencv.features2d.FastFeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.SIFT;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;

class KeypointDetectionPanel extends JPanel {
    private JButton loadImageBtn;
    private JButton harrisBtn, siftBtn, surfBtn, fastBtn;
    private JLabel imageLabel;
    private Mat currentImage;
    private double scaleFactor = 1.0;
    private JScrollPane scrollPane;
    private JSlider zoomSlider;

    public KeypointDetectionPanel() {
        setLayout(new BorderLayout());

        // Панель кнопок
        JPanel buttonPanel = new JPanel();
        loadImageBtn = new JButton("Load Image");
        harrisBtn = new JButton("Harris");
        siftBtn = new JButton("SIFT");
        surfBtn = new JButton("SURF");
        fastBtn = new JButton("FAST");

        buttonPanel.add(loadImageBtn);
        buttonPanel.add(harrisBtn);
        buttonPanel.add(siftBtn);
        buttonPanel.add(surfBtn);
        buttonPanel.add(fastBtn);

        add(buttonPanel, BorderLayout.NORTH);

        // Область для отображения изображения
        imageLabel = new JLabel();
        add(new JScrollPane(imageLabel), BorderLayout.CENTER);

        // Обработчики событий
        loadImageBtn.addActionListener(e -> loadImage());
        harrisBtn.addActionListener(e -> detectHarris());
        siftBtn.addActionListener(e -> detectSIFT());
        surfBtn.addActionListener(e -> detectSURF());
        fastBtn.addActionListener(e -> detectFAST());

        // Настройка скролл-панели
        scrollPane = new JScrollPane(imageLabel);
        scrollPane.setPreferredSize(new Dimension(800, 600));
        add(scrollPane, BorderLayout.CENTER);

        // Панель управления масштабом
        JPanel controlPanel = new JPanel();
        zoomSlider = new JSlider(10, 200, 100);
        zoomSlider.addChangeListener(e -> {
            scaleFactor = zoomSlider.getValue() / 100.0;
            if (currentImage != null) {
                displayImage(currentImage);
            }
        });

        controlPanel.add(new JLabel("Масштаб:"));
        controlPanel.add(zoomSlider);
        add(controlPanel, BorderLayout.SOUTH);
    }

    private void loadImage() {
        JFileChooser fileChooser = new JFileChooser();
        if (fileChooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            File file = fileChooser.getSelectedFile();
            currentImage = Imgcodecs.imread(file.getAbsolutePath());
            displayImage(currentImage);
        }
    }

    private void detectHarris() {
        if (currentImage == null) return;

        Mat gray = new Mat();
        Imgproc.cvtColor(currentImage, gray, Imgproc.COLOR_BGR2GRAY);

        Mat corners = new Mat();
        Mat dst = new Mat();
        Imgproc.cornerHarris(gray, dst, 2, 3, 0.04);

        // Нормализация и масштабирование
        Mat dst_norm = new Mat();
        Core.normalize(dst, dst_norm, 0, 255, Core.NORM_MINMAX);
        Mat dst_norm_scaled = new Mat();
        Core.convertScaleAbs(dst_norm, dst_norm_scaled);

        // Рисование кругов вокруг углов
        Mat result = currentImage.clone();
        for (int i = 0; i < dst_norm.rows(); i++) {
            for (int j = 0; j < dst_norm.cols(); j++) {
                if (dst_norm.get(i, j)[0] > 150) {
                    // Используем org.opencv.core.Point вместо java.awt.Point
                    Imgproc.circle(result, new org.opencv.core.Point(j, i), 5, new Scalar(0, 0, 255), 2);
                }
            }
        }

        displayImage(result);
    }

    private void detectSIFT() {
        if (currentImage == null) return;

        Mat gray = new Mat();
        Imgproc.cvtColor(currentImage, gray, Imgproc.COLOR_BGR2GRAY);

        SIFT sift = SIFT.create();
        MatOfKeyPoint keypoints = new MatOfKeyPoint();
        sift.detect(gray, keypoints);

        Mat result = new Mat();
        Features2d.drawKeypoints(currentImage, keypoints, result);

        displayImage(result);
    }

    private void detectSURF() {
        if (currentImage == null) return;

        Mat gray = new Mat();
        Imgproc.cvtColor(currentImage, gray, Imgproc.COLOR_BGR2GRAY);

        SIFT surf = SIFT.create();
        MatOfKeyPoint keypoints = new MatOfKeyPoint();
        surf.detect(gray, keypoints);

        Mat result = new Mat();
        Features2d.drawKeypoints(currentImage, keypoints, result);

        displayImage(result);
    }

    private void detectFAST() {
        if (currentImage == null) return;

        Mat gray = new Mat();
        Imgproc.cvtColor(currentImage, gray, Imgproc.COLOR_BGR2GRAY);

        FastFeatureDetector fast = FastFeatureDetector.create();
        MatOfKeyPoint keypoints = new MatOfKeyPoint();
        fast.detect(gray, keypoints);

        Mat result = new Mat();
        Features2d.drawKeypoints(currentImage, keypoints, result);

        displayImage(result);
    }

    private void displayImage(Mat mat) {
        BufferedImage image = matToBufferedImage(mat);

        // Создаем масштабируемую иконку
        ImageIcon icon = new ImageIcon(image);
        Image scaledImage = icon.getImage().getScaledInstance(
                (int)(image.getWidth() * scaleFactor),
                (int)(image.getHeight() * scaleFactor),
                Image.SCALE_SMOOTH
        );

        imageLabel.setIcon(new ImageIcon(scaledImage));
        imageLabel.revalidate();
        imageLabel.repaint();
    }

    private BufferedImage matToBufferedImage(Mat mat) {
        // Конвертация Mat в BufferedImage
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (mat.channels() > 1) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }

        int bufferSize = mat.channels() * mat.cols() * mat.rows();
        byte[] buffer = new byte[bufferSize];
        mat.get(0, 0, buffer);

        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
        final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(buffer, 0, targetPixels, 0, buffer.length);

        return image;
    }
}